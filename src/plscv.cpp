// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(KODAMA_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

extern "C" void kodama_cuda_lda_label_sums_row_float(const float*, const int*, int, int, int, float*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_row_float(float*, const float*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_pooled_col_float(float*, const float*, const float*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_copy_cov_float(const float*, float*, int, int, cudaStream_t);
extern "C" void kodama_cuda_symmetrize_lower_float(float*, int, cudaStream_t);
extern "C" void kodama_cuda_lda_add_ridge_float(float*, int, float, float*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_to_rhs_float(const float*, float*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_finalize_linear_row_float(const float*, const float*, const float*, float*, float*, int, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_score_argmax_row_float(const float*, const float*, const float*, int*, int, int, int, cudaStream_t);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float(const float*, int, int, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_crossprod(const float*, int, int, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels(const float*, int, int, const int*, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels_gram(const float*, int, int, const float*, const int*, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels_device(
  const float*, int, int, const float*, const int*, const float*, int, int,
  cudaStream_t, const float**
);
extern "C" void kodama_cuda_transpose_pls_weights_float(
  const float*, float*, int, int, cudaStream_t
);
#endif

namespace kodama {

namespace {

#if defined(KODAMA_ENABLE_CUDA)
void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
  }
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cuBLAS call failed");
  }
}

void check_cusolver(cusolverStatus_t code, const char* where) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cuSolver call failed");
  }
}

class CudaBlasContext {
 public:
  explicit CudaBlasContext(int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cublas(cublasCreate(&handle_), "cublasCreate");
  }

  ~CudaBlasContext() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }

  CudaBlasContext(const CudaBlasContext&) = delete;
  CudaBlasContext& operator=(const CudaBlasContext&) = delete;

  cublasHandle_t handle() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

class CudaLDAContext {
 public:
  explicit CudaLDAContext(int device) : device_(device) {
    check_cuda(cudaSetDevice(device_), "cudaSetDevice(CudaLDAContext)");
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate(CudaLDAContext)");
    check_cublas(cublasCreate(&blas_), "cublasCreate(CudaLDAContext)");
    check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream(CudaLDAContext)");
    check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate(CudaLDAContext)");
    check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream(CudaLDAContext)");
  }

  ~CudaLDAContext() {
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  CudaLDAContext(const CudaLDAContext&) = delete;
  CudaLDAContext& operator=(const CudaLDAContext&) = delete;

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  cublasHandle_t blas() const { return blas_; }
  cusolverDnHandle_t solver() const { return solver_; }

 private:
  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
};

CudaLDAContext& cuda_lda_context(int device) {
  thread_local std::unique_ptr<CudaLDAContext> context;
  if (!context || context->device() != device) {
    context = std::make_unique<CudaLDAContext>(device);
  } else {
    check_cuda(cudaSetDevice(device), "cudaSetDevice(cuda_lda_context)");
  }
  return *context;
}

class DeviceIntBuffer {
 public:
  DeviceIntBuffer() = default;
  explicit DeviceIntBuffer(std::size_t n) { reset(n); }

  ~DeviceIntBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceIntBuffer(const DeviceIntBuffer&) = delete;
  DeviceIntBuffer& operator=(const DeviceIntBuffer&) = delete;
  DeviceIntBuffer(DeviceIntBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceIntBuffer& operator=(DeviceIntBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(int)), "cudaMalloc(int)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  int* data() { return ptr_; }
  const int* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  int* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class DeviceFloatBuffer {
 public:
  DeviceFloatBuffer() = default;
  explicit DeviceFloatBuffer(std::size_t n) { reset(n); }

  ~DeviceFloatBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
  DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;
  DeviceFloatBuffer(DeviceFloatBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceFloatBuffer& operator=(DeviceFloatBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(float)), "cudaMalloc(float)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  float* data() { return ptr_; }
  const float* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct CudaPLSLDAFloatWorkspace {
  explicit CudaPLSLDAFloatWorkspace(int dev) : device(dev) {}

  struct ResidentFold {
    std::uint64_t epoch = 0;
    const float* train_host = nullptr;
    const float* validation_host = nullptr;
    const float* train_colmajor_host = nullptr;
    int train_rows = 0;
    int validation_rows = 0;
    int predictors = 0;
    bool ready = false;
    DeviceFloatBuffer train_rowmajor;
    DeviceFloatBuffer validation_rowmajor;
    DeviceFloatBuffer train_colmajor;
    DeviceFloatBuffer train_gram_colmajor;
    bool train_gram_ready = false;
  };

  int device = 0;
  DeviceFloatBuffer x_train;
  DeviceFloatBuffer x_val;
  DeviceFloatBuffer weights;
  DeviceFloatBuffer train_gram;
  DeviceFloatBuffer train_scores;
  DeviceFloatBuffer val_scores;
  DeviceFloatBuffer class_feature_sums;
  DeviceFloatBuffer projected_gram;
  DeviceIntBuffer labels;
  DeviceFloatBuffer counts;
  DeviceFloatBuffer means;
  DeviceFloatBuffer pooled;
  DeviceFloatBuffer cov;
  DeviceFloatBuffer rhs;
  DeviceFloatBuffer linear;
  DeviceFloatBuffer constants;
  DeviceFloatBuffer lambda;
  DeviceIntBuffer info;
  DeviceIntBuffer pred;
  DeviceFloatBuffer solver_work;
  std::vector<int> encoded;
  std::vector<float> class_counts;
  std::vector<float> weights_prefix;
  std::vector<int> pred_codes;
  std::vector<ResidentFold> resident_folds;
  std::uint64_t matrix_uploads = 0;
  std::uint64_t matrix_reuses = 0;
};

CudaPLSLDAFloatWorkspace& cuda_pls_lda_float_workspace(int device) {
  thread_local std::unique_ptr<CudaPLSLDAFloatWorkspace> workspace;
  if (!workspace || workspace->device != device) {
    workspace = std::make_unique<CudaPLSLDAFloatWorkspace>(device);
  }
  return *workspace;
}

class PersistentFoldExecutor {
 public:
  explicit PersistentFoldExecutor(std::size_t worker_count) {
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers_.push_back(std::make_unique<Worker>());
      Worker* worker = workers_.back().get();
      worker->thread = std::thread([worker]() {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->ready.wait(lock, [worker]() {
              return worker->stop || !worker->tasks.empty();
            });
            if (worker->stop && worker->tasks.empty()) return;
            task = std::move(worker->tasks.front());
            worker->tasks.pop_front();
          }
          task();
        }
      });
    }
  }

  ~PersistentFoldExecutor() {
    for (const std::unique_ptr<Worker>& worker : workers_) {
      {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->stop = true;
      }
      worker->ready.notify_one();
    }
    for (const std::unique_ptr<Worker>& worker : workers_) {
      if (worker->thread.joinable()) worker->thread.join();
    }
  }

  PersistentFoldExecutor(const PersistentFoldExecutor&) = delete;
  PersistentFoldExecutor& operator=(const PersistentFoldExecutor&) = delete;

  void run(const std::vector<std::function<void()>>& tasks) {
    if (tasks.empty()) return;
    struct Batch {
      std::mutex mutex;
      std::condition_variable finished;
      std::size_t remaining = 0;
      std::exception_ptr error;
    };
    auto batch = std::make_shared<Batch>();
    batch->remaining = tasks.size();
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      Worker& worker = *workers_[i % workers_.size()];
      std::function<void()> wrapped = [task = tasks[i], batch]() {
        try {
          task();
        } catch (...) {
          std::lock_guard<std::mutex> lock(batch->mutex);
          if (!batch->error) batch->error = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lock(batch->mutex);
          --batch->remaining;
        }
        batch->finished.notify_one();
      };
      {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.tasks.push_back(std::move(wrapped));
      }
      worker.ready.notify_one();
    }
    std::unique_lock<std::mutex> lock(batch->mutex);
    batch->finished.wait(lock, [batch]() { return batch->remaining == 0; });
    if (batch->error) std::rethrow_exception(batch->error);
  }

  std::size_t size() const { return workers_.size(); }

 private:
  struct Worker {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    bool stop = false;
    std::thread thread;
  };
  std::vector<std::unique_ptr<Worker>> workers_;
};

PersistentFoldExecutor& persistent_fold_executor(std::size_t worker_count) {
  thread_local std::unique_ptr<PersistentFoldExecutor> executor;
  if (!executor || executor->size() != worker_count) {
    executor = std::make_unique<PersistentFoldExecutor>(worker_count);
  }
  return *executor;
}

#endif

struct DenseF {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  DenseF() = default;
  DenseF(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r * c), 0.0f) {}

  float& operator()(int i, int j) { return data[static_cast<std::size_t>(i * cols + j)]; }
  float operator()(int i, int j) const { return data[static_cast<std::size_t>(i * cols + j)]; }
};

float matrix_value_float(MatrixView x, std::size_t i, std::size_t j) {
  return x.value_float(i, j);
}

DenseF subset_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  const std::vector<float>& mean,
  const std::vector<float>& scale
) {
  DenseF out(static_cast<int>(rows.size()), static_cast<int>(x.cols));
  for (int i = 0; i < out.rows; ++i) {
    const int src = rows[static_cast<std::size_t>(i)];
    for (int j = 0; j < out.cols; ++j) {
      out(i, j) = (matrix_value_float(x, static_cast<std::size_t>(src), static_cast<std::size_t>(j)) - mean[static_cast<std::size_t>(j)]) /
                  scale[static_cast<std::size_t>(j)];
    }
  }
  return out;
}

std::vector<float> densef_to_colmajor(const DenseF& x) {
  std::vector<float> out(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols));
  for (int j = 0; j < x.cols; ++j) {
    for (int i = 0; i < x.rows; ++i) {
      out[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.rows)] = x(i, j);
    }
  }
  return out;
}

DenseF densef_gram(const DenseF& x) {
  DenseF gram(x.cols, x.cols);
  for (int row = 0; row < x.cols; ++row) {
    for (int col = 0; col <= row; ++col) {
      float value = 0.0f;
      for (int sample = 0; sample < x.rows; ++sample) value += x(sample, row) * x(sample, col);
      gram(row, col) = value;
      gram(col, row) = value;
    }
  }
  return gram;
}

int sorted_class_index(const std::vector<int>& classes, int label) {
  const auto it = std::lower_bound(classes.begin(), classes.end(), label);
  if (it == classes.end() || *it != label) {
    throw std::invalid_argument("label is not present in the active class set.");
  }
  return static_cast<int>(it - classes.begin());
}

template <typename Count>
void encode_labels_from_sorted_classes(
  const std::vector<int>& labels,
  const std::vector<int>& classes,
  std::vector<int>& encoded,
  std::vector<Count>& counts,
  int code_offset
) {
  encoded.resize(labels.size());
  counts.assign(classes.size(), Count{});
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int cls = sorted_class_index(classes, labels[i]);
    encoded[i] = cls + code_offset;
    counts[static_cast<std::size_t>(cls)] += Count{1};
  }
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<float> densef_gram_colmajor_cuda(
  const std::vector<float>& x_colmajor,
  int rows,
  int cols,
  int gpu_device
) {
  if (rows < 1 || cols < 1) return {};
  if (x_colmajor.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    throw std::invalid_argument("densef_gram_colmajor_cuda input size mismatch.");
  }

  CudaBlasContext ctx(gpu_device);
  DeviceFloatBuffer dx(x_colmajor.size());
  DeviceFloatBuffer dg(static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols));
  check_cuda(cudaMemcpy(dx.data(), x_colmajor.data(), x_colmajor.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy raw float32 X for fold Gram");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
      ctx.handle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      cols,
      cols,
      rows,
      &alpha,
      dx.data(),
      rows,
      dx.data(),
      rows,
      &beta,
      dg.data(),
      cols
    ),
    "cublasSgemm raw float32 fold Gram"
  );

  std::vector<float> gram(static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols));
  check_cuda(cudaMemcpy(gram.data(), dg.data(), gram.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy raw float32 fold Gram to host");
  return gram;
}
#endif

void train_center_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<float>& mean,
  std::vector<float>& scale
);

struct PLSFoldDataF {
  int fold = 0;
  std::vector<int> train;
  std::vector<int> validation;
  std::vector<float> mean;
  std::vector<float> scale;
  DenseF x_train;
  DenseF x_val;
  DenseF x_train_gram;
  std::vector<float> x_train_colmajor;
  std::vector<float> x_train_gram_colmajor;
};

struct PLSFoldXCacheF {
  bool valid = false;
  const void* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  MatrixValueType value_type = MatrixValueType::Float64;
  bool center = true;
  bool scale_columns = false;
  int folds = 0;
  std::uint64_t seed = 0;
  std::size_t constrain_hash = 0;
  std::uint64_t data_epoch = 0;
  bool train_colmajor = false;
  bool train_gram = false;
  std::uint64_t generation = 0;
  std::vector<int> fold_assignments;
  std::vector<int> fold_ids;
  std::vector<PLSFoldDataF> folds_data;
};

std::size_t hash_int_vector(const std::vector<int>& values) {
  std::size_t h = 1469598103934665603ull;
  for (int value : values) {
    h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(value));
    h *= 1099511628211ull;
  }
  return h;
}

bool cache_matches_float(
  const PLSFoldXCacheF& cache,
  MatrixView x,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  bool require_train_colmajor
) {
  return cache.valid &&
         cache.data == x.data &&
         cache.rows == x.rows &&
         cache.cols == x.cols &&
         cache.value_type == x.value_type &&
         cache.center == options.center &&
         cache.scale_columns == options.scale &&
         cache.folds == options.cv.folds &&
         cache.seed == options.cv.seed &&
         cache.data_epoch == options.data_epoch &&
         cache.constrain_hash == hash_int_vector(constrain) &&
         (!require_train_colmajor || cache.train_colmajor);
}

const PLSFoldXCacheF& get_pls_fold_x_cache_float(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  bool cache_train_colmajor,
  bool cache_train_gram = false
) {
  thread_local PLSFoldXCacheF cache;
  thread_local std::uint64_t generation_counter = 0;
  if (cache_matches_float(cache, x, constrain, options, cache_train_colmajor)) {
    if (cache_train_gram && !cache.train_gram) {
      for (PLSFoldDataF& fold : cache.folds_data) {
        if (cache_train_colmajor) {
#if defined(KODAMA_ENABLE_CUDA)
          fold.x_train_gram_colmajor = densef_gram_colmajor_cuda(
            fold.x_train_colmajor,
            fold.x_train.rows,
            fold.x_train.cols,
            options.gpu_device
          );
#endif
        } else if (fold.x_train.cols <= fold.x_train.rows) {
          fold.x_train_gram = densef_gram(fold.x_train);
        }
      }
      cache.train_gram = true;
    }
    return cache;
  }

  cache = PLSFoldXCacheF{};
  cache.valid = true;
  cache.data = x.data;
  cache.rows = x.rows;
  cache.cols = x.cols;
  cache.value_type = x.value_type;
  cache.center = options.center;
  cache.scale_columns = options.scale;
  cache.folds = options.cv.folds;
  cache.seed = options.cv.seed;
  cache.constrain_hash = hash_int_vector(constrain);
  cache.data_epoch = options.data_epoch;
  cache.train_colmajor = cache_train_colmajor;
  cache.train_gram = false;
  cache.generation = ++generation_counter;
  cache.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  cache.fold_ids = detail::sorted_unique_folds(cache.fold_assignments);
  cache.folds_data.reserve(cache.fold_ids.size());

  for (int fold : cache.fold_ids) {
    PLSFoldDataF fold_data;
    fold_data.fold = fold;
    fold_data.validation = detail::indices_where_fold(cache.fold_assignments, fold, true);
    fold_data.train = detail::indices_where_fold(cache.fold_assignments, fold, false);
    train_center_scale_float(x, fold_data.train, options.center, options.scale, fold_data.mean, fold_data.scale);
    fold_data.x_train = subset_scale_float(x, fold_data.train, fold_data.mean, fold_data.scale);
    fold_data.x_val = subset_scale_float(x, fold_data.validation, fold_data.mean, fold_data.scale);
    if (cache_train_colmajor) {
      fold_data.x_train_colmajor = densef_to_colmajor(fold_data.x_train);
#if defined(KODAMA_ENABLE_CUDA)
      if (cache_train_gram) {
        fold_data.x_train_gram_colmajor = densef_gram_colmajor_cuda(
          fold_data.x_train_colmajor,
          fold_data.x_train.rows,
          fold_data.x_train.cols,
          options.gpu_device
        );
      }
#endif
    }
    cache.folds_data.push_back(std::move(fold_data));
  }
  return cache;
}

void train_center_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<float>& mean,
  std::vector<float>& scale
) {
  mean.assign(x.cols, 0.0f);
  scale.assign(x.cols, 1.0f);
  if (center) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) mean[j] += matrix_value_float(x, static_cast<std::size_t>(row), j);
    }
    const float inv_n = rows.empty() ? 0.0f : 1.0f / static_cast<float>(rows.size());
    for (float& v : mean) v *= inv_n;
  }
  if (scale_columns) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) {
        const float d = matrix_value_float(x, static_cast<std::size_t>(row), j) - mean[j];
        scale[j] += d * d;
      }
    }
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float s = std::sqrt(scale[j] / std::max(1.0f, static_cast<float>(rows.size() - 1)));
      scale[j] = s > 0.0f && std::isfinite(s) ? s : 1.0f;
    }
  }
}

std::vector<float> centered_label_crossprod_colmajor_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes
) {
  if (static_cast<int>(y_train.size()) != x.rows) throw std::invalid_argument("label-aware float SIMPLS label size mismatch.");
  const int n = x.rows;
  const int p = x.cols;
  const int m = static_cast<int>(classes.size());
  std::vector<float> s(static_cast<std::size_t>(p) * static_cast<std::size_t>(m), 0.0f);
  std::vector<float> x_sum(static_cast<std::size_t>(p), 0.0f);
  std::vector<int> encoded;
  std::vector<float> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);

  for (int i = 0; i < n; ++i) {
    const int cls = encoded[static_cast<std::size_t>(i)];
    for (int j = 0; j < p; ++j) {
      const float value = x(i, j);
      x_sum[static_cast<std::size_t>(j)] += value;
      s[static_cast<std::size_t>(j) + static_cast<std::size_t>(cls) * static_cast<std::size_t>(p)] += value;
    }
  }

  const float inv_n = n > 0 ? 1.0f / static_cast<float>(n) : 0.0f;
  for (int cls = 0; cls < m; ++cls) {
    const float y_mean = counts[static_cast<std::size_t>(cls)] * inv_n;
    float* col = s.data() + static_cast<std::size_t>(cls) * static_cast<std::size_t>(p);
    for (int j = 0; j < p; ++j) col[j] -= y_mean * x_sum[static_cast<std::size_t>(j)];
  }
  return s;
}

DenseF centered_label_crossprod_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes
) {
  const std::vector<float> colmajor = centered_label_crossprod_colmajor_float(x, y_train, classes);
  DenseF out(x.cols, static_cast<int>(classes.size()));
  for (int c = 0; c < out.cols; ++c) {
    for (int j = 0; j < out.rows; ++j) {
      out(j, c) = colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(c) * static_cast<std::size_t>(out.rows)];
    }
  }
  return out;
}

std::vector<int> components_to_evaluate(const PLSOptions& options, int available_components) {
  if (available_components < 1) return {};
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  return {std::min(requested, available_components)};
}

int pls_component_limit(int requested, int rows, int cols) {
  return std::max(1, std::min({
    requested,
    cols,
    std::max(1, rows - 1)
  }));
}

DenseF solve_linear_float(DenseF a, DenseF b);

struct PLSFitF {
  DenseF weights;
  DenseF y_weights;
};

class DegeneratePLSFit final : public std::runtime_error {
 public:
  explicit DegeneratePLSFit(const std::string& message) : std::runtime_error(message) {}
};

float dot_float(const std::vector<float>& a, const std::vector<float>& b) {
  double out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    out += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(out);
}

float norm2_float(const std::vector<float>& x) {
  return std::sqrt(std::max(0.0f, dot_float(x, x)));
}

float power_norm2_float(const std::vector<float>& x) {
  double sum = 0.0;
  for (float value : x) sum += static_cast<double>(value) * static_cast<double>(value);
  if (!std::isfinite(sum) || sum < 0.0) return std::numeric_limits<float>::infinity();
  if (sum <= static_cast<double>(std::numeric_limits<float>::max())) {
    return std::sqrt(static_cast<float>(sum));
  }
  return static_cast<float>(std::sqrt(sum));
}

std::vector<float> mat_vec_float(const DenseF& a, const std::vector<float>& x) {
  std::vector<float> out(static_cast<std::size_t>(a.rows), 0.0f);
  for (int i = 0; i < a.rows; ++i) {
    double s = 0.0;
    for (int j = 0; j < a.cols; ++j) s += static_cast<double>(a(i, j)) * static_cast<double>(x[static_cast<std::size_t>(j)]);
    out[static_cast<std::size_t>(i)] = static_cast<float>(s);
  }
  return out;
}

std::vector<float> t_mat_vec_float(const DenseF& a, const std::vector<float>& x) {
  std::vector<float> out(static_cast<std::size_t>(a.cols), 0.0f);
  for (int j = 0; j < a.cols; ++j) {
    double s = 0.0;
    for (int i = 0; i < a.rows; ++i) s += static_cast<double>(a(i, j)) * static_cast<double>(x[static_cast<std::size_t>(i)]);
    out[static_cast<std::size_t>(j)] = static_cast<float>(s);
  }
  return out;
}

PLSFitF fit_pls_components_from_crossprod_float(
  const DenseF& x,
  DenseF s,
  int max_components,
  const DenseF* x_gram = nullptr
) {
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  const DenseF s0 = s;
  DenseF weights(x.cols, max_rank);
  DenseF y_weights(s.cols, max_rank);
  DenseF v_basis(x.cols, max_rank);
  std::vector<float> previous_weight;
  int fitted_rank = 0;

  const auto power_refresh = [](const DenseF& cross_product, std::vector<float>& candidate) {
    constexpr int power_iterations = 2;
    for (int iteration = 0; iteration < power_iterations; ++iteration) {
      const std::vector<float> response_projection =
        t_mat_vec_float(cross_product, candidate);
      candidate = mat_vec_float(cross_product, response_projection);
    }
  };

  for (int a = 0; a < max_rank; ++a) {
    std::vector<float> weight(static_cast<std::size_t>(x.cols), 0.0f);
    if (!previous_weight.empty()) {
      weight = previous_weight;
    } else {
      std::mt19937 generator(1u);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      for (float& value : weight) value = normal(generator);
    }

    // fastPLS uses a one-vector incremental randomized refresh with two
    // power iterations before each SIMPLS deflation step.
    power_refresh(s, weight);
    float refresh_norm = power_norm2_float(weight);
    if (
      previous_weight.empty() &&
      (!std::isfinite(refresh_norm) || refresh_norm <= 1.0e-10f)
    ) {
      // Retry only a failed first randomized refresh on the response
      // direction carrying the strongest supervised signal.
      int strongest_response = -1;
      double strongest_norm_squared = 0.0;
      for (int response = 0; response < s.cols; ++response) {
        double norm_squared = 0.0;
        for (int feature = 0; feature < s.rows; ++feature) {
          const double value = static_cast<double>(s(feature, response));
          norm_squared += value * value;
        }
        if (std::isfinite(norm_squared) && norm_squared > strongest_norm_squared) {
          strongest_norm_squared = norm_squared;
          strongest_response = response;
        }
      }
      if (strongest_response >= 0 && strongest_norm_squared > 1.0e-20) {
        for (int feature = 0; feature < s.rows; ++feature) {
          weight[static_cast<std::size_t>(feature)] = s(feature, strongest_response);
        }
        power_refresh(s, weight);
        refresh_norm = power_norm2_float(weight);
      }
    }
    if (!std::isfinite(refresh_norm) || refresh_norm <= 1.0e-10f) break;
    for (float& value : weight) value /= refresh_norm;

    std::vector<float> loading;
    if (x_gram != nullptr && x_gram->rows == x.cols && x_gram->cols == x.cols) {
      loading = mat_vec_float(*x_gram, weight);
      const float score_norm_squared = dot_float(weight, loading);
      if (!std::isfinite(score_norm_squared) || score_norm_squared <= 1.0e-20f) break;
      const float inverse_score_norm = 1.0f / std::sqrt(score_norm_squared);
      for (float& value : weight) value *= inverse_score_norm;
      for (float& value : loading) value *= inverse_score_norm;
    } else {
      std::vector<float> scores(static_cast<std::size_t>(x.rows), 0.0f);
      for (int i = 0; i < x.rows; ++i) {
        float value = 0.0f;
        for (int j = 0; j < x.cols; ++j) value += x(i, j) * weight[static_cast<std::size_t>(j)];
        scores[static_cast<std::size_t>(i)] = value;
      }
      const float score_norm = norm2_float(scores);
      if (!std::isfinite(score_norm) || score_norm <= 1.0e-10f) break;
      const float inverse_score_norm = 1.0f / score_norm;
      for (float& value : scores) value *= inverse_score_norm;
      for (float& value : weight) value *= inverse_score_norm;
      loading = t_mat_vec_float(x, scores);
    }
    const std::vector<float> response_weight = t_mat_vec_float(s0, weight);
    previous_weight = weight;

    for (int previous = 0; previous < a; ++previous) {
      float projection = 0.0f;
      for (int j = 0; j < x.cols; ++j) projection += v_basis(j, previous) * loading[static_cast<std::size_t>(j)];
      for (int j = 0; j < x.cols; ++j) loading[static_cast<std::size_t>(j)] -= projection * v_basis(j, previous);
    }
    const float loading_norm = norm2_float(loading);
    if (!std::isfinite(loading_norm) || loading_norm <= 1.0e-10f) break;
    for (float& value : loading) value /= loading_norm;

    for (int j = 0; j < x.cols; ++j) {
      weights(j, a) = weight[static_cast<std::size_t>(j)];
      v_basis(j, a) = loading[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < s.cols; ++j) y_weights(j, a) = response_weight[static_cast<std::size_t>(j)];

    std::vector<float> vs(static_cast<std::size_t>(s.cols), 0.0f);
    for (int c = 0; c < s.cols; ++c) {
      float value = 0.0f;
      for (int j = 0; j < s.rows; ++j) value += loading[static_cast<std::size_t>(j)] * s(j, c);
      vs[static_cast<std::size_t>(c)] = value;
    }
    for (int j = 0; j < s.rows; ++j) {
      for (int c = 0; c < s.cols; ++c) {
        s(j, c) -= loading[static_cast<std::size_t>(j)] * vs[static_cast<std::size_t>(c)];
      }
    }
    fitted_rank = a + 1;
  }

  if (fitted_rank < 1) {
    throw DegeneratePLSFit(
      "CPU float32 SIMPLS found no supervised latent direction."
    );
  }
  if (fitted_rank == max_rank) return PLSFitF{std::move(weights), std::move(y_weights)};

  DenseF fitted_weights(x.cols, fitted_rank);
  DenseF fitted_y_weights(s.cols, fitted_rank);
  for (int component = 0; component < fitted_rank; ++component) {
    for (int feature = 0; feature < x.cols; ++feature) {
      fitted_weights(feature, component) = weights(feature, component);
    }
    for (int response = 0; response < s.cols; ++response) {
      fitted_y_weights(response, component) = y_weights(response, component);
    }
  }
  return PLSFitF{std::move(fitted_weights), std::move(fitted_y_weights)};
}

PLSFitF fit_pls_components_labels_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components,
  const DenseF* x_gram = nullptr
) {
  const int max_rank = pls_component_limit(max_components, x.rows, x.cols);
  return fit_pls_components_from_crossprod_float(
    x,
    centered_label_crossprod_float(x, y_train, classes),
    max_rank,
    x_gram
  );
}

PLSFitF fit_pls_components_labels_metal_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components
) {
  const DenseF cross_product = centered_label_crossprod_float(x, y_train, classes);
  const detail::MetalSIMPLSResult fit = detail::metal_simpls_fit(
    x.data,
    x.rows,
    x.cols,
    cross_product.data,
    cross_product.cols,
    max_components
  );
  DenseF weights(fit.predictors, fit.components);
  weights.data = fit.weights;
  DenseF y_weights(fit.responses, fit.components);
  y_weights.data = fit.y_weights;
  return PLSFitF{std::move(weights), std::move(y_weights)};
}

DenseF transform_pls_scores_float(const DenseF& x, const PLSFitF& fit, int ncomp) {
  DenseF out(x.rows, ncomp);
  for (int i = 0; i < x.rows; ++i) {
    float* out_row = out.data.data() + static_cast<std::size_t>(i) * ncomp;
    for (int j = 0; j < x.cols; ++j) {
      const float value = x(i, j);
      const float* weight_row = fit.weights.data.data() +
        static_cast<std::size_t>(j) * fit.weights.cols;
      for (int a = 0; a < ncomp; ++a) {
        out_row[a] += value * weight_row[a];
      }
    }
  }
  return out;
}

DenseF transform_pls_scores_metal_float(const DenseF& x, const PLSFitF& fit, int ncomp) {
  if (ncomp < 1 || ncomp > fit.weights.cols || x.cols != fit.weights.rows) {
    throw std::invalid_argument("Metal PLS score projection dimension mismatch.");
  }
  std::vector<float> prefix;
  const std::vector<float>* weights = &fit.weights.data;
  if (ncomp != fit.weights.cols) {
    prefix.resize(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
    for (int row = 0; row < fit.weights.rows; ++row) {
      std::copy_n(
        fit.weights.data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(fit.weights.cols),
        ncomp,
        prefix.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(ncomp)
      );
    }
    weights = &prefix;
  }
  DenseF out(x.rows, ncomp);
  out.data = detail::metal_matrix_multiply(
    x.data,
    x.rows,
    x.cols,
    *weights,
    fit.weights.rows,
    ncomp,
    false,
    false,
    true
  );
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
DenseF transform_pls_scores_cuda_float(const DenseF& x, const PLSFitF& fit, int ncomp, int gpu_device) {
  if (ncomp < 1) return DenseF(x.rows, 0);
  if (ncomp > fit.weights.cols) throw std::invalid_argument("transform_pls_scores_cuda_float ncomp exceeds fit rank.");
  if (x.cols != fit.weights.rows) throw std::invalid_argument("transform_pls_scores_cuda_float column mismatch.");

  CudaBlasContext ctx(gpu_device);
  DeviceFloatBuffer dx(x.data.size());
  DeviceFloatBuffer dw(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  DeviceFloatBuffer dt(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(ncomp));

  std::vector<float> w_prefix(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  for (int i = 0; i < fit.weights.rows; ++i) {
    for (int j = 0; j < ncomp; ++j) {
      w_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  check_cuda(cudaMemcpy(dx.data(), x.data.data(), x.data.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy float score X to device");
  check_cuda(cudaMemcpy(dw.data(), w_prefix.data(), w_prefix.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy float score W to device");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
      ctx.handle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      ncomp,
      x.rows,
      x.cols,
      &alpha,
      dw.data(),
      ncomp,
      dx.data(),
      x.cols,
      &beta,
      dt.data(),
      ncomp
    ),
    "cublasSgemm float32 XW"
  );

  DenseF out(x.rows, ncomp);
  check_cuda(cudaMemcpy(out.data.data(), dt.data(), out.data.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy float scores to host");
  return out;
}

PLSFitF fit_pls_components_cuda_colmajor_y_float(
  const DenseF& x,
  const float* y_colmajor,
  int y_cols,
  int max_components,
  int gpu_device
) {
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(fit_pls_components_cuda_float)");
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  std::vector<float> x_colmajor(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols));
  for (int j = 0; j < x.cols; ++j) {
    for (int i = 0; i < x.rows; ++i) {
      x_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.rows)] = x(i, j);
    }
  }

  std::vector<float> rr_colmajor(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(max_rank), 0.0f);
  std::vector<float> qq_colmajor(static_cast<std::size_t>(y_cols) * static_cast<std::size_t>(max_rank), 0.0f);
  const bool ok = kodama_fastpls_simpls_fit_cuda_float(
    x_colmajor.data(),
    x.rows,
    x.cols,
    y_colmajor,
    y_cols,
    max_rank,
    1,
    rr_colmajor.data(),
    qq_colmajor.data()
  );
  if (!ok) throw std::runtime_error("fastPLS CUDA float32 SIMPLS fit failed.");

  PLSFitF fit{DenseF(x.cols, max_rank), DenseF(y_cols, max_rank)};
  for (int a = 0; a < max_rank; ++a) {
    for (int j = 0; j < x.cols; ++j) {
      fit.weights(j, a) = rr_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(x.cols)];
    }
    for (int j = 0; j < y_cols; ++j) {
      fit.y_weights(j, a) = qq_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(y_cols)];
    }
  }
  return fit;
}

PLSFitF fit_pls_components_cuda_labels_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components,
  int gpu_device,
  const float* x_colmajor_precomputed = nullptr,
  const float* x_gram_colmajor_precomputed = nullptr
) {
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(fit_pls_components_cuda_labels_float)");
  const int y_cols = static_cast<int>(classes.size());
  const int max_rank = pls_component_limit(max_components, x.rows, x.cols);
  std::vector<float> x_colmajor_storage;
  const float* x_colmajor = x_colmajor_precomputed;
  if (x_colmajor == nullptr) {
    x_colmajor_storage = densef_to_colmajor(x);
    x_colmajor = x_colmajor_storage.data();
  }
  if (static_cast<int>(y_train.size()) != x.rows) throw std::invalid_argument("CUDA label-aware float32 SIMPLS label size mismatch.");
  std::vector<int> encoded;
  std::vector<float> class_counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, class_counts, 1);

  int fitted_rank = 0;
  std::vector<float> rr_colmajor;
  std::vector<float> qq_colmajor;
  for (int trial_rank = max_rank; trial_rank >= 1; --trial_rank) {
    rr_colmajor.assign(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(trial_rank), 0.0f);
    qq_colmajor.assign(static_cast<std::size_t>(y_cols) * static_cast<std::size_t>(trial_rank), 0.0f);
    const bool ok = x_gram_colmajor_precomputed != nullptr ?
      kodama_fastpls_simpls_fit_cuda_float_labels_gram(
        x_colmajor,
        x.rows,
        x.cols,
        x_gram_colmajor_precomputed,
        encoded.data(),
        class_counts.data(),
        y_cols,
        trial_rank,
        1,
        rr_colmajor.data(),
        qq_colmajor.data()
      ) :
      kodama_fastpls_simpls_fit_cuda_float_labels(
        x_colmajor,
        x.rows,
        x.cols,
        encoded.data(),
        class_counts.data(),
        y_cols,
        trial_rank,
        1,
        rr_colmajor.data(),
        qq_colmajor.data()
      );
    if (ok) {
      fitted_rank = trial_rank;
      break;
    }
  }
  if (fitted_rank < 1) throw std::runtime_error("fastPLS CUDA label-aware float32 SIMPLS fit failed.");

  PLSFitF fit{DenseF(x.cols, fitted_rank), DenseF(y_cols, fitted_rank)};
  for (int a = 0; a < fitted_rank; ++a) {
    for (int j = 0; j < x.cols; ++j) {
      fit.weights(j, a) = rr_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(x.cols)];
    }
    for (int j = 0; j < y_cols; ++j) {
      fit.y_weights(j, a) = qq_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(y_cols)];
    }
  }
  return fit;
}

#endif

std::vector<float> label_means_float(
  const std::vector<int>& labels,
  const std::vector<int>& classes
) {
  std::vector<int> encoded;
  std::vector<float> y_mean;
  encode_labels_from_sorted_classes(labels, classes, encoded, y_mean, 0);
  const float inv_n = labels.empty() ? 0.0f : 1.0f / static_cast<float>(labels.size());
  for (float& value : y_mean) value *= inv_n;
  return y_mean;
}

DenseF regression_coefficients_centered_one_hot_float(
  const DenseF& t,
  const std::vector<int>& labels,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int ncomp
) {
  if (static_cast<int>(labels.size()) != t.rows) throw std::invalid_argument("float32 PLS-DA label size mismatch.");
  const int cnum = static_cast<int>(classes.size());
  DenseF lhs(ncomp, ncomp);
  DenseF rhs(ncomp, cnum);
  std::vector<int> encoded;
  std::vector<float> class_counts;
  encode_labels_from_sorted_classes(labels, classes, encoded, class_counts, 0);
  std::vector<double> score_sums(static_cast<std::size_t>(ncomp), 0.0);
  std::vector<double> lhs_accum(static_cast<std::size_t>(ncomp) * static_cast<std::size_t>(ncomp), 0.0);

  for (int r = 0; r < t.rows; ++r) {
    const int c = encoded[static_cast<std::size_t>(r)];
    const float* t_row = t.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(t.cols);
    for (int a = 0; a < ncomp; ++a) {
      const double value = t_row[a];
      rhs(a, c) += static_cast<float>(value);
      score_sums[static_cast<std::size_t>(a)] += value;
      for (int b = a; b < ncomp; ++b) {
        lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)] +=
          value * static_cast<double>(t_row[b]);
      }
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    for (int b = a; b < ncomp; ++b) {
      const float value = static_cast<float>(lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)]);
      lhs(a, b) = value;
      lhs(b, a) = value;
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    const double sum = score_sums[static_cast<std::size_t>(a)];
    for (int c = 0; c < cnum; ++c) rhs(a, c) -= static_cast<float>(sum * static_cast<double>(y_mean[static_cast<std::size_t>(c)]));
  }

  return solve_linear_float(lhs, rhs);
}

std::vector<int> predict_pls_da_labels_float(
  const DenseF& t_train,
  const std::vector<int>& y_train_labels,
  const DenseF& t_val,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int ncomp
) {
  const DenseF coef = regression_coefficients_centered_one_hot_float(t_train, y_train_labels, classes, y_mean, ncomp);
  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
      float score = c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0f;
      for (int a = 0; a < ncomp; ++a) {
        score += t_val(i, a) * coef(a, c);
      }
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<int> predict_pls_da_cuda_float(
  const DenseF& t_train,
  const std::vector<int>& y_train_labels,
  const DenseF& t_val,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int gpu_device
) {
  if (t_val.rows < 1) return {};
  const int n = t_val.rows;
  const int kk = t_val.cols;
  const int cnum = static_cast<int>(classes.size());
  const DenseF coef = regression_coefficients_centered_one_hot_float(t_train, y_train_labels, classes, y_mean, kk);

  std::vector<float> linear(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(kk), 0.0f);
  std::vector<float> constants(static_cast<std::size_t>(cnum), 0.0f);
  for (int c = 0; c < cnum; ++c) {
    constants[static_cast<std::size_t>(c)] =
      c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0f;
    for (int a = 0; a < kk; ++a) {
      linear[static_cast<std::size_t>(c) * static_cast<std::size_t>(kk) + static_cast<std::size_t>(a)] = coef(a, c);
    }
  }

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(predict_pls_da_cuda_float)");
  cudaStream_t stream = nullptr;
  DeviceFloatBuffer d_t(t_val.data.size());
  DeviceFloatBuffer d_linear(linear.size());
  DeviceFloatBuffer d_constants(constants.size());
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n));
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(predict_pls_da_cuda_float)");
  check_cuda(cudaMemcpyAsync(d_t.data(), t_val.data.data(), t_val.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA T");
  check_cuda(cudaMemcpyAsync(d_linear.data(), linear.data(), linear.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA linear");
  check_cuda(cudaMemcpyAsync(d_constants.data(), constants.data(), constants.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA constants");
  kodama_cuda_lda_score_argmax_row_float(d_t.data(), d_linear.data(), d_constants.data(), d_pred.data(), n, kk, cnum, stream);
  check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row_float PLS-DA");
  std::vector<int> codes(static_cast<std::size_t>(n), 1);
  check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA float32 PLS-DA labels");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA float32 PLS-DA predict");
  cudaStreamDestroy(stream);

  std::vector<int> pred(static_cast<std::size_t>(n), classes.front());
  for (int i = 0; i < n; ++i) {
    const int cls = std::max(1, std::min(cnum, codes[static_cast<std::size_t>(i)])) - 1;
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
  }
  return pred;
}
#endif

DenseF solve_linear_float(DenseF a, DenseF b) {
  const int n = a.rows;
  const int m = b.cols;
  for (int i = 0; i < n; ++i) a(i, i) += 1.0e-9f;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    float best = std::abs(a(col, col));
    for (int r = col + 1; r < n; ++r) {
      const float val = std::abs(a(r, col));
      if (val > best) {
        best = val;
        pivot = r;
      }
    }
    if (best < 1.0e-12f) continue;
    if (pivot != col) {
      for (int j = 0; j < n; ++j) std::swap(a(col, j), a(pivot, j));
      for (int j = 0; j < m; ++j) std::swap(b(col, j), b(pivot, j));
    }
    const float div = a(col, col);
    for (int j = col; j < n; ++j) a(col, j) /= div;
    for (int j = 0; j < m; ++j) b(col, j) /= div;
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const float f = a(r, col);
      if (f == 0.0f) continue;
      for (int j = col; j < n; ++j) a(r, j) -= f * a(col, j);
      for (int j = 0; j < m; ++j) b(r, j) -= f * b(col, j);
    }
  }
  return b;
}

bool cholesky_solve_float(const DenseF& matrix, const DenseF& rhs, DenseF& solution) {
  if (matrix.rows != matrix.cols || rhs.rows != matrix.rows) return false;
  const int n = matrix.rows;
  DenseF lower(n, n);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col <= row; ++col) {
      float value = matrix(row, col);
      for (int k = 0; k < col; ++k) value -= lower(row, k) * lower(col, k);
      if (row == col) {
        if (!std::isfinite(value) || value <= 0.0f) return false;
        lower(row, col) = std::sqrt(value);
      } else {
        const float diagonal = lower(col, col);
        if (!std::isfinite(diagonal) || diagonal <= 0.0f) return false;
        lower(row, col) = value / diagonal;
      }
    }
  }

  solution = rhs;
  for (int column = 0; column < rhs.cols; ++column) {
    for (int row = 0; row < n; ++row) {
      float value = solution(row, column);
      for (int k = 0; k < row; ++k) value -= lower(row, k) * solution(k, column);
      solution(row, column) = value / lower(row, row);
    }
    for (int row = n - 1; row >= 0; --row) {
      float value = solution(row, column);
      for (int k = row + 1; k < n; ++k) value -= lower(k, row) * solution(k, column);
      solution(row, column) = value / lower(row, row);
    }
  }
  return true;
}

struct LDAFloatModel {
  DenseF linear;
  std::vector<float> constants;
};

LDAFloatModel fit_lda_from_score_statistics_float(
  DenseF class_sums,
  const std::vector<int>& counts,
  const DenseF& score_crossprod,
  int n_train
) {
  const int cnum = class_sums.rows;
  const int ncomp = class_sums.cols;
  for (int c = 0; c < cnum; ++c) {
    const float inv = counts[static_cast<std::size_t>(c)] > 0 ?
      1.0f / static_cast<float>(counts[static_cast<std::size_t>(c)]) : 0.0f;
    for (int a = 0; a < ncomp; ++a) class_sums(c, a) *= inv;
  }

  DenseF pooled(ncomp, ncomp);
  const float df = static_cast<float>(std::max(1, n_train - cnum));
  float trace = 0.0f;
  for (int r = 0; r < ncomp; ++r) {
    for (int col = 0; col <= r; ++col) {
      float between = 0.0f;
      for (int c = 0; c < cnum; ++c) {
        between += static_cast<float>(counts[static_cast<std::size_t>(c)]) *
          class_sums(c, r) * class_sums(c, col);
      }
      const float covariance = (score_crossprod(r, col) - between) / df;
      pooled(r, col) = covariance;
      pooled(col, r) = covariance;
    }
    trace += pooled(r, r);
  }

  DenseF rhs(ncomp, cnum);
  for (int c = 0; c < cnum; ++c) {
    for (int a = 0; a < ncomp; ++a) rhs(a, c) = class_sums(c, a);
  }
  const float ridge_scale = std::isfinite(trace) && trace > 0.0f ?
    trace / static_cast<float>(std::max(1, ncomp)) : 1.0f;
  const float ridge_grid[] = {1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
  DenseF solved;
  bool factorized = false;
  for (float ridge : ridge_grid) {
    DenseF covariance = pooled;
    const float lambda = ridge * ridge_scale;
    for (int component = 0; component < ncomp; ++component) {
      covariance(component, component) += lambda;
    }
    if (cholesky_solve_float(covariance, rhs, solved)) {
      factorized = true;
      break;
    }
  }
  if (!factorized) throw std::runtime_error("Float32 PLS-LDA Cholesky factorization failed.");

  LDAFloatModel model{DenseF(cnum, ncomp), std::vector<float>(static_cast<std::size_t>(cnum), 0.0f)};
  for (int c = 0; c < cnum; ++c) {
    float dot_mu = 0.0f;
    for (int a = 0; a < ncomp; ++a) {
      model.linear(c, a) = solved(a, c);
      dot_mu += class_sums(c, a) * model.linear(c, a);
    }
    const float prior = std::max(
      static_cast<float>(counts[static_cast<std::size_t>(c)]) /
        static_cast<float>(std::max(1, n_train)),
      std::numeric_limits<float>::min()
    );
    model.constants[static_cast<std::size_t>(c)] = -0.5f * dot_mu + std::log(prior);
  }
  return model;
}

std::vector<int> predict_lda_scores_float(
  const DenseF& scores,
  const LDAFloatModel& model,
  const std::vector<int>& classes,
  int ncomp
) {
  std::vector<int> pred(static_cast<std::size_t>(scores.rows), classes.front());
  for (int i = 0; i < scores.rows; ++i) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
      float score = model.constants[static_cast<std::size_t>(c)];
      for (int a = 0; a < ncomp; ++a) score += scores(i, a) * model.linear(c, a);
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

std::vector<int> predict_pls_lda_float(
  const DenseF& t_train,
  const std::vector<int>& y_train,
  const DenseF& t_val,
  const std::vector<int>& classes,
  int ncomp
) {
  const int cnum = static_cast<int>(classes.size());
  std::vector<int> encoded;
  std::vector<int> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);
  DenseF class_sums(cnum, ncomp);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = encoded[static_cast<std::size_t>(i)];
    for (int a = 0; a < ncomp; ++a) class_sums(c, a) += t_train(i, a);
  }

  DenseF score_crossprod(ncomp, ncomp);
  for (int r = 0; r < ncomp; ++r) {
    for (int col = 0; col <= r; ++col) {
      float total = 0.0f;
      for (int i = 0; i < t_train.rows; ++i) total += t_train(i, r) * t_train(i, col);
      score_crossprod(r, col) = total;
      score_crossprod(col, r) = total;
    }
  }
  const LDAFloatModel model = fit_lda_from_score_statistics_float(
    std::move(class_sums), counts, score_crossprod, t_train.rows
  );
  return predict_lda_scores_float(t_val, model, classes, ncomp);
}

std::vector<int> fit_predict_pls_lda_streamed_cpu_float(
  const DenseF& x_train,
  const std::vector<int>& y_train,
  const DenseF& x_val,
  const PLSFitF& fit,
  const std::vector<int>& classes,
  int ncomp
) {
  const int cnum = static_cast<int>(classes.size());
  std::vector<int> encoded;
  std::vector<int> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);
  DenseF class_sums(cnum, ncomp);
  DenseF score_crossprod(ncomp, ncomp);
  std::vector<float> score_row(static_cast<std::size_t>(ncomp), 0.0f);

  for (int i = 0; i < x_train.rows; ++i) {
    std::fill(score_row.begin(), score_row.end(), 0.0f);
    for (int j = 0; j < x_train.cols; ++j) {
      const float value = x_train(i, j);
      const float* weight_row = fit.weights.data.data() +
        static_cast<std::size_t>(j) * static_cast<std::size_t>(fit.weights.cols);
      for (int a = 0; a < ncomp; ++a) score_row[static_cast<std::size_t>(a)] += value * weight_row[a];
    }
    const int cls = encoded[static_cast<std::size_t>(i)];
    for (int a = 0; a < ncomp; ++a) class_sums(cls, a) += score_row[static_cast<std::size_t>(a)];
    for (int r = 0; r < ncomp; ++r) {
      for (int col = 0; col <= r; ++col) {
        score_crossprod(r, col) +=
          score_row[static_cast<std::size_t>(r)] * score_row[static_cast<std::size_t>(col)];
      }
    }
  }
  for (int r = 0; r < ncomp; ++r) {
    for (int col = 0; col < r; ++col) score_crossprod(col, r) = score_crossprod(r, col);
  }

  const LDAFloatModel model = fit_lda_from_score_statistics_float(
    std::move(class_sums), counts, score_crossprod, x_train.rows
  );
  std::vector<int> pred(static_cast<std::size_t>(x_val.rows), classes.front());
  for (int i = 0; i < x_val.rows; ++i) {
    std::fill(score_row.begin(), score_row.end(), 0.0f);
    for (int j = 0; j < x_val.cols; ++j) {
      const float value = x_val(i, j);
      const float* weight_row = fit.weights.data.data() +
        static_cast<std::size_t>(j) * static_cast<std::size_t>(fit.weights.cols);
      for (int a = 0; a < ncomp; ++a) score_row[static_cast<std::size_t>(a)] += value * weight_row[a];
    }
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cnum; ++c) {
      float score = model.constants[static_cast<std::size_t>(c)];
      for (int a = 0; a < ncomp; ++a) {
        score += score_row[static_cast<std::size_t>(a)] * model.linear(c, a);
      }
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

std::vector<float> packed_pls_weight_prefix_float(const PLSFitF& fit, int ncomp) {
  if (ncomp < 1 || ncomp > fit.weights.cols) {
    throw std::invalid_argument("PLS weight prefix is outside the fitted component range.");
  }
  if (ncomp == fit.weights.cols) return fit.weights.data;
  std::vector<float> prefix(
    static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp),
    0.0f
  );
  for (int row = 0; row < fit.weights.rows; ++row) {
    std::copy_n(
      fit.weights.data.data() +
        static_cast<std::size_t>(row) * static_cast<std::size_t>(fit.weights.cols),
      ncomp,
      prefix.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(ncomp)
    );
  }
  return prefix;
}

std::vector<int> fit_predict_pls_lda_metal_float(
  const DenseF& x_train,
  const std::vector<int>& y_train,
  const DenseF& x_val,
  const PLSFitF& fit,
  const std::vector<int>& classes,
  int ncomp
) {
  std::vector<int> encoded;
  std::vector<int> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);
  const std::vector<float> weights = packed_pls_weight_prefix_float(fit, ncomp);
  const detail::MetalPLSScoreStatistics statistics = detail::metal_pls_score_statistics(
    x_train.data,
    x_train.rows,
    x_train.cols,
    weights,
    ncomp,
    encoded,
    static_cast<int>(classes.size())
  );
  DenseF class_sums(statistics.classes, statistics.components);
  class_sums.data = statistics.class_sums;
  DenseF score_crossprod(statistics.components, statistics.components);
  score_crossprod.data = statistics.score_crossprod;
  const LDAFloatModel model = fit_lda_from_score_statistics_float(
    std::move(class_sums), counts, score_crossprod, x_train.rows
  );
  return detail::metal_pls_lda_predict(
    x_val.data,
    x_val.rows,
    x_val.cols,
    weights,
    ncomp,
    model.linear.data,
    model.constants,
    classes
  );
}

#if defined(KODAMA_ENABLE_CUDA)
CudaPLSLDAFloatWorkspace::ResidentFold& prepare_cuda_resident_pls_fold(
  CudaPLSLDAFloatWorkspace& workspace,
  std::size_t fold_slot,
  std::uint64_t epoch,
  const DenseF& x_train,
  const DenseF& x_val,
  const std::vector<float>& x_train_colmajor,
  cudaStream_t stream
) {
  if (workspace.resident_folds.size() <= fold_slot) {
    workspace.resident_folds.resize(fold_slot + 1);
  }
  CudaPLSLDAFloatWorkspace::ResidentFold& resident =
    workspace.resident_folds[fold_slot];
  const bool matches =
    resident.epoch == epoch &&
    resident.train_host == x_train.data.data() &&
    resident.validation_host == x_val.data.data() &&
    resident.train_colmajor_host == x_train_colmajor.data() &&
    resident.train_rows == x_train.rows &&
    resident.validation_rows == x_val.rows &&
    resident.predictors == x_train.cols &&
    resident.ready;
  if (matches) {
    ++workspace.matrix_reuses;
    return resident;
  }

  const std::size_t train_items = x_train.data.size();
  const std::size_t validation_items = x_val.data.size();
  resident.train_rowmajor.ensure(train_items);
  resident.validation_rowmajor.ensure(validation_items);
  resident.train_colmajor.ensure(x_train_colmajor.size());
  check_cuda(
    cudaMemcpyAsync(
      resident.train_rowmajor.data(),
      x_train.data.data(),
      train_items * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ),
    "cudaMemcpyAsync resident float32 PLS-LDA Xtrain"
  );
  check_cuda(
    cudaMemcpyAsync(
      resident.validation_rowmajor.data(),
      x_val.data.data(),
      validation_items * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ),
    "cudaMemcpyAsync resident float32 PLS-LDA Xvalidation"
  );
  check_cuda(
    cudaMemcpyAsync(
      resident.train_colmajor.data(),
      x_train_colmajor.data(),
      x_train_colmajor.size() * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ),
    "cudaMemcpyAsync resident float32 PLS-LDA Xtrain column-major"
  );
  check_cuda(
    cudaStreamSynchronize(stream),
    "cudaStreamSynchronize resident PLS-LDA fold upload"
  );
  resident.epoch = epoch;
  resident.train_host = x_train.data.data();
  resident.validation_host = x_val.data.data();
  resident.train_colmajor_host = x_train_colmajor.data();
  resident.train_rows = x_train.rows;
  resident.validation_rows = x_val.rows;
  resident.predictors = x_train.cols;
  resident.train_gram_ready = false;
  resident.ready = true;
  ++workspace.matrix_uploads;
  return resident;
}

std::vector<int> predict_pls_lda_device_float(
  const float* x_train_device,
  const float* x_val_device,
  const float* train_gram_colmajor_device,
  const float* weights_rowmajor_device,
  int n,
  int p,
  int n_val,
  int k,
  const std::vector<int>& classes,
  CudaPLSLDAFloatWorkspace& workspace,
  CudaLDAContext& context
) {
  const int cnum = static_cast<int>(classes.size());
  cudaStream_t stream = context.stream();
  cublasHandle_t blas = context.blas();
  cusolverDnHandle_t solver = context.solver();
  workspace.val_scores.ensure(static_cast<std::size_t>(n_val) * static_cast<std::size_t>(k));
  workspace.class_feature_sums.ensure(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(p));
  workspace.projected_gram.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(p));
  workspace.means.ensure(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  workspace.pooled.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  workspace.cov.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  workspace.rhs.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(cnum));
  workspace.linear.ensure(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  workspace.constants.ensure(static_cast<std::size_t>(cnum));
  workspace.lambda.ensure(1);
  workspace.info.ensure(1);
  workspace.pred.ensure(static_cast<std::size_t>(n_val));

  const float one = 1.0f;
  const float zero = 0.0f;
  // LDA only needs class score sums and T'T from the training scores T=XW.
  // Compute them algebraically as (class sums of X)W and W'(X'X)W so the
  // sample-sized training score matrix never has to be materialized.
  kodama_cuda_lda_label_sums_row_float(
    x_train_device,
    workspace.labels.data(),
    n,
    p,
    cnum,
    workspace.class_feature_sums.data(),
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA feature class sums");
  check_cublas(
    cublasSgemm(
      blas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      k,
      cnum,
      p,
      &one,
      weights_rowmajor_device,
      k,
      workspace.class_feature_sums.data(),
      p,
      &zero,
      workspace.means.data(),
      k
    ),
    "cublasSgemm resident float32 PLS-LDA class score sums"
  );
  check_cublas(
    cublasSgemm(
      blas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      k,
      n_val,
      p,
      &one,
      weights_rowmajor_device,
      k,
      x_val_device,
      p,
      &zero,
      workspace.val_scores.data(),
      k
    ),
    "cublasSgemm resident float32 PLS-LDA validation scores"
  );

  kodama_cuda_lda_means_row_float(
    workspace.means.data(),
    workspace.counts.data(),
    k,
    cnum,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA means");
  check_cublas(
    cublasSgemm(
      blas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      k,
      p,
      p,
      &one,
      weights_rowmajor_device,
      k,
      train_gram_colmajor_device,
      p,
      &zero,
      workspace.projected_gram.data(),
      k
    ),
    "cublasSgemm resident float32 CUDA LDA WtXtX"
  );
  check_cublas(
    cublasSgemm(
      blas,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      k,
      k,
      p,
      &one,
      workspace.projected_gram.data(),
      k,
      weights_rowmajor_device,
      k,
      &zero,
      workspace.pooled.data(),
      k
    ),
    "cublasSgemm resident float32 CUDA LDA WtXtXW"
  );
  kodama_cuda_lda_pooled_col_float(
    workspace.pooled.data(),
    workspace.means.data(),
    workspace.counts.data(),
    n,
    k,
    cnum,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA pooled covariance");
  kodama_cuda_lda_means_to_rhs_float(
    workspace.means.data(),
    workspace.rhs.data(),
    k,
    k,
    cnum,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA means RHS");

  int lwork = 0;
  check_cusolver(
    cusolverDnSpotrf_bufferSize(
      solver,
      CUBLAS_FILL_MODE_LOWER,
      k,
      workspace.cov.data(),
      k,
      &lwork
    ),
    "cusolverDnSpotrf_bufferSize resident float32 PLS-LDA"
  );
  workspace.solver_work.ensure(static_cast<std::size_t>(std::max(lwork, 1)));
  int info = 0;
  bool factorized = false;
  const float ridge_grid[] = {1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
  for (float ridge : ridge_grid) {
    kodama_cuda_lda_copy_cov_float(
      workspace.pooled.data(),
      workspace.cov.data(),
      k,
      k,
      stream
    );
    check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA covariance copy");
    kodama_cuda_lda_add_ridge_float(
      workspace.cov.data(),
      k,
      ridge,
      workspace.lambda.data(),
      stream
    );
    check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA ridge");
    check_cusolver(
      cusolverDnSpotrf(
        solver,
        CUBLAS_FILL_MODE_LOWER,
        k,
        workspace.cov.data(),
        k,
        workspace.solver_work.data(),
        lwork,
        workspace.info.data()
      ),
      "cusolverDnSpotrf resident float32 PLS-LDA"
    );
    check_cuda(
      cudaMemcpyAsync(
        &info,
        workspace.info.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
      ),
      "cudaMemcpyAsync resident float32 PLS-LDA potrf info"
    );
    check_cuda(
      cudaStreamSynchronize(stream),
      "cudaStreamSynchronize resident float32 PLS-LDA potrf"
    );
    if (info == 0) {
      factorized = true;
      break;
    }
  }
  if (!factorized) {
    throw std::runtime_error(
      "cusolverDnSpotrf resident float32 PLS-LDA returned non-zero info."
    );
  }
  check_cusolver(
    cusolverDnSpotrs(
      solver,
      CUBLAS_FILL_MODE_LOWER,
      k,
      cnum,
      workspace.cov.data(),
      k,
      workspace.rhs.data(),
      k,
      workspace.info.data()
    ),
    "cusolverDnSpotrs resident float32 PLS-LDA"
  );
  check_cuda(
    cudaMemcpyAsync(
      &info,
      workspace.info.data(),
      sizeof(int),
      cudaMemcpyDeviceToHost,
      stream
    ),
    "cudaMemcpyAsync resident float32 PLS-LDA potrs info"
  );
  kodama_cuda_lda_finalize_linear_row_float(
    workspace.rhs.data(),
    workspace.means.data(),
    workspace.counts.data(),
    workspace.linear.data(),
    workspace.constants.data(),
    n,
    k,
    k,
    cnum,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA finalize");
  check_cuda(
    cudaStreamSynchronize(stream),
    "cudaStreamSynchronize resident float32 PLS-LDA solve"
  );
  if (info != 0) {
    throw std::runtime_error(
      "cusolverDnSpotrs resident float32 PLS-LDA returned non-zero info."
    );
  }
  kodama_cuda_lda_score_argmax_row_float(
    workspace.val_scores.data(),
    workspace.linear.data(),
    workspace.constants.data(),
    workspace.pred.data(),
    n_val,
    k,
    cnum,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS-LDA prediction");
  workspace.pred_codes.assign(static_cast<std::size_t>(n_val), 1);
  check_cuda(
    cudaMemcpyAsync(
      workspace.pred_codes.data(),
      workspace.pred.data(),
      workspace.pred_codes.size() * sizeof(int),
      cudaMemcpyDeviceToHost,
      stream
    ),
    "cudaMemcpyAsync resident float32 PLS-LDA labels"
  );
  check_cuda(
    cudaStreamSynchronize(stream),
    "cudaStreamSynchronize resident float32 PLS-LDA predict"
  );

  std::vector<int> pred(static_cast<std::size_t>(n_val), classes.front());
  for (int i = 0; i < n_val; ++i) {
    const int cls =
      std::max(1, std::min(cnum, workspace.pred_codes[static_cast<std::size_t>(i)])) - 1;
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
  }
  return pred;
}

std::vector<int> fit_predict_pls_lda_resident_cuda_float(
  const DenseF& x_train,
  const std::vector<float>& x_train_colmajor,
  const std::vector<int>& y_train,
  const DenseF& x_val,
  int max_components,
  const std::vector<int>& classes,
  int gpu_device,
  std::uint64_t epoch,
  std::size_t fold_slot,
  int& fitted_components
) {
  if (x_train.rows < 1 || x_train.cols < 1 || x_val.rows < 1) {
    throw std::invalid_argument("Resident CUDA PLS-LDA requires non-empty fold matrices.");
  }
  if (static_cast<int>(y_train.size()) != x_train.rows) {
    throw std::invalid_argument("Resident CUDA PLS-LDA label size mismatch.");
  }
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  CudaPLSLDAFloatWorkspace& workspace = cuda_pls_lda_float_workspace(gpu_device);
  CudaPLSLDAFloatWorkspace::ResidentFold& resident =
    prepare_cuda_resident_pls_fold(
      workspace,
      fold_slot,
      epoch,
      x_train,
      x_val,
      x_train_colmajor,
      stream
    );

  if (!resident.train_gram_ready) {
    resident.train_gram_colmajor.ensure(
      static_cast<std::size_t>(x_train.cols) *
      static_cast<std::size_t>(x_train.cols)
    );
    const float one = 1.0f;
    const float zero = 0.0f;
    check_cublas(
      cublasSgemm(
        context.blas(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        x_train.cols,
        x_train.cols,
        x_train.rows,
        &one,
        resident.train_colmajor.data(),
        x_train.rows,
        resident.train_colmajor.data(),
        x_train.rows,
        &zero,
        resident.train_gram_colmajor.data(),
        x_train.cols
      ),
      "cublasSgemm resident float32 PLS-LDA X'X"
    );
    resident.train_gram_ready = true;
  }

  encode_labels_from_sorted_classes(
    y_train,
    classes,
    workspace.encoded,
    workspace.class_counts,
    1
  );
  workspace.labels.ensure(workspace.encoded.size());
  workspace.counts.ensure(workspace.class_counts.size());
  check_cuda(
    cudaMemcpyAsync(
      workspace.labels.data(),
      workspace.encoded.data(),
      workspace.encoded.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      stream
    ),
    "cudaMemcpyAsync resident PLS-LDA labels"
  );
  check_cuda(
    cudaMemcpyAsync(
      workspace.counts.data(),
      workspace.class_counts.data(),
      workspace.class_counts.size() * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ),
    "cudaMemcpyAsync resident PLS-LDA class counts"
  );
  const int max_rank = pls_component_limit(
    max_components,
    x_train.rows,
    x_train.cols
  );
  const float* weights_column_major = nullptr;
  fitted_components = 0;
  for (int trial_rank = max_rank; trial_rank >= 1; --trial_rank) {
    const bool ok = kodama_fastpls_simpls_fit_cuda_float_labels_device(
      resident.train_colmajor.data(),
      x_train.rows,
      x_train.cols,
      resident.train_gram_colmajor.data(),
      workspace.labels.data(),
      workspace.counts.data(),
      static_cast<int>(classes.size()),
      trial_rank,
      stream,
      &weights_column_major
    );
    if (ok) {
      fitted_components = trial_rank;
      break;
    }
  }
  if (fitted_components < 1 || weights_column_major == nullptr) {
    throw std::runtime_error(
      "Resident CUDA label-aware float32 SIMPLS fit failed."
    );
  }

  workspace.weights.ensure(
    static_cast<std::size_t>(x_train.cols) *
    static_cast<std::size_t>(fitted_components)
  );
  kodama_cuda_transpose_pls_weights_float(
    weights_column_major,
    workspace.weights.data(),
    x_train.cols,
    fitted_components,
    stream
  );
  check_cuda(cudaGetLastError(), "resident CUDA PLS weight transpose");
  return predict_pls_lda_device_float(
    resident.train_rowmajor.data(),
    resident.validation_rowmajor.data(),
    resident.train_gram_colmajor.data(),
    workspace.weights.data(),
    x_train.rows,
    x_train.cols,
    x_val.rows,
    fitted_components,
    classes,
    workspace,
    context
  );
}

std::vector<int> train_predict_pls_lda_projected_cuda_float(
  const DenseF& x_train,
  const std::vector<int>& y_train,
  const DenseF& x_val,
  const PLSFitF& fit,
  int ncomp,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (x_train.rows < 1 || x_train.cols < 1) throw std::invalid_argument("CUDA float32 PLS-LDA requires a non-empty training matrix.");
  if (static_cast<int>(y_train.size()) != x_train.rows) throw std::invalid_argument("CUDA float32 PLS-LDA label size mismatch.");
  if (x_val.rows < 1) return {};
  if (x_val.cols != x_train.cols) throw std::invalid_argument("CUDA float32 PLS-LDA train/validation column mismatch.");
  if (fit.weights.rows != x_train.cols) throw std::invalid_argument("CUDA float32 PLS-LDA projection column mismatch.");
  if (ncomp < 1 || ncomp > fit.weights.cols) throw std::invalid_argument("CUDA float32 PLS-LDA component count exceeds fit rank.");

  const int n = x_train.rows;
  const int p = x_train.cols;
  const int n_val = x_val.rows;
  const int k = ncomp;
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  cublasHandle_t blas = context.blas();
  CudaPLSLDAFloatWorkspace& workspace = cuda_pls_lda_float_workspace(gpu_device);

  encode_labels_from_sorted_classes(y_train, classes, workspace.encoded, workspace.class_counts, 1);

  workspace.weights_prefix.resize(static_cast<std::size_t>(p) * static_cast<std::size_t>(k));
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < k; ++j) {
      workspace.weights_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  workspace.x_train.ensure(x_train.data.size());
  workspace.x_val.ensure(x_val.data.size());
  workspace.weights.ensure(workspace.weights_prefix.size());
  workspace.labels.ensure(workspace.encoded.size());
  workspace.counts.ensure(workspace.class_counts.size());
  check_cuda(cudaMemcpyAsync(workspace.x_train.data(), x_train.data.data(), x_train.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA Xtrain");
  check_cuda(cudaMemcpyAsync(workspace.x_val.data(), x_val.data.data(), x_val.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA Xval");
  check_cuda(cudaMemcpyAsync(workspace.weights.data(), workspace.weights_prefix.data(), workspace.weights_prefix.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA weights");
  check_cuda(cudaMemcpyAsync(workspace.labels.data(), workspace.encoded.data(), workspace.encoded.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA labels");
  check_cuda(cudaMemcpyAsync(workspace.counts.data(), workspace.class_counts.data(), workspace.class_counts.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA counts");

  workspace.train_gram.ensure(
    static_cast<std::size_t>(p) * static_cast<std::size_t>(p)
  );
  const float one = 1.0f;
  const float zero = 0.0f;
  // Row-major X is a column-major view of X'. Therefore A A' below is X'X.
  check_cublas(
    cublasSgemm(
      blas,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      p,
      p,
      n,
      &one,
      workspace.x_train.data(),
      p,
      workspace.x_train.data(),
      p,
      &zero,
      workspace.train_gram.data(),
      p
    ),
    "cublasSgemm float32 PLS-LDA X'X"
  );
  return predict_pls_lda_device_float(
    workspace.x_train.data(),
    workspace.x_val.data(),
    workspace.train_gram.data(),
    workspace.weights.data(),
    n,
    p,
    n_val,
    k,
    classes,
    workspace,
    context
  );
}

#endif


}  // namespace

namespace {

PLSCVResult run_plscv_host(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  PLSMode mode,
  Backend backend
) {
  if (backend != Backend::CPU && backend != Backend::Metal) {
    throw std::invalid_argument("Host PLS-CV backend must be CPU or Metal.");
  }
  detail::validate_inputs(x, labels, constrain);
  if (options.max_components < 1) throw std::invalid_argument("PLSOptions::max_components must be positive.");
  if (options.fixed_components < 0) throw std::invalid_argument("PLSOptions::fixed_components must be non-negative.");
  if (options.fixed_components > options.max_components) throw std::invalid_argument("PLSOptions::fixed_components cannot exceed max_components.");

  detail::Timer timer;
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified && options.data_epoch != 0;
  const PLSFoldXCacheF* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache_float(x, labels, constrain, options, false, backend == Backend::CPU) : nullptr;
  std::uint64_t metal_residency_epoch = 0;
  if (backend == Backend::Metal) {
    static std::atomic<std::uint64_t> uncached_metal_epoch{1};
    metal_residency_epoch = fold_cache != nullptr ?
      fold_cache->generation :
      uncached_metal_epoch.fetch_add(1, std::memory_order_relaxed);
    detail::metal_set_pls_residency_epoch(metal_residency_epoch);
  }
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments :
    detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<int> selected_pred(labels.size(), labels.empty() ? 0 : labels.front());
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids :
    detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;

  std::vector<int> fold_evaluated_components(fold_ids.size(), evaluated_component);
  auto process_fold = [&](std::size_t fold_pos) {
    if (backend == Backend::Metal) {
      // Keep invocation and fold identity in disjoint bit ranges. XOR-based
      // mixing allowed adjacent invocations to alias different folds, which
      // could reuse a stale resident matrix when host allocations matched.
      const std::uint64_t fold_epoch =
        ((metal_residency_epoch & 0xffffffffULL) << 32U) |
        (static_cast<std::uint64_t>(fold_pos) & 0xffffffffULL);
      detail::metal_set_pls_residency_epoch(fold_epoch);
    }
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    DenseF x_train_storage;
    DenseF x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const DenseF* x_train = nullptr;
    const DenseF* x_val = nullptr;
    const DenseF* x_train_gram = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldDataF& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
      if (fold_data.x_train_gram.rows > 0) x_train_gram = &fold_data.x_train_gram;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<float> x_mean;
      std::vector<float> x_scale;
      train_center_scale_float(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale_float(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale_float(x, validation_storage, x_mean, x_scale);
      x_train = &x_train_storage;
      x_val = &x_val_storage;
      validation = &validation_storage;
      train = &train_storage;
    }
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    const std::vector<int> fold_classes = detail::unique_labels(y_train_labels);
    if (fold_classes.size() <= 1U) {
      const int pred_label = fold_classes.empty() ? (labels.empty() ? 0 : labels.front()) : fold_classes.front();
      fold_evaluated_components[fold_pos] = 1;
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = pred_label;
      }
      return;
    }
    PLSFitF fit;
    try {
      fit = backend == Backend::Metal ?
        fit_pls_components_labels_metal_float(
          *x_train,
          y_train_labels,
          fold_classes,
          options.max_components
        ) :
        fit_pls_components_labels_float(*x_train, y_train_labels, fold_classes, options.max_components, x_train_gram);
    } catch (const DegeneratePLSFit&) {
      std::map<int, int> class_counts;
      for (int label : y_train_labels) ++class_counts[label];
      int pred_label = fold_classes.front();
      int best_count = -1;
      for (const auto& entry : class_counts) {
        if (entry.second > best_count) {
          pred_label = entry.first;
          best_count = entry.second;
        }
      }
      fold_evaluated_components[fold_pos] = 1;
      for (int row : *validation) {
        selected_pred[static_cast<std::size_t>(row)] = pred_label;
      }
      return;
    }
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    fold_evaluated_components[fold_pos] = eval_components.front();
    if ((backend == Backend::CPU || backend == Backend::Metal) && mode == PLSMode::PLS_LDA) {
      for (int a : eval_components) {
        const std::vector<int> fold_pred = backend == Backend::Metal ?
          fit_predict_pls_lda_metal_float(
            *x_train, y_train_labels, *x_val, fit, fold_classes, a
          ) :
          fit_predict_pls_lda_streamed_cpu_float(
            *x_train, y_train_labels, *x_val, fit, fold_classes, a
          );
        for (std::size_t i = 0; i < validation->size(); ++i) {
          selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
      return;
    }
    DenseF t_train_full = backend == Backend::Metal ?
      transform_pls_scores_metal_float(*x_train, fit, fit.weights.cols) :
      transform_pls_scores_float(*x_train, fit, fit.weights.cols);
    DenseF t_val_full = backend == Backend::Metal ?
      transform_pls_scores_metal_float(*x_val, fit, fit.weights.cols) :
      transform_pls_scores_float(*x_val, fit, fit.weights.cols);
    const std::vector<float> y_mean = label_means_float(y_train_labels, fold_classes);
    for (int a : eval_components) {
      std::vector<int> fold_pred = mode == PLSMode::PLS_LDA ?
        predict_pls_lda_float(t_train_full, y_train_labels, t_val_full, fold_classes, a) :
        predict_pls_da_labels_float(t_train_full, y_train_labels, t_val_full, fold_classes, y_mean, a);
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
      }
    }
  };

  const int fold_workers = std::max(
    1,
    std::min(options.n_threads, static_cast<int>(fold_ids.size()))
  );
  if (fold_workers <= 1) {
    for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) process_fold(fold_pos);
  } else {
    std::atomic<std::size_t> next_fold{0};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    auto worker = [&]() {
      for (;;) {
        const std::size_t fold_pos = next_fold.fetch_add(1);
        if (fold_pos >= fold_ids.size()) break;
        try {
          process_fold(fold_pos);
        } catch (...) {
          std::lock_guard<std::mutex> lock(worker_error_mutex);
          if (!worker_error) worker_error = std::current_exception();
          break;
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(fold_workers - 1));
    for (int w = 1; w < fold_workers; ++w) workers.emplace_back(worker);
    worker();
    for (std::thread& th : workers) th.join();
    if (worker_error) std::rethrow_exception(worker_error);
  }
  for (int comp : fold_evaluated_components) evaluated_component = std::min(evaluated_component, comp);

  const int best_comp = std::min(evaluated_component, options.max_components);
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, selected_pred);
  result.selected_components = best_comp;
  result.predicted_labels = std::move(selected_pred);
  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  result.parameters.backend = backend;
  result.parameters.mode = mode;
  result.parameters.max_components = options.max_components;
  result.parameters.selected_components = best_comp;
  result.parameters.fixed_components = options.fixed_components;
  result.parameters.center = options.center;
  result.parameters.scale = options.scale;
  result.parameters.gpu_device = options.gpu_device;
  result.parameters.n_threads = options.n_threads;
  return result;
}

#if defined(KODAMA_ENABLE_CUDA)
PLSCVResult run_plscv_cuda(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  PLSMode mode
) {
  detail::validate_inputs(x, labels, constrain);
  if (options.max_components < 1) throw std::invalid_argument("PLSOptions::max_components must be positive.");
  if (options.fixed_components < 0) throw std::invalid_argument("PLSOptions::fixed_components must be non-negative.");
  if (options.fixed_components > options.max_components) throw std::invalid_argument("PLSOptions::fixed_components cannot exceed max_components.");

  detail::Timer timer;
  check_cuda(cudaSetDevice(options.gpu_device), "cudaSetDevice(run_plscv_cuda)");
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified && options.data_epoch != 0;
  const PLSFoldXCacheF* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache_float(
      x,
      labels,
      constrain,
      options,
      true,
      mode != PLSMode::PLS_LDA
    ) : nullptr;
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments :
    detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<int> selected_pred(labels.size(), labels.empty() ? 0 : labels.front());
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids :
    detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  std::vector<int> fold_evaluated_components(
    fold_ids.size(),
    evaluated_component
  );

  auto process_fold = [&](std::size_t fold_pos) {
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    DenseF x_train_storage;
    DenseF x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const DenseF* x_train = nullptr;
    const DenseF* x_val = nullptr;
    const std::vector<float>* x_train_colmajor = nullptr;
    const std::vector<float>* x_train_gram_colmajor = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldDataF& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
      x_train_colmajor = &fold_data.x_train_colmajor;
      x_train_gram_colmajor = &fold_data.x_train_gram_colmajor;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<float> x_mean;
      std::vector<float> x_scale;
      train_center_scale_float(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale_float(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale_float(x, validation_storage, x_mean, x_scale);
      x_train = &x_train_storage;
      x_val = &x_val_storage;
      validation = &validation_storage;
      train = &train_storage;
    }
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    const std::vector<int> fold_classes = detail::unique_labels(y_train_labels);
    if (fold_classes.size() <= 1U) {
      const int pred_label = fold_classes.empty() ? (labels.empty() ? 0 : labels.front()) : fold_classes.front();
      fold_evaluated_components[fold_pos] = 1;
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = pred_label;
      }
      return;
    }
    if (mode == PLSMode::PLS_LDA &&
        fold_cache != nullptr &&
        x_train_colmajor != nullptr) {
      int fitted_components = 0;
      const std::vector<int> fold_pred =
        fit_predict_pls_lda_resident_cuda_float(
          *x_train,
          *x_train_colmajor,
          y_train_labels,
          *x_val,
          options.max_components,
          fold_classes,
          options.gpu_device,
          fold_cache->generation,
          fold_pos,
          fitted_components
        );
      const int requested = options.fixed_components > 0 ?
        options.fixed_components : options.max_components;
      fold_evaluated_components[fold_pos] = std::min(
        fold_evaluated_components[fold_pos],
        std::min(requested, fitted_components)
      );
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
      }
      return;
    }

    PLSFitF fit = fit_pls_components_cuda_labels_float(
        *x_train,
        y_train_labels,
        fold_classes,
        options.max_components,
        options.gpu_device,
        x_train_colmajor != nullptr ? x_train_colmajor->data() : nullptr,
        (x_train_gram_colmajor != nullptr && !x_train_gram_colmajor->empty()) ? x_train_gram_colmajor->data() : nullptr
      );
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    fold_evaluated_components[fold_pos] = std::min(
      fold_evaluated_components[fold_pos],
      eval_components.front()
    );
    if (mode == PLSMode::PLS_LDA) {
      for (int a : eval_components) {
        const std::vector<int> fold_pred = train_predict_pls_lda_projected_cuda_float(
            *x_train,
            y_train_labels,
            *x_val,
            fit,
            a,
            fold_classes,
            options.gpu_device
          );
        for (std::size_t i = 0; i < validation->size(); ++i) {
          selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    } else {
      const std::vector<float> y_mean = label_means_float(y_train_labels, fold_classes);
      for (int a : eval_components) {
        DenseF t_train_prefix = transform_pls_scores_cuda_float(*x_train, fit, a, options.gpu_device);
        DenseF t_val_prefix = transform_pls_scores_cuda_float(*x_val, fit, a, options.gpu_device);
        const std::vector<int> fold_pred = predict_pls_da_cuda_float(
            t_train_prefix,
            y_train_labels,
            t_val_prefix,
            fold_classes,
            y_mean,
            options.gpu_device
          );
        for (std::size_t i = 0; i < validation->size(); ++i) {
          selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    }
  };

  if (mode == PLSMode::PLS_LDA && fold_ids.size() > 1U) {
    std::vector<std::function<void()>> fold_tasks;
    fold_tasks.reserve(fold_ids.size());
    for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) {
      fold_tasks.emplace_back([&, fold_pos]() { process_fold(fold_pos); });
    }
    persistent_fold_executor(fold_ids.size()).run(fold_tasks);
  } else {
    for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) {
      process_fold(fold_pos);
    }
  }
  for (int component : fold_evaluated_components) {
    evaluated_component = std::min(evaluated_component, component);
  }

  const int best_comp = std::min(evaluated_component, options.max_components);
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, selected_pred);
  result.selected_components = best_comp;
  result.predicted_labels = std::move(selected_pred);
  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  result.parameters.backend = Backend::CUDA;
  result.parameters.mode = mode;
  result.parameters.max_components = options.max_components;
  result.parameters.selected_components = best_comp;
  result.parameters.fixed_components = options.fixed_components;
  result.parameters.center = options.center;
  result.parameters.scale = options.scale;
  result.parameters.gpu_device = options.gpu_device;
  result.parameters.n_threads = options.n_threads;
  return result;
}

std::vector<int> plslda_predict_cuda(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (train.data == nullptr || test.data == nullptr) throw std::invalid_argument("PLSLDAPredict input matrix pointer is null.");
  if (train.rows != labels.size()) throw std::invalid_argument("PLSLDAPredict labels size must match training rows.");
  if (train.cols != test.cols) throw std::invalid_argument("PLSLDAPredict train/test column mismatch.");
  if (train.rows == 0 || test.rows == 0 || train.cols == 0) return {};
  const std::vector<int> classes = detail::unique_labels(labels);
  if (classes.empty()) return {};
  if (classes.size() == 1) return std::vector<int>(test.rows, classes.front());

  std::vector<int> train_rows(train.rows);
  std::iota(train_rows.begin(), train_rows.end(), 0);
  std::vector<int> test_rows(test.rows);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_cuda_labels_float(
    x_train,
    labels,
    classes,
    ncomp,
    options.gpu_device,
    nullptr,
    nullptr
  );
  return train_predict_pls_lda_projected_cuda_float(
    x_train,
    labels,
    x_test,
    fit,
    fit.weights.cols,
    classes,
    options.gpu_device
  );
}
#endif

PLSFitResult fit_general_pls(
  MatrixView x,
  MatrixView y,
  const PLSOptions& options,
  const Backend backend
) {
  if (x.data == nullptr || y.data == nullptr || x.rows < 2 || x.cols < 1 || y.cols < 1) {
    throw std::invalid_argument("PLS requires non-empty X and Y matrices with at least two rows.");
  }
  if (x.rows != y.rows) throw std::invalid_argument("PLS X and Y must have the same rows.");
  const int n = static_cast<int>(x.rows);
  const int p = static_cast<int>(x.cols);
  const int q = static_cast<int>(y.cols);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  if (requested < 1) throw std::invalid_argument("PLS component count must be positive.");

  detail::Timer timer;
  std::vector<int> rows(static_cast<std::size_t>(n));
  std::iota(rows.begin(), rows.end(), 0);
  std::vector<float> x_center;
  std::vector<float> x_scale;
  train_center_scale_float(x, rows, options.center, options.scale, x_center, x_scale);
  DenseF x_prepared = subset_scale_float(x, rows, x_center, x_scale);
  DenseF y_prepared(n, q);
  std::vector<float> y_center(static_cast<std::size_t>(q), 0.0f);
  for (int row = 0; row < n; ++row) {
    for (int response = 0; response < q; ++response) {
      const float value = y.value_float(static_cast<std::size_t>(row), static_cast<std::size_t>(response));
      if (!std::isfinite(value)) throw std::invalid_argument("PLS Y must contain only finite values.");
      y_center[static_cast<std::size_t>(response)] += value;
      y_prepared(row, response) = value;
    }
  }
  if (options.center) {
    const float inverse_n = 1.0f / static_cast<float>(n);
    for (float& value : y_center) value *= inverse_n;
    for (int row = 0; row < n; ++row) {
      for (int response = 0; response < q; ++response) {
        y_prepared(row, response) -= y_center[static_cast<std::size_t>(response)];
      }
    }
  }

  const int components = pls_component_limit(requested, n, p);
  PLSFitF fit;
  DenseF scores;
  if (backend == Backend::CUDA) {
#if defined(KODAMA_ENABLE_CUDA)
    std::vector<float> y_colmajor(static_cast<std::size_t>(n) * static_cast<std::size_t>(q));
    for (int response = 0; response < q; ++response) {
      for (int row = 0; row < n; ++row) {
        y_colmajor[static_cast<std::size_t>(row) + static_cast<std::size_t>(response) * n] =
          y_prepared(row, response);
      }
    }
    fit = fit_pls_components_cuda_colmajor_y_float(
      x_prepared, y_colmajor.data(), q, components, options.gpu_device
    );
    scores = transform_pls_scores_cuda_float(x_prepared, fit, fit.weights.cols, options.gpu_device);
#else
    throw std::runtime_error("PLS_CUDA requires a CUDA build.");
#endif
  } else {
    DenseF cross_product(p, q);
    for (int feature = 0; feature < p; ++feature) {
      for (int response = 0; response < q; ++response) {
        float value = 0.0f;
        for (int row = 0; row < n; ++row) {
          value += x_prepared(row, feature) * y_prepared(row, response);
        }
        cross_product(feature, response) = value;
      }
    }
    if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
      const detail::MetalSIMPLSResult metal = detail::metal_simpls_fit(
        x_prepared.data, n, p, cross_product.data, q, components
      );
      fit.weights = DenseF(metal.predictors, metal.components);
      fit.weights.data = metal.weights;
      fit.y_weights = DenseF(metal.responses, metal.components);
      fit.y_weights.data = metal.y_weights;
      scores = transform_pls_scores_metal_float(x_prepared, fit, fit.weights.cols);
#else
      throw std::runtime_error("PLS_METAL requires an Apple Metal build.");
#endif
    } else {
      fit = fit_pls_components_from_crossprod_float(
        x_prepared, std::move(cross_product), components
      );
      scores = transform_pls_scores_float(x_prepared, fit, fit.weights.cols);
    }
  }

  PLSFitResult result;
  result.samples = n;
  result.predictors = p;
  result.responses = q;
  result.components = fit.weights.cols;
  result.backend = backend;
  result.weights = fit.weights.data;
  result.response_loadings = fit.y_weights.data;
  result.scores = scores.data;
  result.x_center = std::move(x_center);
  result.x_scale = std::move(x_scale);
  result.y_center = std::move(y_center);
  result.coefficients.assign(static_cast<std::size_t>(p) * q, 0.0f);
  for (int feature = 0; feature < p; ++feature) {
    for (int response = 0; response < q; ++response) {
      float value = 0.0f;
      for (int component = 0; component < result.components; ++component) {
        value += fit.weights(feature, component) * fit.y_weights(response, component);
      }
      result.coefficients[static_cast<std::size_t>(feature) * q + response] = value;
    }
  }
  result.fitted.assign(static_cast<std::size_t>(n) * q, 0.0f);
  for (int row = 0; row < n; ++row) {
    for (int response = 0; response < q; ++response) {
      float value = result.y_center[static_cast<std::size_t>(response)];
      for (int component = 0; component < result.components; ++component) {
        value += scores(row, component) * fit.y_weights(response, component);
      }
      result.fitted[static_cast<std::size_t>(row) * q + response] = value;
    }
  }
  result.runtime_seconds = timer.seconds();
  return result;
}

}  // namespace

PLSCVResult PLSDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSDACV_CUDA(x, labels, constrain, options);
  return PLSDACV_CPU(x, labels, constrain, options);
}

PLSFitResult PLS_CPU(MatrixView x, MatrixView y, const PLSOptions& options) {
  return fit_general_pls(x, y, options, Backend::CPU);
}

PLSFitResult PLS_CUDA(MatrixView x, MatrixView y, const PLSOptions& options) {
  return fit_general_pls(x, y, options, Backend::CUDA);
}

PLSFitResult PLS_METAL(MatrixView x, MatrixView y, const PLSOptions& options) {
  return fit_general_pls(x, y, options, Backend::Metal);
}

PLSFitResult PLS(MatrixView x, MatrixView y, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLS_CUDA(x, y, options);
  if (options.backend == Backend::Metal) return PLS_METAL(x, y, options);
  return PLS_CPU(x, y, options);
}

PLSCVResult PLSLDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSLDACV_CUDA(x, labels, constrain, options);
  if (options.backend == Backend::Metal) return PLSLDACV_METAL(x, labels, constrain, options);
  return PLSLDACV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_host(x, labels, constrain, options, PLSMode::PLS_DA, Backend::CPU);
}

PLSCVResult PLSLDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_host(x, labels, constrain, options, PLSMode::PLS_LDA, Backend::CPU);
}

PLSCVResult PLSDACV_CUDA(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_CUDA)
  PLSOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_plscv_cuda(x, labels, constrain, cuda_options, PLSMode::PLS_DA);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSDACV_CUDA requires a CUDA/cuBLAS build.");
#endif
}

PLSCVResult PLSLDACV_CUDA(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_CUDA)
  PLSOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_plscv_cuda(x, labels, constrain, cuda_options, PLSMode::PLS_LDA);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSLDACV_CUDA requires a CUDA/cuBLAS build.");
#endif
}

PLSCVResult PLSLDACV_METAL(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_METAL)
  PLSOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  return run_plscv_host(x, labels, constrain, metal_options, PLSMode::PLS_LDA, Backend::Metal);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSLDACV_METAL requires an Apple Metal build.");
#endif
}

std::vector<int> PLSLDAPredict_CPU(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (train.data == nullptr || test.data == nullptr) throw std::invalid_argument("PLSLDAPredict input matrix pointer is null.");
  if (train.rows != labels.size()) throw std::invalid_argument("PLSLDAPredict labels size must match training rows.");
  if (train.cols != test.cols) throw std::invalid_argument("PLSLDAPredict train/test column mismatch.");
  if (train.rows == 0 || test.rows == 0 || train.cols == 0) return {};
  const std::vector<int> classes = detail::unique_labels(labels);
  if (classes.empty()) return {};
  if (classes.size() == 1) return std::vector<int>(test.rows, classes.front());

  std::vector<int> train_rows(train.rows);
  std::iota(train_rows.begin(), train_rows.end(), 0);
  std::vector<int> test_rows(test.rows);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_labels_float(x_train, labels, classes, ncomp);
  return fit_predict_pls_lda_streamed_cpu_float(
    x_train, labels, x_test, fit, classes, fit.weights.cols
  );
}

std::vector<int> PLSLDAPredict_CUDA(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  return plslda_predict_cuda(train, labels, test, options);
#else
  (void)train;
  (void)labels;
  (void)test;
  (void)options;
  throw std::runtime_error("PLSLDAPredict_CUDA requires a CUDA/cuBLAS build.");
#endif
}

std::vector<int> PLSLDAPredict_METAL(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  static std::atomic<std::uint64_t> direct_prediction_epoch{1};
  detail::metal_set_pls_residency_epoch(
    direct_prediction_epoch.fetch_add(1, std::memory_order_relaxed)
  );
  if (train.data == nullptr || test.data == nullptr) throw std::invalid_argument("PLSLDAPredict input matrix pointer is null.");
  if (train.rows != labels.size()) throw std::invalid_argument("PLSLDAPredict labels size must match training rows.");
  if (train.cols != test.cols) throw std::invalid_argument("PLSLDAPredict train/test column mismatch.");
  if (train.rows == 0 || test.rows == 0 || train.cols == 0) return {};
  const std::vector<int> classes = detail::unique_labels(labels);
  if (classes.empty()) return {};
  if (classes.size() == 1) return std::vector<int>(test.rows, classes.front());
  std::vector<int> train_rows(train.rows);
  std::iota(train_rows.begin(), train_rows.end(), 0);
  std::vector<int> test_rows(test.rows);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_labels_metal_float(x_train, labels, classes, ncomp);
  DenseF t_train = transform_pls_scores_metal_float(x_train, fit, fit.weights.cols);
  DenseF t_test = transform_pls_scores_metal_float(x_test, fit, fit.weights.cols);
  return predict_pls_lda_float(
    t_train, labels, t_test, classes, fit.weights.cols
  );
#else
  (void)train;
  (void)labels;
  (void)test;
  (void)options;
  throw std::runtime_error("PLSLDAPredict_METAL requires an Apple Metal build.");
#endif
}

std::vector<int> PLSLDAPredict(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (options.backend == Backend::CUDA) return PLSLDAPredict_CUDA(train, labels, test, options);
  if (options.backend == Backend::Metal) return PLSLDAPredict_METAL(train, labels, test, options);
  return PLSLDAPredict_CPU(train, labels, test, options);
}

}  // namespace kodama
