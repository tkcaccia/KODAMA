// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include "metal_backend.hpp"
#ifdef KODAMA_ENABLE_CUDA
#include "pca_cuda_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;

struct PreparedPCA {
  std::vector<float> values;
  std::vector<float> center;
  std::vector<float> scale;
  float sum_squares = 0.0f;
};

struct EigenResult {
  std::vector<float> values;
  std::vector<float> vectors;
};

using Multiply = std::function<std::vector<float>(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  int,
  bool,
  bool
)>;

int pca_threads(const int requested, const int items) {
  if (items <= 1) return 1;
  return std::max(1, std::min({requested, 4, items}));
}

template <typename Function>
void parallel_rows(const int rows, const int requested_threads, Function function) {
  const int threads = pca_threads(requested_threads, rows);
  if (threads == 1) {
    function(0, rows);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads - 1));
  for (int t = 1; t < threads; ++t) {
    const int begin = static_cast<int>(
      static_cast<std::int64_t>(rows) * t / threads
    );
    const int end = static_cast<int>(
      static_cast<std::int64_t>(rows) * (t + 1) / threads
    );
    workers.emplace_back([&, begin, end]() { function(begin, end); });
  }
  function(0, rows / threads);
  for (auto& worker : workers) worker.join();
}

PreparedPCA prepare_pca_matrix(const MatrixView x, const PCAOptions& options) {
  if (x.data == nullptr || x.rows < 2 || x.cols < 1) {
    throw std::invalid_argument("PCA requires at least two rows and one column.");
  }
  if (x.rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      x.cols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("PCA matrix dimensions exceed the supported integer range.");
  }

  const int n = static_cast<int>(x.rows);
  const int p = static_cast<int>(x.cols);
  PreparedPCA out;
  out.values.resize(static_cast<std::size_t>(n) * p);
  out.center.assign(static_cast<std::size_t>(p), 0.0f);
  out.scale.assign(static_cast<std::size_t>(p), options.scale ? 0.0f : 1.0f);

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < p; ++col) {
      const float value = x.value_float(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      if (!std::isfinite(value)) {
        throw std::invalid_argument("PCA input must contain only finite values.");
      }
      out.values[static_cast<std::size_t>(row) * p + col] = value;
      if (options.center) out.center[static_cast<std::size_t>(col)] += value;
    }
  }
  if (options.center) {
    const float inv_n = 1.0f / static_cast<float>(n);
    for (float& value : out.center) value *= inv_n;
  }

  parallel_rows(n, options.n_threads, [&](const int begin, const int end) {
    for (int row = begin; row < end; ++row) {
      float* values = out.values.data() + static_cast<std::size_t>(row) * p;
      for (int col = 0; col < p; ++col) {
        values[col] -= out.center[static_cast<std::size_t>(col)];
      }
    }
  });

  if (options.scale) {
    for (int row = 0; row < n; ++row) {
      const float* values = out.values.data() + static_cast<std::size_t>(row) * p;
      for (int col = 0; col < p; ++col) {
        const float value = values[col];
        out.scale[static_cast<std::size_t>(col)] += value * value;
      }
    }
    for (float& value : out.scale) {
      value = std::sqrt(std::max(0.0f, value / static_cast<float>(n - 1)));
      if (!std::isfinite(value) || value <= std::numeric_limits<float>::epsilon()) value = 1.0f;
    }
    parallel_rows(n, options.n_threads, [&](const int begin, const int end) {
      for (int row = begin; row < end; ++row) {
        float* values = out.values.data() + static_cast<std::size_t>(row) * p;
        for (int col = 0; col < p; ++col) {
          values[col] /= out.scale[static_cast<std::size_t>(col)];
        }
      }
    });
  }

  float sum_squares = 0.0f;
  for (const float value : out.values) sum_squares += value * value;
  out.sum_squares = sum_squares;
  return out;
}

std::vector<float> multiply_cpu(
  const std::vector<float>& left,
  const int left_rows,
  const int left_cols,
  const std::vector<float>& right,
  const int right_rows,
  const int right_cols,
  const bool transpose_left,
  const bool transpose_right,
  const int n_threads
) {
  const int out_rows = transpose_left ? left_cols : left_rows;
  const int inner_left = transpose_left ? left_rows : left_cols;
  const int inner_right = transpose_right ? right_cols : right_rows;
  const int out_cols = transpose_right ? right_rows : right_cols;
  if (left.size() != static_cast<std::size_t>(left_rows) * left_cols ||
      right.size() != static_cast<std::size_t>(right_rows) * right_cols ||
      inner_left != inner_right) {
    throw std::invalid_argument("Non-conformable PCA matrix multiplication.");
  }
  std::vector<float> out(static_cast<std::size_t>(out_rows) * out_cols, 0.0f);
  parallel_rows(out_rows, n_threads, [&](const int begin, const int end) {
    for (int row = begin; row < end; ++row) {
      for (int col = 0; col < out_cols; ++col) {
        float sum = 0.0f;
        for (int inner = 0; inner < inner_left; ++inner) {
          const float a = transpose_left ?
            left[static_cast<std::size_t>(inner) * left_cols + row] :
            left[static_cast<std::size_t>(row) * left_cols + inner];
          const float b = transpose_right ?
            right[static_cast<std::size_t>(col) * right_cols + inner] :
            right[static_cast<std::size_t>(inner) * right_cols + col];
          sum += a * b;
        }
        out[static_cast<std::size_t>(row) * out_cols + col] = sum;
      }
    }
  });
  return out;
}

std::vector<float> thin_qr(const std::vector<float>& matrix, const int rows, const int cols) {
  if (matrix.size() != static_cast<std::size_t>(rows) * cols || rows < cols) {
    throw std::invalid_argument("Invalid matrix dimensions for PCA thin QR.");
  }
  std::vector<float> q(matrix);
  const float tolerance = 32.0f * std::numeric_limits<float>::epsilon();
  for (int col = 0; col < cols; ++col) {
    for (int pass = 0; pass < 2; ++pass) {
      for (int prev = 0; prev < col; ++prev) {
        float dot = 0.0f;
        for (int row = 0; row < rows; ++row) {
          dot += q[static_cast<std::size_t>(row) * cols + prev] *
                 q[static_cast<std::size_t>(row) * cols + col];
        }
        for (int row = 0; row < rows; ++row) {
          q[static_cast<std::size_t>(row) * cols + col] -=
            dot * q[static_cast<std::size_t>(row) * cols + prev];
        }
      }
    }
    float norm_squared = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = q[static_cast<std::size_t>(row) * cols + col];
      norm_squared += value * value;
    }
    const float norm = std::sqrt(std::max(0.0f, norm_squared));
    if (!std::isfinite(norm) || norm <= tolerance) {
      for (int row = 0; row < rows; ++row) {
        q[static_cast<std::size_t>(row) * cols + col] = 0.0f;
      }
      continue;
    }
    const float inverse = 1.0f / norm;
    for (int row = 0; row < rows; ++row) {
      q[static_cast<std::size_t>(row) * cols + col] *= inverse;
    }
  }
  return q;
}

EigenResult jacobi_symmetric_eigen(std::vector<float> matrix, const int size) {
  EigenResult out;
  out.values.assign(static_cast<std::size_t>(size), 0.0f);
  out.vectors.assign(static_cast<std::size_t>(size) * size, 0.0f);
  for (int i = 0; i < size; ++i) {
    out.vectors[static_cast<std::size_t>(i) * size + i] = 1.0f;
  }

  const int max_iterations = std::max(32, 40 * size * size);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    int p = 0;
    int q = 1;
    float largest = 0.0f;
    for (int row = 0; row < size; ++row) {
      for (int col = row + 1; col < size; ++col) {
        const float value = std::abs(matrix[static_cast<std::size_t>(row) * size + col]);
        if (value > largest) {
          largest = value;
          p = row;
          q = col;
        }
      }
    }
    float diagonal_scale = 1.0f;
    for (int i = 0; i < size; ++i) {
      diagonal_scale = std::max(diagonal_scale, std::abs(matrix[static_cast<std::size_t>(i) * size + i]));
    }
    if (largest <= 16.0f * std::numeric_limits<float>::epsilon() * diagonal_scale) break;

    const float app = matrix[static_cast<std::size_t>(p) * size + p];
    const float aqq = matrix[static_cast<std::size_t>(q) * size + q];
    const float apq = matrix[static_cast<std::size_t>(p) * size + q];
    const float angle = 0.5f * std::atan2(2.0f * apq, aqq - app);
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    for (int k = 0; k < size; ++k) {
      if (k == p || k == q) continue;
      const float mkp = matrix[static_cast<std::size_t>(k) * size + p];
      const float mkq = matrix[static_cast<std::size_t>(k) * size + q];
      const float new_kp = c * mkp - s * mkq;
      const float new_kq = s * mkp + c * mkq;
      matrix[static_cast<std::size_t>(k) * size + p] = new_kp;
      matrix[static_cast<std::size_t>(p) * size + k] = new_kp;
      matrix[static_cast<std::size_t>(k) * size + q] = new_kq;
      matrix[static_cast<std::size_t>(q) * size + k] = new_kq;
    }
    matrix[static_cast<std::size_t>(p) * size + p] =
      c * c * app - 2.0f * s * c * apq + s * s * aqq;
    matrix[static_cast<std::size_t>(q) * size + q] =
      s * s * app + 2.0f * s * c * apq + c * c * aqq;
    matrix[static_cast<std::size_t>(p) * size + q] = 0.0f;
    matrix[static_cast<std::size_t>(q) * size + p] = 0.0f;

    for (int row = 0; row < size; ++row) {
      const float vip = out.vectors[static_cast<std::size_t>(row) * size + p];
      const float viq = out.vectors[static_cast<std::size_t>(row) * size + q];
      out.vectors[static_cast<std::size_t>(row) * size + p] = c * vip - s * viq;
      out.vectors[static_cast<std::size_t>(row) * size + q] = s * vip + c * viq;
    }
  }

  for (int i = 0; i < size; ++i) {
    out.values[static_cast<std::size_t>(i)] =
      matrix[static_cast<std::size_t>(i) * size + i];
  }
  std::vector<int> order(static_cast<std::size_t>(size));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return out.values[static_cast<std::size_t>(a)] > out.values[static_cast<std::size_t>(b)];
  });
  EigenResult sorted;
  sorted.values.resize(static_cast<std::size_t>(size));
  sorted.vectors.resize(static_cast<std::size_t>(size) * size);
  for (int col = 0; col < size; ++col) {
    const int source = order[static_cast<std::size_t>(col)];
    sorted.values[static_cast<std::size_t>(col)] = out.values[static_cast<std::size_t>(source)];
    for (int row = 0; row < size; ++row) {
      sorted.vectors[static_cast<std::size_t>(row) * size + col] =
        out.vectors[static_cast<std::size_t>(row) * size + source];
    }
  }
  return sorted;
}

std::pair<int, int> resolve_rsvd_tuning(
  const int n,
  const int p,
  const int rank,
  const Backend backend,
  const PCAOptions& options
) {
  int oversample = options.oversample;
  int power = options.power_iterations;
  if (oversample < 0) {
    if (backend == Backend::CUDA || backend == Backend::Metal) {
      oversample = static_cast<std::int64_t>(n) * p >= 5000000 ? 16 : 10;
    } else {
      oversample = rank <= 10 ? 10 : std::min(20, std::max(10, (rank + 1) / 2));
    }
  }
  if (power < 0) {
    if (backend == Backend::CUDA) {
      power = (rank <= 20 || p <= 128) ? 2 : 1;
    } else if (backend == Backend::Metal) {
      power = rank <= 30 ? 1 : 0;
    } else {
      power = rank <= 30 ? 1 : 0;
    }
  }
  if (oversample < 0 || power < 0) {
    throw std::invalid_argument("PCA oversample and power_iterations must be non-negative.");
  }
  return {oversample, power};
}

PCAResult pca_backend(const MatrixView x, PCAOptions options, const Backend backend) {
  const auto started = Clock::now();
  if (options.n_components < 1) {
    throw std::invalid_argument("PCA n_components must be positive.");
  }
  if (options.n_threads < 1) {
    throw std::invalid_argument("PCA n_threads must be positive.");
  }
  if (options.oversample < -1 || options.power_iterations < -1) {
    throw std::invalid_argument("PCA oversample and power_iterations must be -1 or non-negative.");
  }
  if (backend == Backend::Metal && !detail::metal_backend_available()) {
    throw std::runtime_error("The Metal backend is not available for PCA in this build.");
  }
#ifndef KODAMA_ENABLE_CUDA
  if (backend == Backend::CUDA) {
    throw std::runtime_error("The CUDA backend is not available for PCA in this build.");
  }
#endif

  PreparedPCA prepared = prepare_pca_matrix(x, options);
  const int n = static_cast<int>(x.rows);
  const int p = static_cast<int>(x.cols);
  const int maximum_rank = std::min(p, options.center ? n - 1 : n);
  const int rank = std::min(options.n_components, maximum_rank);
  if (rank < 1) throw std::runtime_error("PCA input has no usable rank.");
  const auto tuning = resolve_rsvd_tuning(n, p, rank, backend, options);
  const int oversample = tuning.first;
  const int power = tuning.second;
  const int sketch_rank = std::min(std::min(n, p), rank + oversample);

  Multiply multiply;
  if (backend == Backend::Metal) {
    multiply = [](const std::vector<float>& left, const int left_rows, const int left_cols,
                  const std::vector<float>& right, const int right_rows, const int right_cols,
                  const bool transpose_left, const bool transpose_right) {
      return detail::metal_matrix_multiply(
        left, left_rows, left_cols, right, right_rows, right_cols,
        transpose_left, transpose_right
      );
    };
  } else if (backend == Backend::CUDA) {
#ifdef KODAMA_ENABLE_CUDA
    const int device = options.gpu_device;
    multiply = [device](const std::vector<float>& left, const int left_rows, const int left_cols,
                        const std::vector<float>& right, const int right_rows, const int right_cols,
                        const bool transpose_left, const bool transpose_right) {
      return detail::cuda_pca_matrix_multiply(
        left, left_rows, left_cols, right, right_rows, right_cols,
        transpose_left, transpose_right, device
      );
    };
#endif
  } else {
    const int threads = options.n_threads;
    multiply = [threads](const std::vector<float>& left, const int left_rows, const int left_cols,
                         const std::vector<float>& right, const int right_rows, const int right_cols,
                         const bool transpose_left, const bool transpose_right) {
      return multiply_cpu(
        left, left_rows, left_cols, right, right_rows, right_cols,
        transpose_left, transpose_right, threads
      );
    };
  }

  std::mt19937 rng(static_cast<std::uint32_t>(options.seed));
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> omega(static_cast<std::size_t>(p) * sketch_rank);
  for (float& value : omega) value = normal(rng);

  std::vector<float> y = multiply(
    prepared.values, n, p, omega, p, sketch_rank, false, false
  );
  if (power == 1) {
    std::vector<float> z = multiply(
      prepared.values, n, p, y, n, sketch_rank, true, false
    );
    y = multiply(prepared.values, n, p, z, p, sketch_rank, false, false);
  } else {
    for (int iteration = 0; iteration < power; ++iteration) {
      std::vector<float> z = multiply(
        prepared.values, n, p, y, n, sketch_rank, true, false
      );
      z = thin_qr(z, p, sketch_rank);
      y = multiply(prepared.values, n, p, z, p, sketch_rank, false, false);
    }
  }

  const std::vector<float> q = thin_qr(y, n, sketch_rank);
  const std::vector<float> b = multiply(
    q, n, sketch_rank, prepared.values, n, p, true, false
  );
  const std::vector<float> gram = multiply_cpu(
    b, sketch_rank, p, b, sketch_rank, p, false, true, options.n_threads
  );
  const EigenResult eigen = jacobi_symmetric_eigen(gram, sketch_rank);

  PCAResult result;
  result.samples = n;
  result.variables = p;
  result.components = rank;
  result.oversample = sketch_rank - rank;
  result.power_iterations = power;
  result.backend = backend;
  result.center = std::move(prepared.center);
  result.scale = std::move(prepared.scale);
  result.loadings.assign(static_cast<std::size_t>(p) * rank, 0.0f);
  result.singular_values.assign(static_cast<std::size_t>(rank), 0.0f);

  const float singular_tolerance =
    std::sqrt(std::max(1.0f, eigen.values.empty() ? 1.0f : eigen.values[0])) *
    64.0f * std::numeric_limits<float>::epsilon();
  for (int component = 0; component < rank; ++component) {
    const float singular = std::sqrt(std::max(0.0f, eigen.values[static_cast<std::size_t>(component)]));
    result.singular_values[static_cast<std::size_t>(component)] = singular;
    if (singular <= singular_tolerance) continue;
    const float inverse = 1.0f / singular;
    for (int variable = 0; variable < p; ++variable) {
      float value = 0.0f;
      for (int row = 0; row < sketch_rank; ++row) {
        value += b[static_cast<std::size_t>(row) * p + variable] *
                 eigen.vectors[static_cast<std::size_t>(row) * sketch_rank + component];
      }
      result.loadings[static_cast<std::size_t>(variable) * rank + component] = value * inverse;
    }
  }

  result.scores = multiply(
    prepared.values, n, p, result.loadings, p, rank, false, false
  );
  for (int component = 0; component < rank; ++component) {
    int pivot = 0;
    float largest = 0.0f;
    for (int variable = 0; variable < p; ++variable) {
      const float value = std::abs(result.loadings[static_cast<std::size_t>(variable) * rank + component]);
      if (value > largest) {
        largest = value;
        pivot = variable;
      }
    }
    if (result.loadings[static_cast<std::size_t>(pivot) * rank + component] < 0.0f) {
      for (int variable = 0; variable < p; ++variable) {
        result.loadings[static_cast<std::size_t>(variable) * rank + component] *= -1.0f;
      }
      for (int row = 0; row < n; ++row) {
        result.scores[static_cast<std::size_t>(row) * rank + component] *= -1.0f;
      }
    }
  }

  const float denominator = static_cast<float>(std::max(1, n - 1));
  result.total_variance = static_cast<double>(prepared.sum_squares / denominator);
  result.sdev.resize(static_cast<std::size_t>(rank));
  result.variance.resize(static_cast<std::size_t>(rank));
  result.variance_explained.resize(static_cast<std::size_t>(rank));
  result.cumulative_variance_explained.resize(static_cast<std::size_t>(rank));
  float cumulative = 0.0f;
  for (int component = 0; component < rank; ++component) {
    const float singular = result.singular_values[static_cast<std::size_t>(component)];
    const float variance = singular * singular / denominator;
    const float explained = result.total_variance > 0.0 ?
      variance / static_cast<float>(result.total_variance) :
      std::numeric_limits<float>::quiet_NaN();
    cumulative += explained;
    result.sdev[static_cast<std::size_t>(component)] = singular / std::sqrt(denominator);
    result.variance[static_cast<std::size_t>(component)] = variance;
    result.variance_explained[static_cast<std::size_t>(component)] = explained;
    result.cumulative_variance_explained[static_cast<std::size_t>(component)] = cumulative;
  }
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

}  // namespace

PCAResult PCA_CPU(const MatrixView x, const PCAOptions& options) {
  PCAOptions resolved = options;
  resolved.backend = Backend::CPU;
  return pca_backend(x, resolved, Backend::CPU);
}

PCAResult PCA_CUDA(const MatrixView x, const PCAOptions& options) {
  PCAOptions resolved = options;
  resolved.backend = Backend::CUDA;
  return pca_backend(x, resolved, Backend::CUDA);
}

PCAResult PCA_METAL(const MatrixView x, const PCAOptions& options) {
  PCAOptions resolved = options;
  resolved.backend = Backend::Metal;
  return pca_backend(x, resolved, Backend::Metal);
}

PCAResult PCA(const MatrixView x, const PCAOptions& options) {
  switch (options.backend) {
    case Backend::CUDA:
      return PCA_CUDA(x, options);
    case Backend::Metal:
      return PCA_METAL(x, options);
    case Backend::CPU:
    case Backend::Auto:
      return PCA_CPU(x, options);
  }
  throw std::invalid_argument("Unknown PCA backend.");
}

PCAResult RSVD(const MatrixView x, const PCAOptions& options) {
  return PCA(x, options);
}

}  // namespace kodama
