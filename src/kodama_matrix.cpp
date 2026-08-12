// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"

#include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kodama {
namespace {

#if defined(KODAMA_DISABLE_NATIVE_PROGRESS)
class NullProgressBuffer final : public std::streambuf {
 protected:
  int overflow(int value) override { return value; }
};

std::ostream& progress_stream() {
  static NullProgressBuffer buffer;
  static std::ostream stream(&buffer);
  return stream;
}
#else
std::ostream& progress_stream() { return std::cerr; }
#endif

#if defined(KODAMA_ENABLE_CUDA) || defined(KODAMA_ENABLE_METAL)
constexpr double kKODAMAGraphTargetRecall = 0.99;
#endif

class OmpThreadScope {
 public:
  explicit OmpThreadScope(int n_threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(std::max(1, n_threads));
#else
    (void)n_threads;
#endif
  }

  ~OmpThreadScope() {
#ifdef _OPENMP
    if (previous_ > 0) omp_set_num_threads(previous_);
#endif
  }

 private:
#ifdef _OPENMP
  int previous_ = 0;
#endif
};

inline float dot_float32(const float* lhs, const float* rhs, int size) {
  int i = 0;
#if defined(__aarch64__)
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  for (; i + 15 < size; i += 16) {
    sum0 = vfmaq_f32(sum0, vld1q_f32(lhs + i), vld1q_f32(rhs + i));
    sum1 = vfmaq_f32(sum1, vld1q_f32(lhs + i + 4), vld1q_f32(rhs + i + 4));
    sum2 = vfmaq_f32(sum2, vld1q_f32(lhs + i + 8), vld1q_f32(rhs + i + 8));
    sum3 = vfmaq_f32(sum3, vld1q_f32(lhs + i + 12), vld1q_f32(rhs + i + 12));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
#elif defined(__x86_64__) || defined(_M_X64)
  __m128 sum0 = _mm_setzero_ps();
  __m128 sum1 = _mm_setzero_ps();
  __m128 sum2 = _mm_setzero_ps();
  __m128 sum3 = _mm_setzero_ps();
  for (; i + 15 < size; i += 16) {
    sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(lhs + i), _mm_loadu_ps(rhs + i)));
    sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(lhs + i + 4), _mm_loadu_ps(rhs + i + 4)));
    sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(lhs + i + 8), _mm_loadu_ps(rhs + i + 8)));
    sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_loadu_ps(lhs + i + 12), _mm_loadu_ps(rhs + i + 12)));
  }
  alignas(16) float lanes[4];
  _mm_store_ps(lanes, _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3)));
  float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#else
  float sum = 0.0f;
#endif
  for (; i < size; ++i) sum += lhs[i] * rhs[i];
  return sum;
}

inline float squared_norm_float32(const float* values, int size) {
  return dot_float32(values, values, size);
}

std::vector<float> copy_float32(MatrixView x, const std::vector<int>& rows = std::vector<int>()) {
  const std::size_t n = rows.empty() ? x.rows : rows.size();
  std::vector<float> out(n * x.cols, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t src = rows.empty() ? i : static_cast<std::size_t>(rows[i]);
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[i * x.cols + j] = x.value_float(src, j);
    }
  }
  return out;
}

void copy_float32_rows_into(
  const std::vector<float>& x,
  std::size_t cols,
  const std::vector<int>& rows,
  std::vector<float>& out
) {
  out.resize(rows.size() * cols);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t src = static_cast<std::size_t>(rows[i]) * cols;
    std::copy_n(x.data() + src, cols, out.data() + i * cols);
  }
}

void normalize_rows_for_cosine(std::vector<float>& x, std::size_t rows, std::size_t cols) {
  for (std::size_t i = 0; i < rows; ++i) {
    long double ss = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      const float v = x[i * cols + j];
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double norm = std::sqrt(static_cast<double>(ss));
    if (norm <= 0.0 || !std::isfinite(norm)) continue;
    const float scale = static_cast<float>(1.0 / norm);
    for (std::size_t j = 0; j < cols; ++j) x[i * cols + j] *= scale;
  }
}

int majority_label(const std::vector<int>& values) {
  std::map<int, int> counts;
  for (int v : values) counts[v] += 1;
  int best_label = counts.begin()->first;
  int best_count = counts.begin()->second;
  for (const auto& kv : counts) {
    if (kv.second > best_count || (kv.second == best_count && kv.first < best_label)) {
      best_label = kv.first;
      best_count = kv.second;
    }
  }
  return best_label;
}

std::vector<int> normalize_constrain(const std::vector<int>& constrain, std::size_t n) {
  if (constrain.empty()) {
    std::vector<int> out(n);
    std::iota(out.begin(), out.end(), 1);
    return out;
  }
  if (constrain.size() != n) throw std::invalid_argument("constrain size must be zero or match number of rows.");
  std::map<int, int> ids;
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    auto it = ids.find(constrain[i]);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(constrain[i], id).first;
    }
    out[i] = it->second;
  }
  return out;
}

std::vector<int> normalize_fixed(const std::vector<int>& fixed, std::size_t n) {
  if (fixed.empty()) return std::vector<int>(n, 0);
  if (fixed.size() != n) throw std::invalid_argument("fixed size must be zero or match number of rows.");
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) out[i] = fixed[i] != 0 ? 1 : 0;
  return out;
}

bool is_identity_constrain(const std::vector<int>& constrain) {
  for (std::size_t i = 0; i < constrain.size(); ++i) {
    if (constrain[i] != static_cast<int>(i + 1)) return false;
  }
  return true;
}

NeighborGraph hnsw_graph(
  const std::vector<float>& base,
  const std::vector<float>& query,
  int n_base,
  int n_query,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int n_threads,
  bool self_search = false,
  bool include_self = true,
  detail::NativeHNSWIndex* retained_index = nullptr
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("HNSW graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  const std::vector<float>* xb = &base;
  const std::vector<float>* xq = &query;
  std::vector<float> normalized_base;
  std::vector<float> normalized_query;
  if (metric == DistanceMetric::Cosine) {
    normalized_base = base;
    normalize_rows_for_cosine(
      normalized_base,
      static_cast<std::size_t>(n_base),
      static_cast<std::size_t>(dim)
    );
    xb = &normalized_base;
    if (&base == &query && n_base == n_query) {
      xq = xb;
    } else {
      normalized_query = query;
      normalize_rows_for_cosine(
        normalized_query,
        static_cast<std::size_t>(n_query),
        static_cast<std::size_t>(dim)
      );
      xq = &normalized_query;
    }
  }

  std::vector<int> query_train_indices;
  if (self_search && !include_self) {
    query_train_indices.resize(static_cast<std::size_t>(n_query));
    std::iota(query_train_indices.begin(), query_train_indices.end(), 0);
  }
  const int m = std::min(32, std::max(2, n_base > 1 ? n_base - 1 : 2));
  const detail::NativeHNSWParameters parameters{
    m, std::max(200, m), std::max(150, neighbors)
  };
  detail::NativeKNNResult search;
  if (retained_index != nullptr && self_search && n_base == n_query) {
    *retained_index = detail::native_build_hnsw_index(
      *xb, n_base, dim, metric, parameters, n_threads
    );
    search = detail::native_hnsw_index_search(
      *retained_index, *xq,
      n_query, neighbors, n_threads, query_train_indices
    );
  } else {
    search = detail::native_hnsw_search(
      *xb, n_base, *xq, n_query, dim, neighbors, metric,
      parameters, n_threads, query_train_indices
    );
  }

  NeighborGraph graph;
  graph.neighbors = search.neighbors;
  graph.index_base = GraphIndexBase::Zero;
  graph.indices = search.indices;
  graph.distances.resize(search.distances.size());
  for (std::size_t i = 0; i < search.distances.size(); ++i) {
    graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
  }
  return graph;
}

#ifdef KODAMA_ENABLE_CUDA
NeighborGraph native_cuda_flat_graph(
  const std::vector<float>& base,
  const std::vector<float>& query,
  int n_base,
  int n_query,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int gpu_device,
  bool exclude_self
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("CUDA graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  std::vector<float> xb = base;
  std::vector<float> xq = query;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(xb, static_cast<std::size_t>(n_base), static_cast<std::size_t>(dim));
    normalize_rows_for_cosine(xq, static_cast<std::size_t>(n_query), static_cast<std::size_t>(dim));
  }

  std::vector<int> exclusions;
  if (exclude_self) {
    exclusions.resize(static_cast<std::size_t>(n_query));
    std::iota(exclusions.begin(), exclusions.end(), 0);
  }
  const detail::NativeKNNResult search = detail::native_cuda_exact_knn_search(
    xb,
    n_base,
    xq,
    n_query,
    dim,
    neighbors,
    metric,
    gpu_device,
    exclusions
  );
  NeighborGraph graph;
  graph.neighbors = search.neighbors;
  graph.index_base = GraphIndexBase::Zero;
  graph.indices = search.indices;
  graph.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.distances.size(); ++i) {
    graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
  }
  return graph;
}

NeighborGraph native_cuda_ivf_graph(
  const std::vector<float>& data,
  int n,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int gpu_device,
  bool include_self,
  int requested_nlist,
  int requested_nprobe,
  detail::NativeCudaIVFStats* stats
) {
  std::vector<float> prepared = data;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(prepared, static_cast<std::size_t>(n), static_cast<std::size_t>(dim));
  }
  std::vector<int> exclusions;
  if (!include_self) {
    exclusions.resize(static_cast<std::size_t>(n));
    std::iota(exclusions.begin(), exclusions.end(), 0);
  }
  detail::NativeCudaIVFIndex index = detail::native_cuda_build_ivf_index(
    prepared,
    n,
    dim,
    metric,
    requested_nlist,
    gpu_device
  );
  const detail::NativeKNNResult search = detail::native_cuda_ivf_index_self_search(
    index,
    neighbors,
    requested_nprobe,
    kKODAMAGraphTargetRecall,
    exclusions,
    stats
  );
  NeighborGraph graph;
  graph.neighbors = search.neighbors;
  graph.index_base = GraphIndexBase::Zero;
  graph.indices = search.indices;
  graph.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.distances.size(); ++i) {
    graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
  }
  return graph;
}
#endif

NeighborGraph self_knn_graph(
  const std::vector<float>& data,
  int n,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int n_threads,
  Backend backend,
  int gpu_device,
  bool include_self,
  KNNIndexType index_type,
  int ivf_nlist,
  int ivf_nprobe,
  int* used_nlist = nullptr,
  int* used_nprobe = nullptr,
  double* pilot_recall = nullptr,
  KNNIndexType* used_index_type = nullptr
) {
#if !defined(KODAMA_ENABLE_CUDA) && !defined(KODAMA_ENABLE_METAL)
  (void)index_type;
  (void)ivf_nlist;
  (void)ivf_nprobe;
  (void)used_nlist;
  (void)used_nprobe;
  (void)pilot_recall;
#endif
  if (detail::should_use_spatial_grid_knn(n, dim, metric)) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend == Backend::CUDA && neighbors <= 256) {
      return detail::spatial_grid_self_knn_cuda(data, n, dim, neighbors, gpu_device, false, include_self);
    }
#else
    (void)gpu_device;
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend == Backend::Metal && neighbors <= 128) {
      const detail::NativeKNNResult search = detail::metal_spatial_grid_self_knn(
        data, n, dim, neighbors, include_self
      );
      NeighborGraph graph;
      graph.neighbors = search.neighbors;
      graph.index_base = GraphIndexBase::Zero;
      graph.indices = search.indices;
      graph.distances.resize(search.distances.size());
      for (std::size_t i = 0; i < search.distances.size(); ++i) {
        graph.distances[i] = detail::native_knn_output_distance(
          search.distances[i], DistanceMetric::Euclidean
        );
      }
      return graph;
    }
#endif
    return detail::spatial_grid_self_knn(data.data(), n, dim, neighbors, n_threads, false, include_self);
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (backend == Backend::CUDA) {
    const double exact_work =
      static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(dim);
    const bool use_ivf = index_type == KNNIndexType::CudaIVFFlat ||
      (index_type != KNNIndexType::CudaExact && n > 5000 && exact_work > 2.0e8);
    if (!use_ivf) {
      if (used_index_type != nullptr) *used_index_type = KNNIndexType::CudaExact;
      return native_cuda_flat_graph(data, data, n, n, dim, neighbors, metric, gpu_device, !include_self);
    }
    if (used_index_type != nullptr) *used_index_type = KNNIndexType::CudaIVFFlat;
    detail::NativeCudaIVFStats stats;
    NeighborGraph graph = native_cuda_ivf_graph(
      data,
      n,
      dim,
      neighbors,
      metric,
      gpu_device,
      include_self,
      ivf_nlist,
      ivf_nprobe,
      &stats
    );
    if (used_nlist != nullptr) *used_nlist = stats.nlist;
    if (used_nprobe != nullptr) *used_nprobe = stats.nprobe;
    if (pilot_recall != nullptr) *pilot_recall = stats.pilot_recall;
    return graph;
  }
#else
  (void)gpu_device;
#endif
  if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
    std::vector<float> prepared = data;
    if (metric == DistanceMetric::Cosine) normalize_rows_for_cosine(prepared, n, dim);
    std::vector<int> exclusions;
    if (!include_self) {
      exclusions.resize(static_cast<std::size_t>(n));
      std::iota(exclusions.begin(), exclusions.end(), 0);
    }
    const double exact_work =
      static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(dim);
    const bool use_ivf = index_type == KNNIndexType::MetalIVFFlat ||
      (index_type != KNNIndexType::MetalExact && n > 5000 && exact_work > 2.0e8);
    detail::MetalIVFStats stats;
    const detail::NativeKNNResult search = use_ivf ?
      detail::metal_ivf_index_self_search(
        detail::metal_build_ivf_index(prepared, n, dim, metric, ivf_nlist),
        neighbors,
        ivf_nprobe,
        kKODAMAGraphTargetRecall,
        exclusions,
        &stats
      ) :
      detail::metal_exact_knn_search(
        prepared,
        n,
        prepared,
        n,
        dim,
        neighbors,
        metric,
        exclusions
      );
    if (used_index_type != nullptr) {
      *used_index_type = use_ivf ? KNNIndexType::MetalIVFFlat : KNNIndexType::MetalExact;
    }
    if (use_ivf) {
      if (used_nlist != nullptr) *used_nlist = stats.nlist;
      if (used_nprobe != nullptr) *used_nprobe = stats.nprobe;
      if (pilot_recall != nullptr) *pilot_recall = stats.pilot_recall;
    }
    NeighborGraph graph;
    graph.neighbors = search.neighbors;
    graph.index_base = GraphIndexBase::Zero;
    graph.indices = search.indices;
    graph.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < search.distances.size(); ++i) {
      graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
    }
    return graph;
#else
    throw std::runtime_error("KODAMA Metal graph construction requires an Apple Metal build.");
#endif
  }
  if (used_index_type != nullptr) *used_index_type = KNNIndexType::NativeHNSW;
  return hnsw_graph(data, data, n, n, dim, neighbors, metric, n_threads, true, include_self);
}

class FaissCompatibleRandom {
 public:
  explicit FaissCompatibleRandom(std::uint64_t seed)
      : generator_(static_cast<std::uint32_t>(seed)) {}

  int integer(int maximum) {
    return static_cast<int>(generator_() % static_cast<std::uint32_t>(maximum));
  }

  float uniform() {
    return static_cast<float>(generator_()) / static_cast<float>(std::mt19937::max());
  }

 private:
  std::mt19937 generator_;
};

std::vector<int> faiss_compatible_permutation(int rows, std::uint64_t seed) {
  std::vector<int> permutation(static_cast<std::size_t>(rows));
  std::iota(permutation.begin(), permutation.end(), 0);
  FaissCompatibleRandom generator(seed);
  for (int row = 0; row + 1 < rows; ++row) {
    const int replacement = row + generator.integer(rows - row);
    std::swap(
      permutation[static_cast<std::size_t>(row)],
      permutation[static_cast<std::size_t>(replacement)]
    );
  }
  return permutation;
}

class FloatFenwickTree {
 public:
  explicit FloatFenwickTree(const std::vector<float>& values)
      : tree_(values.size() + 1, 0.0f), values_(values.size(), 0.0f) {
    for (std::size_t index = 0; index < values.size(); ++index) set(index, values[index]);
  }

  void set(std::size_t index, float value) {
    const float delta = value - values_[index];
    values_[index] = value;
    for (++index; index < tree_.size(); index += index & (~index + 1)) tree_[index] += delta;
  }

  float total() const {
    float result = 0.0f;
    for (std::size_t index = values_.size(); index > 0; index -= index & (~index + 1)) result += tree_[index];
    return result;
  }

  int select(float target) const {
    std::size_t index = 0;
    float prefix = 0.0f;
    std::size_t step = 1;
    while ((step << 1) < tree_.size()) step <<= 1;
    for (; step > 0; step >>= 1) {
      const std::size_t next = index + step;
      if (next < tree_.size() && prefix + tree_[next] <= target) {
        index = next;
        prefix += tree_[next];
      }
    }
    return index < values_.size() ? static_cast<int>(index) : -1;
  }

 private:
  std::vector<float> tree_;
  std::vector<float> values_;
};

void split_empty_kmeans_clusters(
  int rows,
  int dimensions,
  int clusters,
  std::vector<float>& cluster_sizes,
  std::vector<float>& centroids
) {
  if (rows <= clusters) return;
  constexpr float epsilon = 1.0f / 1024.0f;
  FaissCompatibleRandom generator(1234u);
  std::vector<float> donor_weights(static_cast<std::size_t>(clusters), 0.0f);
  for (int cluster = 0; cluster < clusters; ++cluster) {
    donor_weights[static_cast<std::size_t>(cluster)] = std::max(
      0.0f,
      cluster_sizes[static_cast<std::size_t>(cluster)] - 1.0f
    );
  }
  FloatFenwickTree donor_tree(donor_weights);
  for (int empty = 0; empty < clusters; ++empty) {
    if (cluster_sizes[static_cast<std::size_t>(empty)] != 0.0f) continue;
    const float total_weight = donor_tree.total();
    int donor = total_weight > 0.0f ? donor_tree.select(generator.uniform() * total_weight) : -1;
    if (donor < 0) {
      donor = 0;
      for (int cluster = 1; cluster < clusters; ++cluster) {
        if (cluster_sizes[static_cast<std::size_t>(cluster)] >
            cluster_sizes[static_cast<std::size_t>(donor)]) {
          donor = cluster;
        }
      }
    }

    float* empty_centroid = centroids.data() + static_cast<std::size_t>(empty) * dimensions;
    float* donor_centroid = centroids.data() + static_cast<std::size_t>(donor) * dimensions;
    std::copy_n(donor_centroid, dimensions, empty_centroid);
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      if ((dimension & 1) == 0) {
        empty_centroid[dimension] *= 1.0f + epsilon;
        donor_centroid[dimension] *= 1.0f - epsilon;
      } else {
        empty_centroid[dimension] *= 1.0f - epsilon;
        donor_centroid[dimension] *= 1.0f + epsilon;
      }
    }
    cluster_sizes[static_cast<std::size_t>(empty)] =
      cluster_sizes[static_cast<std::size_t>(donor)] / 2.0f;
    cluster_sizes[static_cast<std::size_t>(donor)] -=
      cluster_sizes[static_cast<std::size_t>(empty)];
    donor_tree.set(
      static_cast<std::size_t>(empty),
      std::max(0.0f, cluster_sizes[static_cast<std::size_t>(empty)] - 1.0f)
    );
    donor_tree.set(
      static_cast<std::size_t>(donor),
      std::max(0.0f, cluster_sizes[static_cast<std::size_t>(donor)] - 1.0f)
    );
  }
}

std::vector<int> kmeans_labels(
  const std::vector<float>& x,
  int n,
  int p,
  int k,
  std::mt19937_64& rng,
  int max_iter = 10,
  int n_threads = 1,
  Backend backend = Backend::CPU,
  int gpu_device = 0,
  int worker_lane = 0,
  detail::NativeCudaKMeansContext* cuda_context = nullptr,
  detail::NativeMetalKMeansContext* metal_context = nullptr
) {
  k = std::max(1, std::min(k, n));
#if !defined(KODAMA_ENABLE_METAL)
  (void)metal_context;
#endif

  if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    if (metal_context != nullptr) {
      return detail::metal_kmeans_context_labels(
        *metal_context, worker_lane, k, order, std::max(1, max_iter)
      );
    }
    return detail::metal_kmeans_labels(x, n, p, k, order, std::max(1, max_iter));
#else
    throw std::runtime_error("KODAMA Metal k-means requires an Apple Metal build.");
#endif
  }

  const std::uint64_t kmeans_seed = rng() & 0x7fffffffULL;

#ifdef KODAMA_ENABLE_CUDA
  if (backend == Backend::CUDA) {
    if (cuda_context != nullptr) {
      return detail::native_cuda_kmeans_context_labels(
        *cuda_context, worker_lane, k, std::max(1, max_iter), kmeans_seed
      );
    }
    return detail::native_cuda_kmeans_labels(
      x,
      n,
      p,
      k,
      std::max(1, max_iter),
      kmeans_seed,
      gpu_device
    );
  }
#else
  (void)gpu_device;
  (void)cuda_context;
#endif

  const std::vector<int> order = faiss_compatible_permutation(n, kmeans_seed + 1u);
  std::vector<float> centroids(static_cast<std::size_t>(k) * static_cast<std::size_t>(p), 0.0f);
  for (int cluster = 0; cluster < k; ++cluster) {
    std::copy_n(
      x.data() + static_cast<std::size_t>(order[static_cast<std::size_t>(cluster)]) * static_cast<std::size_t>(p),
      p,
      centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p)
    );
  }

  std::vector<int> assignments(static_cast<std::size_t>(n), -1);
  std::vector<float> point_norms(static_cast<std::size_t>(n), 0.0f);
  std::vector<float> centroid_norms(static_cast<std::size_t>(k), 0.0f);
  for (int row = 0; row < n; ++row) {
    const float* point = x.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
    point_norms[static_cast<std::size_t>(row)] = squared_norm_float32(point, p);
  }
  std::vector<float> cluster_sizes(static_cast<std::size_t>(k), 0.0f);
  OmpThreadScope threads(n_threads);

  auto assign = [&]() {
    for (int cluster = 0; cluster < k; ++cluster) {
      const float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      centroid_norms[static_cast<std::size_t>(cluster)] = squared_norm_float32(centroid, p);
    }
    std::atomic<int> changed{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < n; ++row) {
      int best_cluster = 0;
      float best_distance = std::numeric_limits<float>::infinity();
      const float* point = x.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
      for (int cluster = 0; cluster < k; ++cluster) {
        const float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
        const float dot = dot_float32(point, centroid, p);
        const float distance = point_norms[static_cast<std::size_t>(row)] +
          centroid_norms[static_cast<std::size_t>(cluster)] - 2.0f * dot;
        if (distance < best_distance || (distance == best_distance && cluster < best_cluster)) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      if (assignments[static_cast<std::size_t>(row)] != best_cluster) {
        assignments[static_cast<std::size_t>(row)] = best_cluster;
        changed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return changed.load(std::memory_order_relaxed);
  };

  for (int iteration = 0; iteration < std::max(1, max_iter); ++iteration) {
    const int changed = assign();
    std::fill(centroids.begin(), centroids.end(), 0.0f);
    std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0.0f);
    for (int row = 0; row < n; ++row) {
      const int cluster = assignments[static_cast<std::size_t>(row)];
      cluster_sizes[static_cast<std::size_t>(cluster)] += 1.0f;
      const float* point = x.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
      float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      for (int feature = 0; feature < p; ++feature) centroid[feature] += point[feature];
    }
    for (int cluster = 0; cluster < k; ++cluster) {
      float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      const float size = cluster_sizes[static_cast<std::size_t>(cluster)];
      if (size == 0.0f) continue;
      const float scale = 1.0f / size;
      for (int feature = 0; feature < p; ++feature) centroid[feature] *= scale;
    }
    split_empty_kmeans_clusters(n, p, k, cluster_sizes, centroids);
    if (changed == 0) break;
  }
  assign();
  std::vector<int> labels(static_cast<std::size_t>(n), 1);
  for (int row = 0; row < n; ++row) labels[static_cast<std::size_t>(row)] = assignments[static_cast<std::size_t>(row)] + 1;
  return labels;
}

struct IndexedStrata {
  std::vector<int> offsets;
  std::vector<int> rows;
};

IndexedStrata index_strata(const std::vector<int>& labels, int strata) {
  if (strata < 1) throw std::invalid_argument("Landmark strata count must be positive.");
  IndexedStrata out;
  out.offsets.assign(static_cast<std::size_t>(strata) + 1, 0);
  out.rows.resize(labels.size());
  for (int label : labels) {
    if (label < 1 || label > strata) throw std::invalid_argument("Landmark stratum label is out of range.");
    ++out.offsets[static_cast<std::size_t>(label)];
  }
  for (int stratum = 1; stratum <= strata; ++stratum) {
    out.offsets[static_cast<std::size_t>(stratum)] +=
      out.offsets[static_cast<std::size_t>(stratum - 1)];
  }
  std::vector<int> cursor = out.offsets;
  for (std::size_t row = 0; row < labels.size(); ++row) {
    const int stratum = labels[row] - 1;
    out.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(stratum)]++)] =
      static_cast<int>(row);
  }
  return out;
}

struct LandmarkSample {
  std::vector<int> rows;
  int occupied_strata = 0;
  int represented_strata = 0;
  int grid_bins = 0;
};

std::vector<int> intersect_groupings(
  const std::vector<int>& groups,
  const std::vector<int>& samples
) {
  if (samples.empty()) return groups;
  if (groups.size() != samples.size()) {
    throw std::invalid_argument("Grouping and samples sizes must match.");
  }
  std::map<std::pair<int, int>, int> compact;
  std::vector<int> out(groups.size(), 0);
  for (std::size_t row = 0; row < groups.size(); ++row) {
    const std::pair<int, int> key{samples[row], groups[row]};
    auto it = compact.find(key);
    if (it == compact.end()) {
      it = compact.emplace(key, static_cast<int>(compact.size()) + 1).first;
    }
    out[row] = it->second;
  }
  return out;
}

LandmarkSample quota_sample_landmarks(
  const std::vector<int>& offsets,
  const std::vector<int>& rows,
  int target,
  std::mt19937_64& rng,
  bool preserve_strata_order
) {
  if (offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != static_cast<int>(rows.size())) {
    throw std::invalid_argument("Invalid landmark strata index.");
  }
  const int samples = static_cast<int>(rows.size());
  target = std::max(1, std::min(target, samples));
  const int strata = static_cast<int>(offsets.size()) - 1;
  std::vector<int> quota(static_cast<std::size_t>(strata), 0);
  std::vector<long double> fractional(static_cast<std::size_t>(strata), 0.0L);
  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(strata));

  int assigned = 0;
  for (int stratum = 0; stratum < strata; ++stratum) {
    const int count =
      offsets[static_cast<std::size_t>(stratum + 1)] -
      offsets[static_cast<std::size_t>(stratum)];
    if (count <= 0) continue;
    order.push_back(stratum);
    const long double expected =
      static_cast<long double>(target) * static_cast<long double>(count) /
      static_cast<long double>(samples);
    const int base = std::min(count, static_cast<int>(std::floor(expected)));
    quota[static_cast<std::size_t>(stratum)] = base;
    fractional[static_cast<std::size_t>(stratum)] = expected - static_cast<long double>(base);
    assigned += base;
  }

  if (!preserve_strata_order) std::shuffle(order.begin(), order.end(), rng);
  const int residual_slots = target - assigned;
  if (residual_slots > 0) {
    const long double offset = std::generate_canonical<long double, 64>(rng);
    long double cumulative = 0.0L;
    int selected = 0;
    std::vector<char> rounded_up(static_cast<std::size_t>(strata), 0);
    for (int stratum : order) {
      cumulative += fractional[static_cast<std::size_t>(stratum)];
      const long double next = offset + static_cast<long double>(selected);
      if (selected < residual_slots && next < cumulative) {
        ++quota[static_cast<std::size_t>(stratum)];
        rounded_up[static_cast<std::size_t>(stratum)] = 1;
        ++selected;
      }
    }

    // Floating-point summation can miss only the final boundary. Fill such slots
    // by the largest unrounded residuals, preserving exact sample size.
    while (selected < residual_slots) {
      int best = -1;
      long double best_fraction = -1.0L;
      for (int stratum : order) {
        if (rounded_up[static_cast<std::size_t>(stratum)]) continue;
        const long double value = fractional[static_cast<std::size_t>(stratum)];
        if (value > best_fraction) {
          best_fraction = value;
          best = stratum;
        }
      }
      if (best < 0) throw std::runtime_error("Unable to complete the landmark quota allocation.");
      ++quota[static_cast<std::size_t>(best)];
      rounded_up[static_cast<std::size_t>(best)] = 1;
      ++selected;
    }
  }

  LandmarkSample out;
  out.occupied_strata = static_cast<int>(order.size());
  out.rows.reserve(static_cast<std::size_t>(target));
  for (int stratum = 0; stratum < strata; ++stratum) {
    const int take = quota[static_cast<std::size_t>(stratum)];
    if (take <= 0) continue;
    ++out.represented_strata;
    const auto first = rows.begin() + offsets[static_cast<std::size_t>(stratum)];
    const auto last = rows.begin() + offsets[static_cast<std::size_t>(stratum + 1)];
    std::sample(first, last, std::back_inserter(out.rows), take, rng);
  }
  if (static_cast<int>(out.rows.size()) != target) {
    throw std::runtime_error("Landmark quota allocation produced the wrong sample size.");
  }
  std::sort(out.rows.begin(), out.rows.end());
  return out;
}

LandmarkSample spatial_grid_landmarks(
  const std::vector<float>& spatial,
  int samples,
  int dimensions,
  int target,
  std::mt19937_64& rng,
  const std::vector<int>& sample_ids
) {
  if (dimensions != 2 && dimensions != 3) {
    throw std::invalid_argument("Spatial grid landmark selection supports 2D or 3D coordinates.");
  }
  const double root = std::pow(
    static_cast<double>(std::max(1, target)),
    1.0 / static_cast<double>(dimensions)
  );
  const int bins = std::max(1, std::min(4096, static_cast<int>(std::ceil(root))));
  const detail::SpatialGridIndex grid =
    detail::build_spatial_grid_index(spatial.data(), samples, dimensions, bins);
  IndexedStrata strata{grid.offsets, grid.rows};
  if (!sample_ids.empty()) {
    if (sample_ids.size() != static_cast<std::size_t>(samples)) {
      throw std::invalid_argument("samples size must match spatial landmark rows.");
    }
    std::vector<int> grid_labels(static_cast<std::size_t>(samples), 0);
    for (std::size_t cell = 0; cell + 1 < grid.offsets.size(); ++cell) {
      for (int cursor = grid.offsets[cell]; cursor < grid.offsets[cell + 1]; ++cursor) {
        grid_labels[static_cast<std::size_t>(grid.rows[static_cast<std::size_t>(cursor)])] =
          static_cast<int>(cell) + 1;
      }
    }
    const std::vector<int> sample_grid_labels =
      intersect_groupings(grid_labels, sample_ids);
    const int sample_grid_strata = *std::max_element(
      sample_grid_labels.begin(), sample_grid_labels.end());
    strata = index_strata(sample_grid_labels, sample_grid_strata);
  }
  LandmarkSample out = quota_sample_landmarks(
    strata.offsets, strata.rows, target, rng, true);
  out.grid_bins = bins;
  return out;
}

std::vector<int> factor_subset(const std::vector<int>& values, const std::vector<int>& rows) {
  std::map<int, int> ids;
  std::vector<int> out(rows.size(), 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int value = values[static_cast<std::size_t>(rows[i])];
    auto it = ids.find(value);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(value, id).first;
    }
    out[i] = it->second;
  }
  return out;
}

std::vector<int> constrained_majority(const std::vector<int>& labels, const std::vector<int>& constrain) {
  std::map<int, std::vector<int>> by_group;
  for (std::size_t i = 0; i < labels.size(); ++i) by_group[constrain[i]].push_back(labels[i]);
  std::map<int, int> group_label;
  for (const auto& kv : by_group) group_label[kv.first] = majority_label(kv.second);
  std::vector<int> out(labels.size(), 0);
  for (std::size_t i = 0; i < labels.size(); ++i) out[i] = group_label[constrain[i]];
  return out;
}

std::vector<int> majority_by_constrain(const std::vector<int>& values, const std::vector<int>& constrain) {
  if (values.size() != constrain.size()) throw std::invalid_argument("majority_by_constrain size mismatch.");
  std::map<int, std::map<int, int>> counts;
  for (std::size_t i = 0; i < values.size(); ++i) {
    counts[constrain[i]][values[i]] += 1;
  }
  std::map<int, int> group_value;
  for (const auto& group : counts) {
    int best_value = group.second.begin()->first;
    int best_count = group.second.begin()->second;
    for (const auto& kv : group.second) {
      if (kv.second > best_count || (kv.second == best_count && kv.first < best_value)) {
        best_value = kv.first;
        best_count = kv.second;
      }
    }
    group_value[group.first] = best_value;
  }
  std::vector<int> out(values.size(), 0);
  for (std::size_t i = 0; i < values.size(); ++i) out[i] = group_value[constrain[i]];
  return out;
}

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> size;

  explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)), size(static_cast<std::size_t>(n), 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  }

  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (size[static_cast<std::size_t>(a)] < size[static_cast<std::size_t>(b)]) std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    size[static_cast<std::size_t>(a)] += size[static_cast<std::size_t>(b)];
    return true;
  }
};

std::vector<int> spatial_graph_components(
  const std::vector<float>& spatial,
  int n,
  int dims,
  int target_components,
  int n_threads,
  Backend backend,
  int gpu_device
) {
  target_components = std::max(1, std::min(target_components, n));
  if (target_components >= n) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::iota(out.begin(), out.end(), 1);
    return out;
  }
  const int k = std::max(2, std::min(n, 32));
  const NeighborGraph graph = self_knn_graph(
    spatial,
    n,
    dims,
    k,
    DistanceMetric::Euclidean,
    n_threads,
    backend,
    gpu_device,
    true,
    KNNIndexType::MetalExact,
    0,
    0
  );
  struct Edge {
    float distance;
    int a;
    int b;
  };
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * graph.neighbors + static_cast<std::size_t>(j);
      const int b = graph.indices[offset];
      if (b < 0 || b == i || b < i) continue;
      edges.push_back({graph.distances[offset], i, b});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
    if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    return lhs.b < rhs.b;
  });
  DisjointSet dsu(n);
  int components = n;
  for (const Edge& edge : edges) {
    if (components <= target_components) break;
    if (dsu.unite(edge.a, edge.b)) --components;
  }

  std::map<int, int> ids;
  std::vector<int> out(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int root = dsu.find(i);
    auto it = ids.find(root);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(root, id).first;
    }
    out[static_cast<std::size_t>(i)] = it->second;
  }
  return out;
}

std::vector<float> spatial_jitter_from_graph(
  const std::vector<float>& spatial,
  int n,
  int dims,
  int neighbors,
  int n_threads,
  Backend backend,
  int gpu_device
) {
  const int k = std::max(1, std::min(neighbors, n));
  NeighborGraph graph = self_knn_graph(
    spatial,
    n,
    dims,
    k,
    DistanceMetric::Euclidean,
    n_threads,
    backend,
    gpu_device,
    true,
    KNNIndexType::MetalExact,
    0,
    0
  );
  const bool one_based = graph.index_base == GraphIndexBase::One;
  const int far_col = std::min(19, graph.neighbors - 1);
  std::vector<double> sums(static_cast<std::size_t>(dims), 0.0);
  for (int i = 0; i < n; ++i) {
    const int near_stored = graph.indices[static_cast<std::size_t>(i) * graph.neighbors];
    const int far_stored = graph.indices[static_cast<std::size_t>(i) * graph.neighbors + static_cast<std::size_t>(far_col)];
    const int near_row = near_stored >= 0 && one_based ? near_stored - 1 : near_stored;
    const int far_row = far_stored >= 0 && one_based ? far_stored - 1 : far_stored;
    if (near_row < 0 || far_row < 0) continue;
    for (int d = 0; d < dims; ++d) {
      sums[static_cast<std::size_t>(d)] += std::abs(
        spatial[static_cast<std::size_t>(near_row) * dims + static_cast<std::size_t>(d)] -
        spatial[static_cast<std::size_t>(far_row) * dims + static_cast<std::size_t>(d)]
      );
    }
  }
  std::vector<float> jitter(static_cast<std::size_t>(dims), 0.0f);
  for (int d = 0; d < dims; ++d) jitter[static_cast<std::size_t>(d)] = static_cast<float>(3.0 * sums[static_cast<std::size_t>(d)] / std::max(1, n));
  return jitter;
}

std::vector<float> spatial_jitter_from_precomputed_graph(
  const std::vector<float>& spatial,
  const NeighborGraph& graph,
  int n,
  int dims
) {
  if (graph.neighbors < 1 ||
      graph.indices.size() != static_cast<std::size_t>(n) * graph.neighbors) {
    throw std::invalid_argument("Precomputed spatial graph dimensions are inconsistent.");
  }
  const bool one_based = graph.index_base == GraphIndexBase::One;
  const int far_col = std::min(19, graph.neighbors - 1);
  std::vector<double> sums(static_cast<std::size_t>(dims), 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t offset = static_cast<std::size_t>(i) * graph.neighbors;
    const int near_stored = graph.indices[offset];
    const int far_stored = graph.indices[offset + static_cast<std::size_t>(far_col)];
    const int near_row = near_stored >= 0 && one_based ? near_stored - 1 : near_stored;
    const int far_row = far_stored >= 0 && one_based ? far_stored - 1 : far_stored;
    if (near_row < 0 || near_row >= n || far_row < 0 || far_row >= n) continue;
    for (int d = 0; d < dims; ++d) {
      sums[static_cast<std::size_t>(d)] += std::abs(
        spatial[static_cast<std::size_t>(near_row) * dims + static_cast<std::size_t>(d)] -
        spatial[static_cast<std::size_t>(far_row) * dims + static_cast<std::size_t>(d)]
      );
    }
  }
  std::vector<float> jitter(static_cast<std::size_t>(dims), 0.0f);
  for (int d = 0; d < dims; ++d) {
    jitter[static_cast<std::size_t>(d)] =
      static_cast<float>(3.0 * sums[static_cast<std::size_t>(d)] / std::max(1, n));
  }
  return jitter;
}

void repair_singleton_spatial_clusters(
  std::vector<int>& clusters,
  const std::vector<float>& spatial,
  int n,
  int dims,
  int n_threads,
  const std::vector<int>& samples
) {
  if (!samples.empty() && samples.size() != static_cast<std::size_t>(n)) {
    throw std::invalid_argument("samples size must match spatial cluster rows.");
  }
  std::map<int, int> counts;
  for (int c : clusters) counts[c] += 1;
  std::map<int, std::vector<int>> rows_by_sample;
  for (int row = 0; row < n; ++row) {
    rows_by_sample[samples.empty() ? 1 : samples[static_cast<std::size_t>(row)]].push_back(row);
  }
  for (const auto& sample_rows : rows_by_sample) {
    std::vector<int> base_rows;
    std::vector<int> query_rows;
    for (const int row : sample_rows.second) {
      if (counts[clusters[static_cast<std::size_t>(row)]] > 1) base_rows.push_back(row);
      else query_rows.push_back(row);
    }
    if (base_rows.empty() || query_rows.empty()) continue;
    std::vector<float> base(base_rows.size() * static_cast<std::size_t>(dims), 0.0f);
    std::vector<float> query(query_rows.size() * static_cast<std::size_t>(dims), 0.0f);
    for (std::size_t i = 0; i < base_rows.size(); ++i) {
      std::copy_n(spatial.data() + static_cast<std::size_t>(base_rows[i]) * dims,
                  dims, base.data() + i * dims);
    }
    for (std::size_t i = 0; i < query_rows.size(); ++i) {
      std::copy_n(spatial.data() + static_cast<std::size_t>(query_rows[i]) * dims,
                  dims, query.data() + i * dims);
    }
    NeighborGraph nearest = detail::spatial_grid_query_nearest(
      base.data(), static_cast<int>(base_rows.size()), query.data(),
      static_cast<int>(query_rows.size()), dims, n_threads);
    for (std::size_t i = 0; i < query_rows.size(); ++i) {
      const int local = nearest.indices[i];
      if (local >= 0) {
        clusters[static_cast<std::size_t>(query_rows[i])] =
          clusters[static_cast<std::size_t>(base_rows[static_cast<std::size_t>(local)])];
      }
    }
  }
}

void trim_self_neighbors_in_place(NeighborGraph& graph, int samples, int neighbors) {
  const int input_neighbors = graph.neighbors;
  if (neighbors <= 0 || neighbors > input_neighbors) {
    throw std::invalid_argument("Requested graph width is outside the input graph width.");
  }
  for (int i = 0; i < samples; ++i) {
    int out_col = 0;
    for (int j = 0; j < input_neighbors && out_col < neighbors; ++j) {
      const std::size_t in_offset =
        static_cast<std::size_t>(i) * input_neighbors + static_cast<std::size_t>(j);
      const int id = graph.indices[in_offset];
      if (id == i) continue;
      const std::size_t out_offset = static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(out_col);
      graph.indices[out_offset] = id;
      graph.distances[out_offset] = graph.distances[in_offset];
      ++out_col;
    }
    while (out_col < neighbors) {
      const std::size_t out_offset =
        static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(out_col);
      graph.indices[out_offset] = -1;
      graph.distances[out_offset] = std::numeric_limits<float>::infinity();
      ++out_col;
    }
  }
  graph.neighbors = neighbors;
  graph.indices.resize(static_cast<std::size_t>(samples) * neighbors);
  graph.distances.resize(static_cast<std::size_t>(samples) * neighbors);
}

NeighborGraph normalize_external_graph(const NeighborGraph& graph, int samples, int max_neighbors) {
  if (samples < 2) throw std::invalid_argument("NeighborGraph samples must be at least 2.");
  if (graph.neighbors <= 0) throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  const std::size_t expected = static_cast<std::size_t>(samples) * static_cast<std::size_t>(graph.neighbors);
  if (graph.indices.size() != expected || graph.distances.size() != expected) {
    throw std::invalid_argument("NeighborGraph indices/distances size must equal samples * neighbors.");
  }
  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int id : graph.indices) {
    if (id < 0) continue;
    min_index = std::min(min_index, id);
    max_index = std::max(max_index, id);
  }
  const bool one_based = graph.index_base == GraphIndexBase::One ||
    (graph.index_base == GraphIndexBase::Auto && min_index >= 1 && max_index <= samples);
  const int neighbors = std::max(1, std::min(max_neighbors > 0 ? max_neighbors : graph.neighbors, graph.neighbors));
  NeighborGraph out;
  out.neighbors = neighbors;
  out.index_base = GraphIndexBase::Zero;
  out.indices.assign(static_cast<std::size_t>(samples) * static_cast<std::size_t>(neighbors), -1);
  out.distances.assign(static_cast<std::size_t>(samples) * static_cast<std::size_t>(neighbors), std::numeric_limits<float>::infinity());
  for (int i = 0; i < samples; ++i) {
    int out_col = 0;
    for (int j = 0; j < graph.neighbors && out_col < neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(j);
      int id = graph.indices[src];
      if (one_based && id > 0) --id;
      if (id < 0 || id >= samples || id == i) continue;
      const float d = graph.distances[src];
      if (!std::isfinite(d)) continue;
      const std::size_t dst = static_cast<std::size_t>(i) * static_cast<std::size_t>(neighbors) + static_cast<std::size_t>(out_col);
      out.indices[dst] = id;
      out.distances[dst] = std::max(0.0f, d);
      ++out_col;
    }
  }
  return out;
}

void subset_graph_to_rows(
  const NeighborGraph& graph,
  const std::vector<int>& rows,
  int samples,
  std::vector<int>& global_to_local,
  NeighborGraph& out
) {
  global_to_local.assign(static_cast<std::size_t>(samples), -1);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    global_to_local[static_cast<std::size_t>(rows[i])] = static_cast<int>(i);
  }
  out.neighbors = graph.neighbors;
  out.indices.assign(rows.size() * static_cast<std::size_t>(graph.neighbors), -1);
  out.distances.assign(rows.size() * static_cast<std::size_t>(graph.neighbors), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int row = rows[i];
    int out_col = 0;
    for (int j = 0; j < graph.neighbors && out_col < graph.neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(row) * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(j);
      const int global_nb = graph.indices[src];
      if (global_nb < 0 || global_nb >= samples) continue;
      const int local_nb = global_to_local[static_cast<std::size_t>(global_nb)];
      if (local_nb < 0 || local_nb == static_cast<int>(i)) continue;
      const std::size_t dst = i * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(out_col);
      // CoreKNNGraph_CPU accepts public-style graphs and normalizes 1-based
      // indices at its boundary. Emit the local subset graph in that form so
      // index-base auto detection cannot misclassify a zero-based subset when
      // local index 0 is absent from the neighbor list.
      out.indices[dst] = local_nb + 1;
      out.distances[dst] = graph.distances[src];
      ++out_col;
    }
  }
}

struct SparseGraphOperator {
  int samples = 0;
  std::vector<int> indptr;
  std::vector<int> indices;
  std::vector<float> weights;
};

std::vector<float> graph_local_scales(const NeighborGraph& graph, int samples) {
  std::vector<float> scales(static_cast<std::size_t>(samples), 1.0f);
  std::vector<float> row_dist;
  row_dist.reserve(static_cast<std::size_t>(graph.neighbors));
  for (int i = 0; i < samples; ++i) {
    row_dist.clear();
    const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors);
    for (int j = 0; j < graph.neighbors; ++j) {
      const int id = graph.indices[base + static_cast<std::size_t>(j)];
      const float d = graph.distances[base + static_cast<std::size_t>(j)];
      if (id < 0 || !std::isfinite(d)) continue;
      row_dist.push_back(std::max(0.0f, d));
    }
    if (row_dist.empty()) continue;
    const std::size_t mid = row_dist.size() / 2;
    std::nth_element(row_dist.begin(), row_dist.begin() + static_cast<std::ptrdiff_t>(mid), row_dist.end());
    float scale = row_dist[mid];
    if (scale <= 1.0e-6f) {
      double mean = 0.0;
      for (float d : row_dist) mean += static_cast<double>(d);
      scale = static_cast<float>(mean / static_cast<double>(row_dist.size()));
    }
    scales[static_cast<std::size_t>(i)] = std::max(scale, 1.0e-6f);
  }
  return scales;
}

SparseGraphOperator make_sparse_graph_operator(
  const NeighborGraph& graph,
  int samples,
  bool symmetrize,
  bool self_tuning,
  bool symmetric_normalize
) {
  NeighborGraph g = normalize_external_graph(graph, samples, graph.neighbors);
  const std::vector<float> scales = self_tuning ? graph_local_scales(g, samples) : std::vector<float>();
  std::vector<std::unordered_map<int, float>> rows(static_cast<std::size_t>(samples));
  for (auto& row : rows) row.reserve(static_cast<std::size_t>(std::max(1, g.neighbors)));

  auto compute_weight = [&](int i, int j, float d) {
    d = std::max(0.0f, d);
    if (self_tuning) {
      const double denom =
        std::max(1.0e-12, static_cast<double>(scales[static_cast<std::size_t>(i)]) *
                          static_cast<double>(scales[static_cast<std::size_t>(j)]));
      return static_cast<float>(std::exp(-(static_cast<double>(d) * static_cast<double>(d)) / denom));
    }
    return 1.0f / (1.0f + d);
  };
  auto add_edge = [&](int from, int to, float w) {
    if (from < 0 || from >= samples || to < 0 || to >= samples || from == to || !std::isfinite(w) || w <= 0.0f) return;
    auto& row = rows[static_cast<std::size_t>(from)];
    auto it = row.find(to);
    if (it == row.end()) {
      row.emplace(to, w);
    } else {
      it->second = std::max(it->second, w);
    }
  };

  for (int i = 0; i < samples; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(g.neighbors);
    for (int j = 0; j < g.neighbors; ++j) {
      const std::size_t offset = base + static_cast<std::size_t>(j);
      const int nb = g.indices[offset];
      if (nb < 0 || nb >= samples) continue;
      const float d = g.distances[offset];
      if (!std::isfinite(d)) continue;
      const float w = compute_weight(i, nb, d);
      add_edge(i, nb, w);
      if (symmetrize) add_edge(nb, i, w);
    }
  }

  std::vector<double> degree(static_cast<std::size_t>(samples), 0.0);
  for (int i = 0; i < samples; ++i) {
    for (const auto& kv : rows[static_cast<std::size_t>(i)]) degree[static_cast<std::size_t>(i)] += static_cast<double>(kv.second);
  }

  SparseGraphOperator op;
  op.samples = samples;
  op.indptr.assign(static_cast<std::size_t>(samples) + 1, 0);
  std::size_t nnz = 0;
  for (int i = 0; i < samples; ++i) {
    op.indptr[static_cast<std::size_t>(i)] = static_cast<int>(nnz);
    nnz += rows[static_cast<std::size_t>(i)].size();
  }
  op.indptr[static_cast<std::size_t>(samples)] = static_cast<int>(nnz);
  op.indices.reserve(nnz);
  op.weights.reserve(nnz);
  for (int i = 0; i < samples; ++i) {
    std::vector<std::pair<int, float>> ordered(rows[static_cast<std::size_t>(i)].begin(), rows[static_cast<std::size_t>(i)].end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    const double row_degree = std::max(degree[static_cast<std::size_t>(i)], 1.0e-12);
    for (const auto& kv : ordered) {
      const int nb = kv.first;
      float w = kv.second;
      if (symmetric_normalize) {
        const double denom = std::sqrt(row_degree * std::max(degree[static_cast<std::size_t>(nb)], 1.0e-12));
        w = static_cast<float>(static_cast<double>(w) / denom);
      } else {
        w = static_cast<float>(static_cast<double>(w) / row_degree);
      }
      op.indices.push_back(nb);
      op.weights.push_back(w);
    }
  }
  return op;
}

void standardize_feature_columns(std::vector<float>& x, int n, int p) {
  for (int c = 0; c < p; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
    mean /= static_cast<double>(std::max(1, n));
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
      float& v = x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)];
      v = static_cast<float>(static_cast<double>(v) - mean);
      ss += static_cast<double>(v) * static_cast<double>(v);
    }
    const double scale = ss > 0.0 ? std::sqrt(ss / static_cast<double>(std::max(1, n - 1))) : 1.0;
    const float inv = static_cast<float>(1.0 / std::max(scale, 1.0e-6));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] *= inv;
  }
}

void center_feature_columns(std::vector<float>& x, int n, int p) {
  for (int c = 0; c < p; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
    mean /= static_cast<double>(std::max(1, n));
    for (int i = 0; i < n; ++i) {
      x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] =
        static_cast<float>(static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) - mean);
    }
  }
}

void orthonormalize_feature_columns(std::vector<float>& x, int n, int p) {
  center_feature_columns(x, n, p);
  for (int c = 0; c < p; ++c) {
    for (int prev = 0; prev < c; ++prev) {
      double dot = 0.0;
      for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) *
               static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(prev)]);
      }
      for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] =
          static_cast<float>(
            static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) -
            dot * static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(prev)])
          );
      }
    }
    double norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
      const double v = static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
      norm2 += v * v;
    }
    if (norm2 <= 1.0e-20) {
      for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] = 0.0f;
      if (n > 0) x[static_cast<std::size_t>(c % n) * p + static_cast<std::size_t>(c)] = 1.0f;
      center_feature_columns(x, n, p);
      norm2 = 0.0;
      for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
        norm2 += v * v;
      }
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(std::max(norm2, 1.0e-20)));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] *= inv;
  }
}

void apply_sparse_graph_operator(
  const SparseGraphOperator& op,
  const std::vector<float>& current,
  std::vector<float>& next,
  int components,
  int n_threads
) {
  std::fill(next.begin(), next.end(), 0.0f);
  OmpThreadScope threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < op.samples; ++i) {
    const std::size_t row_base = static_cast<std::size_t>(i) * static_cast<std::size_t>(components);
    for (int ptr = op.indptr[static_cast<std::size_t>(i)]; ptr < op.indptr[static_cast<std::size_t>(i + 1)]; ++ptr) {
      const int nb = op.indices[static_cast<std::size_t>(ptr)];
      const float w = op.weights[static_cast<std::size_t>(ptr)];
      const std::size_t nb_base = static_cast<std::size_t>(nb) * static_cast<std::size_t>(components);
      for (int c = 0; c < components; ++c) {
        next[row_base + static_cast<std::size_t>(c)] += w * current[nb_base + static_cast<std::size_t>(c)];
      }
    }
  }
}

std::vector<float> graph_laplacian_operator_features(
  const NeighborGraph& graph,
  int samples,
  int components,
  int iterations,
  std::uint64_t seed,
  int n_threads
) {
  const SparseGraphOperator op = make_sparse_graph_operator(
    graph,
    samples,
    true,
    true,
    true
  );
  components = std::max(1, std::min(components, samples));
  iterations = std::max(8, iterations);
  std::vector<float> current(static_cast<std::size_t>(samples) * components, 0.0f);
  std::vector<float> next(current.size(), 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (float& v : current) v = normal(rng);
  orthonormalize_feature_columns(current, samples, components);
  for (int iter = 0; iter < iterations; ++iter) {
    apply_sparse_graph_operator(op, current, next, components, n_threads);
    orthonormalize_feature_columns(next, samples, components);
    current.swap(next);
  }
  standardize_feature_columns(current, samples, components);
  return current;
}

void merge_feature_spatial_graphs_in_place(
  NeighborGraph& feature,
  const NeighborGraph& spatial,
  int samples,
  int neighbors
) {
  if (feature.neighbors != neighbors || spatial.neighbors != neighbors) {
    throw std::invalid_argument("Feature and spatial graph widths must match.");
  }
  std::vector<char> seen(static_cast<std::size_t>(samples), 0);
  std::vector<int> touched;
  touched.reserve(static_cast<std::size_t>(neighbors) * 2);
  std::vector<int> feature_row(static_cast<std::size_t>(neighbors), -1);
  std::vector<int> output_row(static_cast<std::size_t>(neighbors), -1);
  std::vector<float> output_distances(
    static_cast<std::size_t>(neighbors),
    std::numeric_limits<float>::infinity()
  );
  auto try_add = [&](int row, int id, int rank, int& out_col) {
    if (out_col >= neighbors || id < 0 || id >= samples || id == row || seen[static_cast<std::size_t>(id)]) return;
    seen[static_cast<std::size_t>(id)] = 1;
    touched.push_back(id);
    output_row[static_cast<std::size_t>(out_col)] = id;
    output_distances[static_cast<std::size_t>(out_col)] =
      static_cast<float>(rank + 1) / static_cast<float>(std::max(1, neighbors));
    ++out_col;
  };
  for (int i = 0; i < samples; ++i) {
    touched.clear();
    const std::size_t row_offset =
      static_cast<std::size_t>(i) * static_cast<std::size_t>(neighbors);
    std::copy_n(feature.indices.data() + row_offset, neighbors, feature_row.begin());
    std::fill(output_row.begin(), output_row.end(), -1);
    std::fill(
      output_distances.begin(),
      output_distances.end(),
      std::numeric_limits<float>::infinity()
    );
    int out_col = 0;
    for (int rank = 0; rank < neighbors && out_col < neighbors; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(rank);
      try_add(i, feature_row[static_cast<std::size_t>(rank)], rank, out_col);
      try_add(i, spatial.indices[offset], rank, out_col);
    }
    std::copy(output_row.begin(), output_row.end(), feature.indices.begin() + row_offset);
    std::copy(
      output_distances.begin(),
      output_distances.end(),
      feature.distances.begin() + row_offset
    );
    for (int id : touched) seen[static_cast<std::size_t>(id)] = 0;
  }
}

struct IterationResult {
  std::vector<int> res;
  std::vector<int> constrain;
  std::vector<double> acc;
  CoreRunDiagnostic run_diagnostic;
  std::vector<CoreCycleDiagnostic> cycle_diagnostics;
  int landmarks_used = 0;
  int landmark_occupied_strata = 0;
  int landmark_represented_strata = 0;
  int landmark_grid_bins = 0;
  double landmark_seconds = 0.0;
  double coarse_partition_seconds = 0.0;
  double landmark_sampling_seconds = 0.0;
  double constraint_seconds = 0.0;
  double landmark_prepare_seconds = 0.0;
  double landmark_initialization_seconds = 0.0;
  double landmark_graph_seconds = 0.0;
  double core_evolution_seconds = 0.0;
  double projection_seconds = 0.0;
  double accbest = std::numeric_limits<double>::quiet_NaN();
  double runtime = 0.0;
  double memory = 0.0;
  bool resident_result = false;
  bool sparse_projection_upload = false;
  bool full_projection_download = false;
  int resident_row_uploads = 0;
};

std::uint64_t hash_integer_vector(const std::vector<int>& values) {
  std::uint64_t hash = 1469598103934665603ull;
  for (int value : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xffu);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct IterationScratch {
  std::vector<int> cluster_counts;
  std::vector<int> selected_landpoints;
  std::vector<int> landpoints;
  std::vector<int> coarse_labels;
  std::vector<char> is_landmark;
  std::vector<int> tpoints;
  std::vector<float> x_land;
  std::vector<float> x_test;
  std::vector<int> x_constrain;
  std::vector<int> x_fixed;
  std::vector<int> tmp_labels;
  std::vector<int> xw;
  std::vector<int> init;
  std::vector<int> global_to_local;
  std::vector<int> projection_votes;
  std::vector<float> spatial_jittered;
  std::vector<float> spatial_regularized;
  std::vector<int> spatial_clusters;
  std::vector<int> run_constrain;
  NeighborGraph local_graph;
};

float median_float(std::vector<float> values) {
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const float upper = values[middle];
  if (values.size() % 2u != 0u) return upper;
  std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
  return 0.5f * (values[middle - 1] + upper);
}

std::vector<int> identical_coordinate_groups(
  const std::vector<float>& spatial,
  const int rows,
  const int dimensions
) {
  std::map<std::pair<float, float>, int> ids;
  std::vector<int> groups(static_cast<std::size_t>(rows), 0);
  int next = 0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * dimensions;
    const std::pair<float, float> key{
      spatial[offset], spatial[offset + 1]
    };
    const auto inserted = ids.emplace(key, next);
    if (inserted.second) ++next;
    groups[static_cast<std::size_t>(row)] = inserted.first->second;
  }
  return groups;
}

void regularize_population_coordinates(
  const std::vector<float>& spatial,
  const int rows,
  const int dimensions,
  std::mt19937_64& rng,
  std::vector<float>& output
) {
  if (dimensions != 2) {
    throw std::invalid_argument(
      "Population spatial mode requires latitude/longitude coordinates with exactly two columns."
    );
  }
  const std::vector<int> groups = identical_coordinate_groups(spatial, rows, dimensions);
  const int group_count = groups.empty() ? 0 :
    *std::max_element(groups.begin(), groups.end()) + 1;
  if (group_count < 2) {
    throw std::invalid_argument(
      "Population spatial mode requires at least two distinct collection coordinates."
    );
  }

  output = spatial;
  std::vector<int> counts(static_cast<std::size_t>(group_count), 0);
  for (const int group : groups) ++counts[static_cast<std::size_t>(group)];
  std::vector<float> centers(static_cast<std::size_t>(group_count) * 2u, 0.0f);
  std::vector<float> distances(static_cast<std::size_t>(group_count), 0.0f);
  std::vector<int> order(static_cast<std::size_t>(group_count), 0);
  std::iota(order.begin(), order.end(), 0);
  std::vector<float> shifts(static_cast<std::size_t>(group_count) * 2u, 0.0f);

  constexpr int attractive_neighbors = 3;
  constexpr float lambda = 2.0f;
  constexpr float repulsive_power = 1.0f;
  constexpr float repulsive_scale = 10.0f;
  constexpr float eta = 0.01f;
  constexpr float tolerance = 1.0e-4f;
  constexpr float gradient_clip = 5.0f;
  constexpr int max_iterations = 200;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    std::fill(centers.begin(), centers.end(), 0.0f);
    for (int row = 0; row < rows; ++row) {
      const int group = groups[static_cast<std::size_t>(row)];
      const std::size_t source = static_cast<std::size_t>(row) * 2u;
      const std::size_t target = static_cast<std::size_t>(group) * 2u;
      centers[target] += output[source];
      centers[target + 1u] += output[source + 1u];
    }
    for (int group = 0; group < group_count; ++group) {
      const float inverse = 1.0f / static_cast<float>(counts[static_cast<std::size_t>(group)]);
      centers[static_cast<std::size_t>(group) * 2u] *= inverse;
      centers[static_cast<std::size_t>(group) * 2u + 1u] *= inverse;
    }

    float maximum_shift = 0.0f;
    for (int group = 0; group < group_count; ++group) {
      const std::size_t current = static_cast<std::size_t>(group) * 2u;
      const float x0 = centers[current];
      const float y0 = centers[current + 1u];
      for (int other = 0; other < group_count; ++other) {
        if (other == group) {
          distances[static_cast<std::size_t>(other)] = std::numeric_limits<float>::infinity();
        } else {
          const float dx = x0 - centers[static_cast<std::size_t>(other) * 2u];
          const float dy = y0 - centers[static_cast<std::size_t>(other) * 2u + 1u];
          distances[static_cast<std::size_t>(other)] = std::sqrt(dx * dx + dy * dy);
        }
        order[static_cast<std::size_t>(other)] = other;
      }
      std::sort(order.begin(), order.end(), [&](const int lhs, const int rhs) {
        if (distances[static_cast<std::size_t>(lhs)] != distances[static_cast<std::size_t>(rhs)]) {
          return distances[static_cast<std::size_t>(lhs)] < distances[static_cast<std::size_t>(rhs)];
        }
        return lhs < rhs;
      });

      const int nearest = std::min(attractive_neighbors, group_count - 1);
      float weight_sum = 0.0f;
      float weighted_x = 0.0f;
      float weighted_y = 0.0f;
      for (int j = 0; j < nearest; ++j) {
        const int neighbor = order[static_cast<std::size_t>(j)];
        const float distance = std::max(
          distances[static_cast<std::size_t>(neighbor)],
          std::numeric_limits<float>::epsilon()
        );
        const float weight = 1.0f / (distance * distance);
        weight_sum += weight;
        weighted_x += weight * centers[static_cast<std::size_t>(neighbor) * 2u];
        weighted_y += weight * centers[static_cast<std::size_t>(neighbor) * 2u + 1u];
      }
      if (!(weight_sum > 0.0f) || !std::isfinite(weight_sum)) {
        weight_sum = static_cast<float>(nearest);
        weighted_x = 0.0f;
        weighted_y = 0.0f;
        for (int j = 0; j < nearest; ++j) {
          const int neighbor = order[static_cast<std::size_t>(j)];
          weighted_x += centers[static_cast<std::size_t>(neighbor) * 2u];
          weighted_y += centers[static_cast<std::size_t>(neighbor) * 2u + 1u];
        }
      }
      weighted_x /= weight_sum;
      weighted_y /= weight_sum;
      float gradient_x = x0 - weighted_x;
      float gradient_y = y0 - weighted_y;

      for (int other = 0; other < group_count; ++other) {
        if (other == group) continue;
        const float dx = x0 - centers[static_cast<std::size_t>(other) * 2u];
        const float dy = y0 - centers[static_cast<std::size_t>(other) * 2u + 1u];
        const float radius_squared = std::max(dx * dx + dy * dy, 1.0e-12f);
        const float scale = lambda * repulsive_power *
          std::pow(repulsive_scale, repulsive_power) /
          std::pow(radius_squared, 0.5f * (repulsive_power + 2.0f));
        gradient_x -= dx * scale;
        gradient_y -= dy * scale;
      }
      const float norm = std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
      if (norm > gradient_clip) {
        const float factor = gradient_clip / norm;
        gradient_x *= factor;
        gradient_y *= factor;
      }
      const float shift_x = -eta * gradient_x;
      const float shift_y = -eta * gradient_y;
      shifts[current] = shift_x;
      shifts[current + 1u] = shift_y;
      maximum_shift = std::max(maximum_shift, std::sqrt(shift_x * shift_x + shift_y * shift_y));
    }
    for (int row = 0; row < rows; ++row) {
      const int group = groups[static_cast<std::size_t>(row)];
      const std::size_t target = static_cast<std::size_t>(row) * 2u;
      const std::size_t shift = static_cast<std::size_t>(group) * 2u;
      output[target] += shifts[shift];
      output[target + 1u] += shifts[shift + 1u];
    }
    if (maximum_shift < tolerance) break;
  }

  std::normal_distribution<float> perturbation(0.0f, 0.001f);
  for (int row = 0; row < rows; ++row) output[static_cast<std::size_t>(row) * 2u] += perturbation(rng);
  for (int row = 0; row < rows; ++row) output[static_cast<std::size_t>(row) * 2u + 1u] += perturbation(rng);

  std::fill(centers.begin(), centers.end(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const int group = groups[static_cast<std::size_t>(row)];
    centers[static_cast<std::size_t>(group) * 2u] += output[static_cast<std::size_t>(row) * 2u];
    centers[static_cast<std::size_t>(group) * 2u + 1u] += output[static_cast<std::size_t>(row) * 2u + 1u];
  }
  for (int group = 0; group < group_count; ++group) {
    const float inverse = 1.0f / static_cast<float>(counts[static_cast<std::size_t>(group)]);
    centers[static_cast<std::size_t>(group) * 2u] *= inverse;
    centers[static_cast<std::size_t>(group) * 2u + 1u] *= inverse;
  }
  std::vector<std::vector<float>> radii(static_cast<std::size_t>(group_count));
  for (int row = 0; row < rows; ++row) {
    const int group = groups[static_cast<std::size_t>(row)];
    const float dx = output[static_cast<std::size_t>(row) * 2u] - centers[static_cast<std::size_t>(group) * 2u];
    const float dy = output[static_cast<std::size_t>(row) * 2u + 1u] - centers[static_cast<std::size_t>(group) * 2u + 1u];
    radii[static_cast<std::size_t>(group)].push_back(std::sqrt(dx * dx + dy * dy));
  }
  std::vector<float> group_radii(static_cast<std::size_t>(group_count), 0.0f);
  for (int group = 0; group < group_count; ++group) {
    group_radii[static_cast<std::size_t>(group)] = median_float(radii[static_cast<std::size_t>(group)]);
  }
  const float target_radius = median_float(group_radii);
  std::vector<float> nearest_centroid(static_cast<std::size_t>(group_count),
    std::numeric_limits<float>::infinity());
  for (int group = 0; group < group_count; ++group) {
    for (int other = 0; other < group_count; ++other) {
      if (other == group) continue;
      const float dx = centers[static_cast<std::size_t>(group) * 2u] - centers[static_cast<std::size_t>(other) * 2u];
      const float dy = centers[static_cast<std::size_t>(group) * 2u + 1u] - centers[static_cast<std::size_t>(other) * 2u + 1u];
      nearest_centroid[static_cast<std::size_t>(group)] = std::min(
        nearest_centroid[static_cast<std::size_t>(group)], std::sqrt(dx * dx + dy * dy));
    }
  }
  const float median_nearest = median_float(nearest_centroid);
  constexpr float epsilon = 1.0e-8f;
  const float alpha = std::isfinite(median_nearest) && median_nearest > epsilon ?
    2.0f * target_radius / median_nearest : 1.0f;
  float global_x = 0.0f;
  float global_y = 0.0f;
  for (int group = 0; group < group_count; ++group) {
    global_x += centers[static_cast<std::size_t>(group) * 2u];
    global_y += centers[static_cast<std::size_t>(group) * 2u + 1u];
  }
  global_x /= static_cast<float>(group_count);
  global_y /= static_cast<float>(group_count);
  for (int row = 0; row < rows; ++row) {
    const int group = groups[static_cast<std::size_t>(row)];
    const float center_x = centers[static_cast<std::size_t>(group) * 2u];
    const float center_y = centers[static_cast<std::size_t>(group) * 2u + 1u];
    const float scaled_x = global_x + alpha * (center_x - global_x);
    const float scaled_y = global_y + alpha * (center_y - global_y);
    const float beta = target_radius /
      std::max(group_radii[static_cast<std::size_t>(group)], epsilon);
    output[static_cast<std::size_t>(row) * 2u] = scaled_x + beta *
      (output[static_cast<std::size_t>(row) * 2u] - center_x);
    output[static_cast<std::size_t>(row) * 2u + 1u] = scaled_y + beta *
      (output[static_cast<std::size_t>(row) * 2u + 1u] - center_y);
  }
}

class DeviceResidentKODAMAGraph {
 public:
  bool has_landmark_index() const noexcept {
    if (!landmark_index_available_) return false;
    if (backend_ == Backend::CPU) return cpu_hnsw_ != nullptr && cpu_hnsw_->valid();
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA) return cuda_ != nullptr && cuda_->has_landmark_index();
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal) return metal_ != nullptr && metal_->has_landmark_index();
#endif
    return false;
  }

  NeighborGraph landmark_knn_graph(
    const std::vector<float>& landmark_data,
    const std::vector<int>& landmark_rows,
    int k,
    const KNNOptions& options
  ) const {
    if (backend_ == Backend::CPU && cpu_hnsw_ != nullptr) {
      std::vector<float> prepared = landmark_data;
      if (cpu_hnsw_->metric() == DistanceMetric::Cosine) {
        normalize_rows_for_cosine(
          prepared, landmark_rows.size(), static_cast<std::size_t>(cpu_hnsw_->dimensions())
        );
      }
      std::vector<int> allowed_local_ids(static_cast<std::size_t>(cpu_hnsw_->rows()), -1);
      for (std::size_t local = 0; local < landmark_rows.size(); ++local) {
        const int global = landmark_rows[local];
        if (global < 0 || global >= cpu_hnsw_->rows()) {
          throw std::out_of_range("CPU landmark row is outside the retained HNSW index.");
        }
        allowed_local_ids[static_cast<std::size_t>(global)] = static_cast<int>(local);
      }
      detail::NativeKNNResult result = detail::native_hnsw_index_filtered_search(
        *cpu_hnsw_, prepared, static_cast<int>(landmark_rows.size()), k,
        std::max(1, options.n_threads), landmark_rows, allowed_local_ids
      );
      NeighborGraph output;
      output.neighbors = result.neighbors;
      output.index_base = GraphIndexBase::Zero;
      output.indices = std::move(result.indices);
      output.distances.resize(result.distances.size());
      for (std::size_t i = 0; i < result.distances.size(); ++i) {
        output.distances[i] = detail::native_knn_output_distance(
          result.distances[i], cpu_hnsw_->metric()
        );
      }
      return output;
    }
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_ != nullptr) {
      return detail::cuda_resident_landmark_knn_graph(
        *cuda_, landmark_data, landmark_rows, k,
        options.ivf_nprobe, options.hnsw_target_recall
      );
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_ != nullptr) {
      return detail::metal_resident_landmark_knn_graph(
        *metal_, landmark_data, landmark_rows, k,
        options.ivf_nprobe, options.hnsw_target_recall
      );
    }
#endif
    throw std::runtime_error("No reusable resident landmark index is available.");
  }

  void build_hnsw(
    const std::vector<float>& data,
    int samples,
    int dimensions,
    int neighbors,
    DistanceMetric metric,
    int n_threads
  ) {
    backend_ = Backend::CPU;
    cpu_hnsw_ = std::make_unique<detail::NativeHNSWIndex>();
    cpu_graph_ = hnsw_graph(
      data, data, samples, samples, dimensions, neighbors, metric,
      n_threads, true, false, cpu_hnsw_.get()
    );
    landmark_index_available_ = cpu_hnsw_->valid();
  }

  void build_ivf(
    const std::vector<float>& data,
    int samples,
    int dimensions,
    int neighbors,
    DistanceMetric metric,
    const KODAMAMatrixOptions& options,
    int lanes,
    int* used_nlist,
    int* used_nprobe,
    double* pilot_recall
  ) {
    backend_ = Backend::CPU;
    landmark_index_available_ = false;
    cpu_hnsw_.reset();
    cpu_graph_ = {};
    const std::vector<float>* graph_data = &data;
    std::vector<float> normalized_data;
    if (metric == DistanceMetric::Cosine) {
      normalized_data = data;
      normalize_rows_for_cosine(
        normalized_data,
        static_cast<std::size_t>(samples),
        static_cast<std::size_t>(dimensions)
      );
      graph_data = &normalized_data;
    }
#if defined(KODAMA_ENABLE_CUDA)
    cuda_.reset();
    if (options.backend == Backend::CUDA) {
      detail::NativeCudaIVFStats stats;
      cuda_ = std::make_unique<detail::CudaResidentKODAMAGraph>(
        detail::make_cuda_resident_kodama_graph_ivf(
          *graph_data,
          samples,
          dimensions,
          neighbors,
          metric,
          options.knn.ivf_nlist,
          options.knn.ivf_nprobe,
          options.knn.gpu_device,
          lanes,
          &stats
        )
      );
      if (used_nlist != nullptr) *used_nlist = stats.nlist;
      if (used_nprobe != nullptr) *used_nprobe = stats.nprobe;
      if (pilot_recall != nullptr) *pilot_recall = stats.pilot_recall;
      backend_ = Backend::CUDA;
      landmark_index_available_ = true;
      return;
    }
#else
    (void)options;
#endif
#if defined(KODAMA_ENABLE_METAL)
    metal_.reset();
    if (options.backend == Backend::Metal) {
      detail::MetalIVFStats stats;
      metal_ = std::make_unique<detail::NativeMetalKODAMAGraph>(
        detail::metal_build_resident_kodama_graph_ivf(
          *graph_data,
          samples,
          dimensions,
          neighbors,
          metric,
          options.knn.ivf_nlist,
          options.knn.ivf_nprobe,
          lanes,
          &stats
        )
      );
      if (used_nlist != nullptr) *used_nlist = stats.nlist;
      if (used_nprobe != nullptr) *used_nprobe = stats.nprobe;
      if (pilot_recall != nullptr) *pilot_recall = stats.pilot_recall;
      backend_ = Backend::Metal;
      landmark_index_available_ = true;
      return;
    }
#endif
    (void)graph_data;
    (void)samples;
    (void)dimensions;
    (void)neighbors;
    (void)metric;
    (void)lanes;
    (void)used_nlist;
    (void)used_nprobe;
    (void)pilot_recall;
  }

  void build(
    const NeighborGraph& graph,
    int samples,
    Backend backend,
    int gpu_device,
    int lanes
  ) {
    (void)backend;
    backend_ = Backend::CPU;
    landmark_index_available_ = false;
    cpu_hnsw_.reset();
    cpu_graph_ = {};
#if defined(KODAMA_ENABLE_CUDA)
    cuda_.reset();
    if (backend == Backend::CUDA) {
      cuda_ = std::make_unique<detail::CudaResidentKODAMAGraph>(
        detail::make_cuda_resident_kodama_graph(
          graph,
          samples,
          gpu_device,
          lanes
        )
      );
      backend_ = Backend::CUDA;
      return;
    }
#else
    (void)gpu_device;
#endif
#if defined(KODAMA_ENABLE_METAL)
    metal_.reset();
    if (backend == Backend::Metal) {
      metal_ = std::make_unique<detail::NativeMetalKODAMAGraph>(
        detail::metal_build_resident_kodama_graph(
          graph,
          samples,
          lanes
        )
      );
      backend_ = Backend::Metal;
      return;
    }
#endif
    (void)graph;
    (void)samples;
    (void)lanes;
  }

  bool valid() const noexcept {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA) return cuda_ && cuda_->valid();
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal) return metal_ && metal_->valid();
#endif
    return false;
  }

  void prepare_results(int runs) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_prepare_results(*cuda_, runs);
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_prepare_resident_results(*metal_, runs);
      return;
    }
#endif
    throw std::runtime_error("Device-resident KODAMA results are unavailable.");
  }

  void project_to_result(
    const std::vector<int>& landmark_rows,
    const std::vector<int>& landmark_labels,
    int projection_k,
    int fallback_label,
    int run,
    int lane
  ) {
    (void)landmark_rows;
    (void)landmark_labels;
    (void)projection_k;
    (void)fallback_label;
    (void)run;
    (void)lane;
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_project_landmark_labels_to_result(
        *cuda_,
        landmark_rows,
        landmark_labels,
        projection_k,
        fallback_label,
        run,
        lane
      );
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_project_landmark_labels_to_result(
        *metal_,
        landmark_rows,
        landmark_labels,
        projection_k,
        fallback_label,
        run,
        lane
      );
      return;
    }
#endif
    throw std::runtime_error(
      "Device-resident KODAMA graph projection is unavailable."
    );
  }

  void store_result_row(const std::vector<int>& labels, int run, int lane) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_store_result_row(*cuda_, labels, run, lane);
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_store_resident_result_row(*metal_, labels, run, lane);
      return;
    }
#endif
    throw std::runtime_error("Device-resident KODAMA result upload is unavailable.");
  }

  void constrain_result_row(
    const std::vector<int>& constrain,
    int max_label,
    int run,
    int lane
  ) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_constrain_result_row(
        *cuda_, constrain, max_label, run, lane);
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_constrain_resident_result_row(
        *metal_, constrain, max_label, run, lane);
      return;
    }
#endif
    throw std::runtime_error("Device-resident constrained majority is unavailable.");
  }

  std::vector<int> download_result_row(int run, int lane) const {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      return detail::cuda_resident_download_result_row(*cuda_, run, lane);
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      return detail::metal_download_resident_result_row(*metal_, run, lane);
    }
#endif
    throw std::runtime_error("Device-resident KODAMA result download is unavailable.");
  }

  std::vector<int> download_results(int runs) const {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      return detail::cuda_resident_download_results(*cuda_, runs);
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      return detail::metal_download_resident_results(*metal_, runs);
    }
#endif
    throw std::runtime_error("Device-resident KODAMA results are unavailable.");
  }

  void replace_graph(const NeighborGraph& graph) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_replace_graph(*cuda_, graph);
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_replace_resident_kodama_graph(*metal_, graph);
      return;
    }
#endif
    throw std::runtime_error("Device-resident graph replacement is unavailable.");
  }

  void apply_dissimilarity(
    int runs,
    bool input_one_based,
    bool output_one_based
  ) {
    (void)runs;
    (void)input_one_based;
    (void)output_one_based;
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_apply_kodama_dissimilarity(
        *cuda_,
        runs,
        input_one_based,
        output_one_based
      );
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_apply_resident_kodama_dissimilarity(
        *metal_,
        runs,
        input_one_based,
        output_one_based
      );
      return;
    }
#endif
    throw std::runtime_error(
      "Device-resident KODAMA graph dissimilarity is unavailable."
    );
  }

  NeighborGraph download() const {
    if (backend_ == Backend::CPU && !cpu_graph_.indices.empty()) return cpu_graph_;
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      return detail::download_cuda_resident_kodama_graph(*cuda_);
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      return detail::metal_download_resident_kodama_graph(*metal_);
    }
#endif
    throw std::runtime_error(
      "Device-resident KODAMA graph download is unavailable."
    );
  }

  void reset_graph() {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend_ == Backend::CUDA && cuda_) {
      detail::cuda_resident_reset_graph(*cuda_);
      return;
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (backend_ == Backend::Metal && metal_) {
      detail::metal_reset_resident_kodama_graph(*metal_);
      return;
    }
#endif
  }

 private:
  Backend backend_ = Backend::CPU;
  bool landmark_index_available_ = false;
  std::unique_ptr<detail::NativeHNSWIndex> cpu_hnsw_;
  NeighborGraph cpu_graph_;
#if defined(KODAMA_ENABLE_CUDA)
  std::unique_ptr<detail::CudaResidentKODAMAGraph> cuda_;
#endif
#if defined(KODAMA_ENABLE_METAL)
  std::unique_ptr<detail::NativeMetalKODAMAGraph> metal_;
#endif
};

}  // namespace

struct KODAMAGraphHandle::Impl {
  Backend backend = Backend::CPU;
  int samples = 0;
  int neighbors = 0;
  std::shared_ptr<DeviceResidentKODAMAGraph> resident;
  mutable NeighborGraph host_graph;
  mutable std::mutex mutex;
};

namespace detail {

struct KODAMAGraphHandleAccess {
  static std::shared_ptr<KODAMAGraphHandle> create(
    Backend backend,
    int samples,
    int neighbors,
    std::shared_ptr<DeviceResidentKODAMAGraph> resident,
    NeighborGraph host_graph
  ) {
    auto impl = std::make_shared<KODAMAGraphHandle::Impl>();
    impl->backend = backend;
    impl->samples = samples;
    impl->neighbors = neighbors;
    impl->resident = std::move(resident);
    impl->host_graph = std::move(host_graph);
    return std::shared_ptr<KODAMAGraphHandle>(new KODAMAGraphHandle(std::move(impl)));
  }

  static std::shared_ptr<DeviceResidentKODAMAGraph> resident(
    const std::shared_ptr<KODAMAGraphHandle>& handle
  ) {
    return handle && handle->impl_ ? handle->impl_->resident : nullptr;
  }

  static const NeighborGraph* host_graph(
    const std::shared_ptr<KODAMAGraphHandle>& handle
  ) {
    return handle && handle->impl_ && !handle->impl_->host_graph.indices.empty() ?
      &handle->impl_->host_graph : nullptr;
  }

  static std::unique_lock<std::mutex> lock(
    const std::shared_ptr<KODAMAGraphHandle>& handle
  ) {
    return handle && handle->impl_ ?
      std::unique_lock<std::mutex>(handle->impl_->mutex) :
      std::unique_lock<std::mutex>();
  }
};

}  // namespace detail

KODAMAGraphHandle::KODAMAGraphHandle() = default;
KODAMAGraphHandle::~KODAMAGraphHandle() = default;
KODAMAGraphHandle::KODAMAGraphHandle(const KODAMAGraphHandle&) noexcept = default;
KODAMAGraphHandle& KODAMAGraphHandle::operator=(const KODAMAGraphHandle&) noexcept = default;
KODAMAGraphHandle::KODAMAGraphHandle(KODAMAGraphHandle&&) noexcept = default;
KODAMAGraphHandle& KODAMAGraphHandle::operator=(KODAMAGraphHandle&&) noexcept = default;
KODAMAGraphHandle::KODAMAGraphHandle(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
bool KODAMAGraphHandle::valid() const noexcept { return impl_ != nullptr; }
Backend KODAMAGraphHandle::backend() const noexcept {
  return impl_ ? impl_->backend : Backend::Auto;
}
int KODAMAGraphHandle::samples() const noexcept { return impl_ ? impl_->samples : 0; }
int KODAMAGraphHandle::neighbors() const noexcept { return impl_ ? impl_->neighbors : 0; }
bool KODAMAGraphHandle::host_materialized() const noexcept {
  return impl_ && !impl_->host_graph.indices.empty();
}

namespace {

struct DatasetExecutionContext {
  std::vector<float> matrix;
  const VisualizationInitResult* pca_initialization = nullptr;
  IndexedStrata shared_landmark_strata;
  int shared_landmark_strata_count = 0;
  bool has_shared_landmark_strata = false;
  std::shared_ptr<DeviceResidentKODAMAGraph> graph =
    std::make_shared<DeviceResidentKODAMAGraph>();
#if defined(KODAMA_ENABLE_CUDA)
  std::unique_ptr<detail::NativeCudaKMeansContext> cuda_kmeans;
#endif
#if defined(KODAMA_ENABLE_METAL)
  std::unique_ptr<detail::NativeMetalKMeansContext> metal_kmeans;
#endif

  std::uint64_t kmeans_input_uploads() const noexcept {
#if defined(KODAMA_ENABLE_CUDA)
    if (cuda_kmeans) return cuda_kmeans->input_uploads();
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (metal_kmeans) return metal_kmeans->input_uploads();
#endif
    return 0;
  }
};

struct GpuWorkerPlan {
  int workers = 1;
  bool automatic = false;
  int sm_count = 0;
  double free_memory_mb = 0.0;
  double total_memory_mb = 0.0;
  double worker_memory_estimate_mb = 0.0;
};

#ifdef KODAMA_ENABLE_CUDA
double estimate_gpu_worker_bytes(
  const KODAMAMatrixOptions& options,
  const int samples,
  const int features,
  const int landmarks,
  const int graph_neighbors
) {
  const double x_bytes = static_cast<double>(samples) * static_cast<double>(features) * sizeof(float);
  const double landmark_bytes = static_cast<double>(landmarks) * static_cast<double>(features) * sizeof(float);
  const double graph_bytes = static_cast<double>(samples) * static_cast<double>(graph_neighbors) *
    static_cast<double>(sizeof(int) + sizeof(float));
  const double component_bytes = static_cast<double>(landmarks) *
    static_cast<double>(std::max(1, options.components)) * sizeof(float);
  double worker_bytes = 512.0 * 1024.0 * 1024.0;
  worker_bytes += 1.5 * x_bytes;
  worker_bytes += 6.0 * landmark_bytes;
  worker_bytes += 0.25 * graph_bytes;
  if (options.classifier != CoreClassifier::KNN) {
    const double feature_component_ratio =
      static_cast<double>(features) / static_cast<double>(std::max(1, options.components));
    worker_bytes += 4.0 * feature_component_ratio * landmark_bytes;
    worker_bytes += 10.0 * component_bytes;
  } else {
    worker_bytes += static_cast<double>(landmarks) * static_cast<double>(std::max(1, options.knn.k)) *
      static_cast<double>(sizeof(int) + sizeof(float));
  }
  return worker_bytes;
}
#endif

double estimate_metal_worker_bytes(
  const KODAMAMatrixOptions& options,
  const int samples,
  const int features,
  const int landmarks,
  const int graph_neighbors
) {
  const double x_bytes = static_cast<double>(samples) * static_cast<double>(features) * sizeof(float);
  const double landmark_bytes = static_cast<double>(landmarks) * static_cast<double>(features) * sizeof(float);
  const double graph_bytes = static_cast<double>(samples) * static_cast<double>(graph_neighbors) *
    static_cast<double>(sizeof(int) + sizeof(float));
  const double component_bytes = static_cast<double>(landmarks) *
    static_cast<double>(std::max(1, options.components)) * sizeof(float);

  // Metal SIMPLS streams X through paired MPS matrix-vector products.
  double worker_bytes = 128.0 * 1024.0 * 1024.0;
  worker_bytes += 2.0 * x_bytes;
  worker_bytes += 0.25 * graph_bytes;
  if (options.classifier != CoreClassifier::KNN) {
    worker_bytes += static_cast<double>(std::max(1, options.pls.cv.folds)) * landmark_bytes;
    worker_bytes += 12.0 * component_bytes;
    worker_bytes += static_cast<double>(features) *
      static_cast<double>(std::max(1, options.components)) * 8.0;
  } else {
    worker_bytes += 6.0 * landmark_bytes;
    worker_bytes += static_cast<double>(landmarks) * static_cast<double>(std::max(1, options.knn.k)) *
      static_cast<double>(sizeof(int) + sizeof(float));
  }
  return worker_bytes;
}

#ifdef KODAMA_ENABLE_CUDA
GpuWorkerPlan resolve_gpu_worker_plan(
  const KODAMAMatrixOptions& options,
  int samples,
  int features,
  int landmarks,
  int graph_neighbors
) {
  GpuWorkerPlan plan;
  plan.workers = std::max(1, std::min(options.n_threads, options.runs));
  if (options.backend != Backend::CUDA || options.n_threads > 0) return plan;

  plan.automatic = true;
  plan.workers = 1;
  cudaSetDevice(options.knn.gpu_device);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, options.knn.gpu_device) == cudaSuccess) {
    plan.sm_count = prop.multiProcessorCount;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    plan.free_memory_mb = static_cast<double>(free_bytes) / (1024.0 * 1024.0);
    plan.total_memory_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  }

  const double worker_bytes = estimate_gpu_worker_bytes(
    options, samples, features, landmarks, graph_neighbors
  );
  plan.worker_memory_estimate_mb = worker_bytes / (1024.0 * 1024.0);

  int sm_cap = 2;
  if (plan.sm_count >= 132) sm_cap = 6;
  else if (plan.sm_count >= 96) sm_cap = 5;
  else if (plan.sm_count >= 72) sm_cap = 4;
  else if (plan.sm_count >= 32) sm_cap = 4;

  int memory_cap = 1;
  if (free_bytes > 0 && worker_bytes > 0.0) {
    memory_cap = static_cast<int>(std::floor((0.70 * static_cast<double>(free_bytes)) / worker_bytes));
    memory_cap = std::max(1, memory_cap);
  }

  plan.workers = std::max(1, std::min({options.runs, sm_cap, memory_cap}));
  return plan;
}
#else
GpuWorkerPlan resolve_gpu_worker_plan(
  const KODAMAMatrixOptions& options,
  int,
  int,
  int,
  int
) {
  GpuWorkerPlan plan;
  plan.workers = std::max(1, std::min(options.n_threads, options.runs));
  return plan;
}
#endif

#ifdef KODAMA_ENABLE_CUDA
class CudaMScheduler {
 public:
  CudaMScheduler(int lanes, int device) :
    lanes_(std::max(1, lanes)),
    device_(device) {}

  int lanes() const { return lanes_; }

  template <typename Runner>
  void run(int runs, Runner&& runner) const {
    std::atomic<int> next_run{1};
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(lanes_));
    for (int lane = 0; lane < lanes_; ++lane) {
      futures.emplace_back(std::async(std::launch::async, [&, lane]() {
        cudaSetDevice(device_);
        cudaFree(nullptr);
        IterationScratch scratch;
        while (true) {
          const int run_id = next_run.fetch_add(1);
          if (run_id > runs) break;
          runner(run_id, lane, scratch);
        }
      }));
    }
    for (auto& future : futures) future.get();
  }

 private:
  int lanes_ = 1;
  int device_ = 0;
};
#endif

void apply_kodama_dissimilarity(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int n_threads,
  bool input_one_based,
  bool output_one_based
) {
  if (runs <= 0 || samples <= 0 || graph.neighbors <= 0) return;
  OmpThreadScope threads(n_threads);
  std::vector<int> labels_by_sample(
    static_cast<std::size_t>(samples) * static_cast<std::size_t>(runs)
  );
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int sample = 0; sample < samples; ++sample) {
    int* destination = labels_by_sample.data() +
      static_cast<std::size_t>(sample) * static_cast<std::size_t>(runs);
    for (int run = 0; run < runs; ++run) {
      destination[run] = res[
        static_cast<std::size_t>(run) * static_cast<std::size_t>(samples) +
        static_cast<std::size_t>(sample)
      ];
    }
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<std::pair<float, int>> row(
      static_cast<std::size_t>(graph.neighbors)
    );
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < samples; ++i) {
      const std::size_t row_offset = static_cast<std::size_t>(i) *
                                     static_cast<std::size_t>(graph.neighbors);
      for (int j = 0; j < graph.neighbors; ++j) {
        const std::size_t offset = row_offset + static_cast<std::size_t>(j);
        const int stored_neighbor = graph.indices[offset];
        const int neighbor = stored_neighbor >= 0 && input_one_based
                                 ? stored_neighbor - 1
                                 : stored_neighbor;
        float distance = graph.distances[offset];
        if (neighbor < 0 || neighbor >= samples || !std::isfinite(distance)) {
          row[static_cast<std::size_t>(j)] = {
              std::numeric_limits<float>::infinity(), stored_neighbor};
          continue;
        }
        int same = 0;
        int valid = 0;
        const int *lhs_labels =
            labels_by_sample.data() +
            static_cast<std::size_t>(i) * static_cast<std::size_t>(runs);
        const int *rhs_labels =
            labels_by_sample.data() +
            static_cast<std::size_t>(neighbor) * static_cast<std::size_t>(runs);
        for (int run = 0; run < runs; ++run) {
          const int lhs = lhs_labels[run];
          const int rhs = rhs_labels[run];
          const int pair_valid = lhs != 0 && rhs != 0;
          valid += pair_valid;
          same += pair_valid && lhs == rhs;
        }
        if (same == 0 || valid == 0) {
          distance = std::numeric_limits<float>::infinity();
        } else {
          const double agreement =
              static_cast<double>(same) / static_cast<double>(valid);
          distance = static_cast<float>((1.0 + static_cast<double>(distance)) /
                                        (agreement * agreement));
        }
        row[static_cast<std::size_t>(j)] = {
            distance, output_one_based ? neighbor + 1 : neighbor};
      }
      std::stable_sort(row.begin(), row.end(),
                       [](const auto &a, const auto &b) {
                         if (a.first != b.first)
                           return a.first < b.first;
                         return a.second < b.second;
                       });
      for (int j = 0; j < graph.neighbors; ++j) {
        const std::size_t offset = row_offset + static_cast<std::size_t>(j);
        graph.distances[offset] = row[static_cast<std::size_t>(j)].first;
        graph.indices[offset] = row[static_cast<std::size_t>(j)].second;
      }
    }
  }
}

void make_graph_indices_one_based(NeighborGraph& graph) {
  if (graph.index_base == GraphIndexBase::One) return;
  for (int& index : graph.indices) {
    if (index >= 0) ++index;
  }
  graph.index_base = GraphIndexBase::One;
}

KODAMAGraphResult build_kodama_graph(
  const std::vector<float>& data,
  int samples,
  int dimensions,
  const KODAMAGraphOptions& options,
  bool compute_visual_init,
  bool output_one_based
) {
  if (samples < 2 || dimensions < 1) {
    throw std::invalid_argument("KODAMAGraph requires at least two rows and one column.");
  }

  detail::Timer timer;
  KODAMAGraphResult result;
  result.samples = samples;
  result.dimensions = dimensions;
  result.backend = options.backend;
  result.index_type = KNNIndexType::NativeHNSW;

  const int neighbors = std::max(1, std::min(options.neighbors, samples - 1));
  detail::Timer graph_timer;
  result.knn = self_knn_graph(
    data,
    samples,
    dimensions,
    neighbors + 1,
    options.metric,
    std::max(1, options.n_threads),
    options.backend,
    options.gpu_device,
    true,
    options.index_type,
    options.ivf_nlist,
    options.ivf_nprobe,
    &result.ivf_nlist,
    &result.ivf_nprobe,
    &result.ivf_pilot_recall,
    &result.index_type
  );
  trim_self_neighbors_in_place(result.knn, samples, neighbors);
  if (output_one_based) make_graph_indices_one_based(result.knn);
  result.graph_seconds = graph_timer.seconds();
  result.graph_builds = 1;

  if (compute_visual_init) {
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = std::max(1, options.n_threads);
    init_options.seed = options.seed;
    init_options.gpu_device = options.gpu_device;
    init_options.backend = options.backend;
    const MatrixView view{
      data.data(),
      static_cast<std::size_t>(samples),
      static_cast<std::size_t>(dimensions)
    };
    result.visual_init = KODAMAVisualizationPCAInit(view, init_options);
    result.visual_init_seconds = result.visual_init.runtime_seconds;
  }

  result.graph_storage_bytes =
    static_cast<std::uint64_t>(result.knn.indices.capacity()) * sizeof(int) +
    static_cast<std::uint64_t>(result.knn.distances.capacity()) * sizeof(float);
  result.runtime_seconds = timer.seconds();
  return result;
}

KODAMAGraphOptions graph_options_from_matrix(
  const KODAMAMatrixOptions& options,
  int neighbors
) {
  KODAMAGraphOptions graph_options;
  graph_options.neighbors = neighbors;
  graph_options.n_threads = options.n_threads;
  graph_options.seed = options.seed;
  graph_options.metric = options.metric;
  graph_options.backend = options.backend;
  graph_options.index_type = options.knn.index_type;
  graph_options.ivf_nlist = options.knn.ivf_nlist;
  graph_options.ivf_nprobe = options.knn.ivf_nprobe;
  graph_options.gpu_device = options.knn.gpu_device;
  return graph_options;
}

IterationResult run_iteration(
  MatrixView full,
  const std::vector<float>& full_float,
  const std::vector<float>& spatial,
  const std::vector<float>& spatial_jitter,
  const NeighborGraph& global_graph,
  DeviceResidentKODAMAGraph* resident_graph,
  DatasetExecutionContext* execution_context,
  int worker_lane,
  bool global_graph_is_input,
  const std::vector<int>& constrain,
  bool constrain_is_identity,
  const std::vector<int>& fixed,
  const std::vector<int>& starting_labels,
  const KODAMAMatrixOptions& options,
  int run_id,
  IterationScratch& scratch
) {
  const int n = static_cast<int>(full.rows);
  const int p = static_cast<int>(full.cols);
  int landmarks = options.landmarks;
  if (n <= landmarks) {
    landmarks = static_cast<int>(std::ceil(static_cast<double>(n) * 0.75));
  }
  landmarks = std::max(2, std::min(landmarks, n - 1));
  const int splitting = options.splitting > 0 ? options.splitting : (n < 40000 ? 100 : 300);
  const bool spatial_flag = !spatial.empty() && options.spatial_cols > 0;
  std::mt19937_64 rng(options.seed + static_cast<std::uint64_t>(run_id));

  detail::Timer iter_timer;
  const int kmeans_gpu_device = options.knn.gpu_device;
  LandmarkSample landmark_sample;
  double coarse_partition_seconds = 0.0;
  double landmark_sampling_seconds = 0.0;
  scratch.coarse_labels.clear();
  if (spatial_flag) {
    if (options.progress) {
      progress_stream() << "[kodama] M " << run_id << "/" << options.runs
                << " spatial grid landmark selection for " << landmarks << " samples" << std::endl;
    }
    detail::Timer sampling_timer;
    landmark_sample = spatial_grid_landmarks(
      spatial,
      n,
      options.spatial_cols,
      landmarks,
      rng,
      options.samples
    );
    landmark_sampling_seconds = sampling_timer.seconds();
  } else {
    const int coarse_k = std::max(2, std::min(splitting, n));
    if (execution_context != nullptr && execution_context->has_shared_landmark_strata) {
      if (options.progress) {
        progress_stream() << "[kodama] M " << run_id << "/" << options.runs
                  << " sampling from shared landmark partition with "
                  << execution_context->shared_landmark_strata_count << " strata"
                  << std::endl;
      }
      // Keep quota sampling on the same per-run random substream whether the
      // partition is computed now or supplied by the shared atlas.
      (void)rng();
      detail::Timer sampling_timer;
      landmark_sample = quota_sample_landmarks(
        execution_context->shared_landmark_strata.offsets,
        execution_context->shared_landmark_strata.rows,
        landmarks,
        rng,
        false
      );
      landmark_sampling_seconds = sampling_timer.seconds();
    } else {
      if (options.progress) {
        progress_stream() << "[kodama] M " << run_id << "/" << options.runs
                  << " coarse landmark partition with " << coarse_k << " centers" << std::endl;
      }
      detail::Timer coarse_timer;
      scratch.coarse_labels = kmeans_labels(
        full_float,
        n,
        p,
        coarse_k,
        rng,
        10,
        options.n_threads,
        options.backend,
        kmeans_gpu_device,
        worker_lane,
#if defined(KODAMA_ENABLE_CUDA)
        execution_context == nullptr ? nullptr : execution_context->cuda_kmeans.get(),
#else
        nullptr,
#endif
#if defined(KODAMA_ENABLE_METAL)
        execution_context == nullptr ? nullptr : execution_context->metal_kmeans.get()
#else
        nullptr
#endif
      );
      coarse_partition_seconds = coarse_timer.seconds();
      const IndexedStrata indexed = index_strata(scratch.coarse_labels, coarse_k);
      detail::Timer sampling_timer;
      landmark_sample = quota_sample_landmarks(
        indexed.offsets,
        indexed.rows,
        landmarks,
        rng,
        false
      );
      landmark_sampling_seconds = sampling_timer.seconds();
    }
    scratch.coarse_labels = kmeans_labels(
      full_float,
      n,
      p,
      coarse_k,
      rng,
      10,
      options.n_threads,
      options.backend,
      kmeans_gpu_device,
      worker_lane,
#if defined(KODAMA_ENABLE_CUDA)
      execution_context == nullptr ? nullptr : execution_context->cuda_kmeans.get(),
#else
      nullptr,
#endif
#if defined(KODAMA_ENABLE_METAL)
      execution_context == nullptr ? nullptr : execution_context->metal_kmeans.get()
#else
      nullptr
#endif
    );
    const IndexedStrata indexed = index_strata(scratch.coarse_labels, coarse_k);
    landmark_sample = quota_sample_landmarks(
      indexed.offsets,
      indexed.rows,
      landmarks,
      rng,
      false
    );
  }
  scratch.landpoints = std::move(landmark_sample.rows);
  const double landmark_seconds = iter_timer.seconds();
  if (options.progress) {
    progress_stream() << "[kodama] M " << run_id << "/" << options.runs
              << " selected " << scratch.landpoints.size() << " landmarks from "
              << landmark_sample.represented_strata << "/" << landmark_sample.occupied_strata
              << (spatial_flag ? " occupied grid cells" : " coarse classes");
    if (landmark_sample.grid_bins > 0) {
      progress_stream() << " using " << landmark_sample.grid_bins << " bins per axis";
    }
    progress_stream() << " in " << landmark_seconds << "s" << std::endl;
  }
  detail::Timer constraint_timer;
  const std::vector<int>* run_constrain_ptr = &constrain;
  bool run_constrain_is_identity = constrain_is_identity;
  if (spatial_flag) {
    const int spatial_dims = options.spatial_cols;
    const int nspatialclusters = std::max(1, static_cast<int>(std::llround(static_cast<double>(landmarks) * options.spatial_resolution)));
    const std::vector<float>* spatial_for_clustering = nullptr;
    if (options.spatial_coordinate_mode == SpatialCoordinateMode::Population) {
      regularize_population_coordinates(
        spatial, n, spatial_dims, rng, scratch.spatial_regularized
      );
      spatial_for_clustering = &scratch.spatial_regularized;
    } else {
      scratch.spatial_jittered.resize(spatial.size());
      for (int i = 0; i < n; ++i) {
        for (int d = 0; d < spatial_dims; ++d) {
          const float width = spatial_jitter[static_cast<std::size_t>(d)];
          std::uniform_real_distribution<float> jitter(-width, width);
          scratch.spatial_jittered[static_cast<std::size_t>(i) * spatial_dims + static_cast<std::size_t>(d)] =
            spatial[static_cast<std::size_t>(i) * spatial_dims + static_cast<std::size_t>(d)] + jitter(rng);
        }
      }
      spatial_for_clustering = &scratch.spatial_jittered;
    }
    const bool use_graph_spatial_constraints = options.spatial_constraint_mode == 1;
    if (use_graph_spatial_constraints) {
      scratch.spatial_clusters = spatial_graph_components(
        *spatial_for_clustering,
        n,
        spatial_dims,
        nspatialclusters,
        options.n_threads,
        options.backend,
        kmeans_gpu_device
      );
    } else {
      scratch.spatial_clusters = kmeans_labels(
        *spatial_for_clustering,
        n,
        spatial_dims,
        nspatialclusters,
        rng,
        10,
        options.n_threads,
        options.backend,
        kmeans_gpu_device
      );
    }
    scratch.spatial_clusters = intersect_groupings(
      scratch.spatial_clusters, options.samples);
    repair_singleton_spatial_clusters(
      scratch.spatial_clusters, spatial, n, spatial_dims, options.n_threads,
      options.samples);
    const std::vector<int> sample_constrain =
      intersect_groupings(constrain, options.samples);
    scratch.run_constrain = constrain_is_identity ?
      scratch.spatial_clusters :
      majority_by_constrain(scratch.spatial_clusters, sample_constrain);
    scratch.run_constrain = intersect_groupings(
      scratch.run_constrain, options.samples);
    run_constrain_ptr = &scratch.run_constrain;
    run_constrain_is_identity = is_identity_constrain(*run_constrain_ptr);
  }
  const double constraint_seconds = constraint_timer.seconds();
  const std::vector<int>& run_constrain = *run_constrain_ptr;
  const std::vector<int>& landpoints = scratch.landpoints;

  detail::Timer prepare_timer;
  scratch.is_landmark.assign(static_cast<std::size_t>(n), 0);
  for (int row : landpoints) scratch.is_landmark[static_cast<std::size_t>(row)] = 1;
  scratch.tpoints.clear();
  scratch.tpoints.reserve(static_cast<std::size_t>(n) - landpoints.size());
  for (int i = 0; i < n; ++i) {
    if (!scratch.is_landmark[static_cast<std::size_t>(i)]) scratch.tpoints.push_back(i);
  }
  const std::vector<int>& tpoints = scratch.tpoints;

  copy_float32_rows_into(full_float, full.cols, landpoints, scratch.x_land);
  if (run_constrain_is_identity) {
    scratch.x_constrain.resize(landpoints.size());
    std::iota(scratch.x_constrain.begin(), scratch.x_constrain.end(), 1);
  } else {
    scratch.x_constrain = factor_subset(run_constrain, landpoints);
  }
  scratch.x_fixed.resize(landpoints.size());
  for (std::size_t i = 0; i < landpoints.size(); ++i) scratch.x_fixed[i] = fixed[static_cast<std::size_t>(landpoints[i])];
  const double landmark_prepare_seconds = prepare_timer.seconds();

  detail::Timer initialization_timer;
  scratch.xw.assign(landpoints.size(), 0);
  if (!starting_labels.empty()) {
    scratch.tmp_labels.resize(landpoints.size());
    for (std::size_t i = 0; i < landpoints.size(); ++i) {
      scratch.tmp_labels[i] = starting_labels[static_cast<std::size_t>(landpoints[i])];
    }
    scratch.xw = run_constrain_is_identity ? scratch.tmp_labels : constrained_majority(scratch.tmp_labels, scratch.x_constrain);
  } else {
    const int init_k = std::max(2, std::min(splitting, static_cast<int>(landpoints.size())));
    scratch.init = kmeans_labels(
      scratch.x_land,
      static_cast<int>(landpoints.size()),
      p,
      init_k,
      rng,
      10,
      options.n_threads,
      options.backend,
      kmeans_gpu_device
    );
    scratch.xw = run_constrain_is_identity ? scratch.init : constrained_majority(scratch.init, scratch.x_constrain);
  }
  const double landmark_initialization_seconds = initialization_timer.seconds();

  MatrixView x_view{scratch.x_land.data(), landpoints.size(), full.cols};
  CoreOptions core;
  core.cycles = options.cycles;
  core.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.classifier = options.classifier;
  core.evolution = options.evolution;
  core.evolutionary_search = starting_labels.empty();
  core.guarded_diversity = true;
  core.auto_class_coarsening = options.classifier == CoreClassifier::PLS_LDA;
  core.many_to_one_absorption = true;
  core.knn = options.knn;
  core.knn.backend = options.backend;
  core.knn.metric = options.metric;
  core.knn.cv.stratified = false;
  core.knn.cv.folds = options.folds;
  core.knn.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.knn.k = std::max(core.knn.k, 1);
  core.knn.hnsw_tune_k = 50;
  core.knn.hnsw_target_recall = 0.99;
  core.knn.n_threads = options.backend == Backend::CPU ?
    options.n_threads : std::max(1, options.knn.n_threads);
  core.pls = options.pls;
  core.pls.backend = options.backend;
  core.pls.cv.stratified = false;
  core.pls.cv.folds = options.folds;
  core.pls.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.pls.max_components = options.components;
  core.pls.fixed_components = options.components;
  core.pls.n_threads = options.backend == Backend::CPU ?
    options.n_threads : std::max(1, options.pls.n_threads);

  auto run_knn_core = [&](const std::vector<int>& labels, const CoreOptions& phase) {
    if (options.backend == Backend::CUDA) {
      return CoreKNN_CUDA(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    if (options.backend == Backend::Metal) {
      return CoreKNN_METAL(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    return CoreKNN_CPU(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
  };
  auto run_pls_core = [&](const std::vector<int>& labels, const CoreOptions& phase) {
    if (options.backend == Backend::CUDA) {
      return CorePLSLDA_CUDA(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    if (options.backend == Backend::Metal) {
      return CorePLSLDA_METAL(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    return CorePLSLDA_CPU(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
  };

  CoreResult core_result;
  double landmark_graph_seconds = 0.0;
  detail::Timer classifier_timer;
  if (options.classifier == CoreClassifier::KNN) {
    if (resident_graph != nullptr && resident_graph->has_landmark_index()) {
      scratch.local_graph = resident_graph->landmark_knn_graph(
        scratch.x_land,
        landpoints,
        core.knn.k,
        core.knn
      );
      if (options.backend == Backend::CUDA) {
        core_result = CoreKNNGraph_CUDA(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      } else if (options.backend == Backend::Metal) {
        core_result = CoreKNNGraph_METAL(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      } else {
        core_result = CoreKNNGraph_CPU(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      }
    } else if (global_graph_is_input) {
      subset_graph_to_rows(
        global_graph,
        landpoints,
        n,
        scratch.global_to_local,
        scratch.local_graph
      );
      if (options.backend == Backend::CUDA) {
        core_result = CoreKNNGraph_CUDA(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      } else if (options.backend == Backend::Metal) {
        core_result = CoreKNNGraph_METAL(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      } else {
        core_result = CoreKNNGraph_CPU(scratch.local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
      }
    } else {
      core_result = run_knn_core(scratch.xw, core);
    }
  } else if (options.classifier == CoreClassifier::PLS_LDA) {
    core_result = run_pls_core(scratch.xw, core);
  } else {
    throw std::invalid_argument("Unsupported KODAMA.matrix classifier.");
  }
  const double classifier_seconds = classifier_timer.seconds();
  const double core_evolution_seconds = std::max(0.0, classifier_seconds - landmark_graph_seconds);

  IterationResult out;
  out.landmarks_used = static_cast<int>(landpoints.size());
  out.landmark_occupied_strata = landmark_sample.occupied_strata;
  out.landmark_represented_strata = landmark_sample.represented_strata;
  out.landmark_grid_bins = landmark_sample.grid_bins;
  out.landmark_seconds = landmark_seconds;
  out.coarse_partition_seconds = coarse_partition_seconds;
  out.landmark_sampling_seconds = landmark_sampling_seconds;
  out.constraint_seconds = constraint_seconds;
  out.landmark_prepare_seconds = landmark_prepare_seconds;
  out.landmark_initialization_seconds = landmark_initialization_seconds;
  out.landmark_graph_seconds = landmark_graph_seconds;
  out.core_evolution_seconds = core_evolution_seconds;
  detail::Timer projection_timer;
  out.res.assign(static_cast<std::size_t>(n), 0);
  out.constrain = run_constrain;
  for (std::size_t i = 0; i < landpoints.size(); ++i) out.res[static_cast<std::size_t>(landpoints[i])] = core_result.clbest[i];
  if (!tpoints.empty()) {
    if (options.classifier == CoreClassifier::KNN) {
      const int projection_k = std::max(1, options.knn.k);
      if (resident_graph != nullptr && resident_graph->valid()) {
        resident_graph->project_to_result(
          landpoints,
          core_result.clbest,
          projection_k,
          core_result.clbest.front(),
          run_id - 1,
          worker_lane
        );
        out.resident_result = true;
        out.sparse_projection_upload = true;
        if (!run_constrain_is_identity) {
          const int max_label = *std::max_element(
            core_result.clbest.begin(), core_result.clbest.end());
          resident_graph->constrain_result_row(
            run_constrain, max_label, run_id - 1, worker_lane);
        }
        out.res.clear();
      } else {
        scratch.global_to_local.assign(static_cast<std::size_t>(n), -1);
        for (std::size_t i = 0; i < landpoints.size(); ++i) {
          scratch.global_to_local[static_cast<std::size_t>(landpoints[i])] = static_cast<int>(i);
        }
        scratch.projection_votes.clear();
        scratch.projection_votes.reserve(static_cast<std::size_t>(projection_k));
        for (std::size_t i = 0; i < tpoints.size(); ++i) {
          const std::size_t row_offset = static_cast<std::size_t>(tpoints[i]) * static_cast<std::size_t>(global_graph.neighbors);
          scratch.projection_votes.clear();
          for (int j = 0; j < global_graph.neighbors && static_cast<int>(scratch.projection_votes.size()) < projection_k; ++j) {
            const int global_neighbor = global_graph.indices[row_offset + static_cast<std::size_t>(j)];
            if (global_neighbor < 0 || global_neighbor >= n) continue;
            const int local = scratch.global_to_local[static_cast<std::size_t>(global_neighbor)];
            if (local >= 0) scratch.projection_votes.push_back(core_result.clbest[static_cast<std::size_t>(local)]);
          }
          out.res[static_cast<std::size_t>(tpoints[i])] =
            scratch.projection_votes.empty() ? core_result.clbest.front() : majority_label(scratch.projection_votes);
        }
      }
    } else {
      copy_float32_rows_into(full_float, full.cols, tpoints, scratch.x_test);
      MatrixView test_view{scratch.x_test.data(), tpoints.size(), full.cols};
      std::vector<int> projected;
      if (options.backend == Backend::CUDA) {
        projected = PLSLDAPredict_CUDA(x_view, core_result.clbest, test_view, core.pls);
      } else if (options.backend == Backend::Metal) {
        projected = PLSLDAPredict_METAL(x_view, core_result.clbest, test_view, core.pls);
      } else {
        projected = PLSLDAPredict_CPU(x_view, core_result.clbest, test_view, core.pls);
      }
      for (std::size_t i = 0; i < tpoints.size(); ++i) {
        out.res[static_cast<std::size_t>(tpoints[i])] = projected[i];
      }
    }
  }
  if (!run_constrain_is_identity && !out.res.empty()) {
    out.res = constrained_majority(out.res, run_constrain);
    if (out.resident_result) {
      resident_graph->store_result_row(
        out.res, run_id - 1, worker_lane);
      ++out.resident_row_uploads;
    }
  }
  if (resident_graph != nullptr && resident_graph->valid() && !out.resident_result) {
    resident_graph->store_result_row(
      out.res, run_id - 1, worker_lane);
    out.resident_result = true;
    ++out.resident_row_uploads;
  }
  out.acc = core_result.vect_acc;
  out.run_diagnostic.run = run_id;
  out.run_diagnostic.cycles_completed = core_result.cycles_completed;
  out.run_diagnostic.transition_attempted = core_result.transition_attempted;
  out.run_diagnostic.transition_accepted = core_result.transition_accepted;
  out.run_diagnostic.many_to_one_attempted = core_result.many_to_one_attempted;
  out.run_diagnostic.many_to_one_accepted = core_result.many_to_one_accepted;
  out.run_diagnostic.pls_coarsening_attempted = core_result.pls_coarsening_attempted;
  out.run_diagnostic.pls_coarsening_accepted = core_result.pls_coarsening_accepted;
  out.run_diagnostic.cv_evaluations = core_result.cv_evaluations;
  out.run_diagnostic.landmark_rows_hash = hash_integer_vector(landpoints);
  out.run_diagnostic.initial_labels_hash = hash_integer_vector(scratch.xw);
  out.run_diagnostic.fold_assignments_hash = hash_integer_vector(core_result.fold_assignments);
  out.cycle_diagnostics.resize(static_cast<std::size_t>(core_result.cycles_completed));
  for (int cycle = 0; cycle < core_result.cycles_completed; ++cycle) {
    CoreCycleDiagnostic& diagnostic = out.cycle_diagnostics[static_cast<std::size_t>(cycle)];
    diagnostic.run = run_id;
    diagnostic.cycle = cycle + 1;
    diagnostic.proposal_size = core_result.proposal_size[static_cast<std::size_t>(cycle)];
    diagnostic.active_classes = core_result.active_classes[static_cast<std::size_t>(cycle)];
    diagnostic.accepted = core_result.accepted[static_cast<std::size_t>(cycle)];
    diagnostic.improving_acceptance =
      core_result.improving_acceptance[static_cast<std::size_t>(cycle)];
    diagnostic.temperature_acceptance =
      core_result.temperature_acceptance[static_cast<std::size_t>(cycle)];
  }
  out.accbest = core_result.accbest;
  out.runtime = core_result.runtime_seconds;
  out.memory = core_result.peak_memory_mb;
  return out;
}

KODAMAMatrixResult run_kodama_matrix(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain_in,
  const std::vector<int>& fixed_in,
  KODAMAMatrixOptions options,
  const NeighborGraph* input_graph = nullptr,
  const KODAMAGraphResult* prepared_graph = nullptr
) {
  detail::validate_inputs(x, std::vector<int>(x.rows, 1), std::vector<int>());
  if (!starting_labels.empty() && starting_labels.size() != x.rows) throw std::invalid_argument("starting_labels size must match number of rows.");
  if (x.rows < 3) throw std::invalid_argument("KODAMAMatrix requires at least 3 rows.");
  if (options.runs < 1) throw std::invalid_argument("KODAMAMatrixOptions::runs must be positive.");
  if (options.cycles < 0) throw std::invalid_argument("KODAMAMatrixOptions::cycles must be non-negative.");
  if (options.folds < 2) throw std::invalid_argument("KODAMAMatrixOptions::folds must be at least two.");
  detail::Timer timer;
  if (options.landmarks <= 0) options.landmarks = 10000;
  if (static_cast<std::size_t>(options.landmarks) >= x.rows) {
    options.landmarks = static_cast<int>(std::ceil(static_cast<double>(x.rows) * 0.75));
  }
  options.landmarks = std::max(2, std::min(options.landmarks, static_cast<int>(x.rows) - 1));
  options.components = std::max(1, std::min(options.components, static_cast<int>(std::min(x.rows, x.cols))));
  const int requested_neighbors = options.graph_neighbors > 0 ? options.graph_neighbors : 100;
  GpuWorkerPlan worker_plan;
  if (options.backend == Backend::CUDA && options.n_threads <= 0) {
    worker_plan = resolve_gpu_worker_plan(
      options,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      options.landmarks,
      requested_neighbors
    );
    options.n_threads = worker_plan.workers;
    if (options.progress) {
      progress_stream() << "[kodama] CUDA auto workers selected " << worker_plan.workers
                << " (SM=" << worker_plan.sm_count
                << ", free=" << worker_plan.free_memory_mb << " MiB"
                << ", per_worker_est=" << worker_plan.worker_memory_estimate_mb << " MiB)"
                << std::endl;
    }
  } else if (options.backend == Backend::Metal && options.n_threads <= 0) {
    worker_plan.automatic = true;
    const double worker_bytes = estimate_metal_worker_bytes(
      options,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      options.landmarks,
      requested_neighbors
    );
    worker_plan.worker_memory_estimate_mb = worker_bytes / (1024.0 * 1024.0);
    worker_plan.workers = detail::metal_recommended_worker_count(
      static_cast<std::size_t>(std::ceil(worker_bytes)),
      options.runs
    );
    options.n_threads = worker_plan.workers;
    if (options.progress) {
      progress_stream() << "[kodama] Metal auto workers selected " << worker_plan.workers
                << " (per_worker_est=" << worker_plan.worker_memory_estimate_mb
                << " MiB)" << std::endl;
    }
  } else {
    options.n_threads = std::max(1, options.n_threads);
    worker_plan.workers = std::min(options.n_threads, options.runs);
  }
  if (!options.spatial.empty()) {
    if (options.spatial_cols <= 0) throw std::invalid_argument("KODAMAMatrixOptions::spatial_cols must be positive when spatial is provided.");
    if (options.spatial.size() != x.rows * static_cast<std::size_t>(options.spatial_cols)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial size must be rows * spatial_cols.");
    }
    if (options.spatial_resolution <= 0.0 || !std::isfinite(options.spatial_resolution)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial_resolution must be positive.");
    }
    options.spatial = KODAMASeparateSpatialSamples(
      options.spatial,
      static_cast<int>(x.rows),
      options.spatial_cols,
      options.samples
    );
  } else if (!options.samples.empty()) {
    throw std::invalid_argument(
      "KODAMAMatrixOptions::samples requires spatial coordinates."
    );
  } else if (options.spatial_coordinate_mode == SpatialCoordinateMode::Population) {
    throw std::invalid_argument(
      "Population spatial mode requires spatial coordinates."
    );
  }

  detail::Timer input_copy_timer;
  DatasetExecutionContext execution_context;
  execution_context.matrix = copy_float32(x);
  const std::vector<float>& full_float = execution_context.matrix;
  const std::vector<int> constrain = normalize_constrain(constrain_in, x.rows);
  const bool constrain_is_identity = is_identity_constrain(constrain);
  const std::vector<int> fixed = normalize_fixed(fixed_in, x.rows);
  const double input_copy_seconds = input_copy_timer.seconds();

  detail::Timer spatial_precompute_timer;
  const bool reuse_spatial_jitter = prepared_graph != nullptr &&
    prepared_graph->samples == static_cast<int>(x.rows) &&
    prepared_graph->spatial_dimensions == options.spatial_cols &&
    prepared_graph->spatial_jitter.size() == static_cast<std::size_t>(options.spatial_cols);
  const std::vector<float> spatial_jitter = options.spatial.empty() ?
    std::vector<float>() : reuse_spatial_jitter ?
    prepared_graph->spatial_jitter : spatial_jitter_from_graph(
      options.spatial,
      static_cast<int>(x.rows),
      options.spatial_cols,
      std::max(20, options.graph_neighbors > 0 ? options.graph_neighbors : 100),
      options.n_threads,
      options.backend,
      options.knn.gpu_device
    );
  const double spatial_precompute_seconds = spatial_precompute_timer.seconds();
  const int neighbors = std::max(1, static_cast<int>(std::floor(std::min({
    static_cast<double>(options.landmarks),
    static_cast<double>(x.rows) * 0.75 - 1.0,
    static_cast<double>(requested_neighbors)
  }))));

  KODAMAMatrixResult result;
  result.runs = options.runs;
  result.samples = static_cast<int>(x.rows);
  result.cycles = options.cycles;
  result.res_constrain_rows = options.spatial.empty() ? 1 : options.runs;
  result.effective_landmarks = options.landmarks;
  result.n_threads = options.n_threads;
  result.backend = options.backend;
  result.gpu_auto_workers = worker_plan.automatic;
  const bool accelerator_scheduler =
    options.backend == Backend::CUDA || options.backend == Backend::Metal;
  result.gpu_scheduler_enabled = accelerator_scheduler;
  result.gpu_scheduler_lanes = accelerator_scheduler ? worker_plan.workers : 0;
  result.gpu_sm_count = worker_plan.sm_count;
  result.gpu_free_memory_mb = worker_plan.free_memory_mb;
  result.gpu_total_memory_mb = worker_plan.total_memory_mb;
  result.gpu_worker_memory_estimate_mb = worker_plan.worker_memory_estimate_mb;
  result.input_copy_seconds = input_copy_seconds;
  result.spatial_precompute_seconds = spatial_precompute_seconds;
  result.spatial_graph_builds = reuse_spatial_jitter ? 0 : (options.spatial.empty() ? 0 : 1);
  result.acc.assign(static_cast<std::size_t>(options.runs), std::numeric_limits<double>::quiet_NaN());
  result.landmark_occupied_strata.assign(static_cast<std::size_t>(options.runs), 0);
  result.landmark_represented_strata.assign(static_cast<std::size_t>(options.runs), 0);
  result.landmark_grid_bins.assign(static_cast<std::size_t>(options.runs), 0);
  result.landmark_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.coarse_partition_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.landmark_sampling_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.constraint_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.landmark_prepare_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.landmark_initialization_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.landmark_graph_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.core_evolution_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.projection_seconds.assign(static_cast<std::size_t>(options.runs), 0.0);
  result.run_diagnostics.assign(static_cast<std::size_t>(options.runs), {});
  result.cycle_diagnostics.assign(
    static_cast<std::size_t>(options.runs) * static_cast<std::size_t>(options.cycles), {}
  );
  result.v.assign(static_cast<std::size_t>(options.runs) * options.cycles, std::numeric_limits<double>::quiet_NaN());
  result.res.assign(static_cast<std::size_t>(options.runs) * x.rows, 0);
  result.res_constrain.assign(static_cast<std::size_t>(result.res_constrain_rows) * x.rows, 0);
  const int workers = std::max(1, std::min(options.n_threads, options.runs));
  NeighborGraph global_graph;
  bool host_graph_trimmed = false;
  const std::shared_ptr<KODAMAGraphHandle> prepared_handle =
    prepared_graph == nullptr ? nullptr : prepared_graph->handle;
  auto prepared_handle_lock = detail::KODAMAGraphHandleAccess::lock(prepared_handle);
  std::shared_ptr<DeviceResidentKODAMAGraph> shared_resident =
    detail::KODAMAGraphHandleAccess::resident(prepared_handle);
  const NeighborGraph* effective_input_graph = input_graph;
  if (effective_input_graph == nullptr) {
    effective_input_graph = detail::KODAMAGraphHandleAccess::host_graph(prepared_handle);
  }
  const bool graph_is_input = effective_input_graph != nullptr ||
    (shared_resident != nullptr && shared_resident->valid());
  result.graph_backend = graph_is_input ? Backend::Auto : options.backend;
  result.optimization_backend = options.backend;
  result.dissimilarity_backend = Backend::CPU;
  DeviceResidentKODAMAGraph& resident_graph = shared_resident ?
    *shared_resident : *execution_context.graph;
  if (shared_resident && resident_graph.valid()) resident_graph.reset_graph();
  const bool use_resident_graph =
    (options.backend == Backend::CUDA || options.backend == Backend::Metal);
  const double graph_exact_work =
    static_cast<double>(x.rows) * static_cast<double>(x.rows) *
    static_cast<double>(x.cols);
  const bool cuda_resident_ivf = options.backend == Backend::CUDA &&
    (options.knn.index_type == KNNIndexType::CudaIVFFlat ||
     (options.knn.index_type != KNNIndexType::CudaExact &&
      x.rows > 5000 && graph_exact_work > 2.0e8));
  const bool metal_resident_ivf = options.backend == Backend::Metal &&
    (options.knn.index_type == KNNIndexType::MetalIVFFlat ||
     (options.knn.index_type != KNNIndexType::MetalExact &&
      x.rows > 5000 && graph_exact_work > 2.0e8));
  const bool direct_resident_ivf =
    !graph_is_input && (cuda_resident_ivf || metal_resident_ivf) &&
    !detail::should_use_spatial_grid_knn(
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      options.metric
    );
  const bool cpu_reusable_hnsw = !graph_is_input &&
    options.backend == Backend::CPU &&
    options.classifier == CoreClassifier::KNN &&
    options.knn.index_type == KNNIndexType::NativeHNSW;
  if (graph_is_input) {
    if (options.compute_visual_init) {
      if (options.progress) {
        progress_stream() << "[kodama] computing shared PCA initialization for UMAP and openTSNE"
                  << std::endl;
      }
      VisualizationInitOptions init_options;
      init_options.n_components = 2;
      init_options.n_threads = options.n_threads;
      init_options.seed = options.seed;
      init_options.gpu_device = options.knn.gpu_device;
      init_options.backend = options.backend;
      const MatrixView float_view{full_float.data(), x.rows, x.cols};
      result.visual_init = KODAMAVisualizationPCAInit(float_view, init_options);
      result.has_visual_init = true;
      result.visual_init_seconds = result.visual_init.runtime_seconds;
    }
    detail::Timer graph_timer;
    if (options.progress) {
      progress_stream() << "[kodama] using caller-supplied KNN graph with "
                << (effective_input_graph != nullptr ?
                      effective_input_graph->neighbors : neighbors)
                << " neighbors" << std::endl;
    }
    if (effective_input_graph != nullptr) {
      const int retained = std::min(effective_input_graph->neighbors, neighbors);
      global_graph = normalize_external_graph(
        *effective_input_graph, static_cast<int>(x.rows), retained);
    }
    result.graph_seconds = graph_timer.seconds();
    host_graph_trimmed = effective_input_graph == nullptr;
  } else if (cpu_reusable_hnsw) {
    if (options.progress) {
      progress_stream() << "[kodama] building one retained CPU HNSW index for all M runs"
                << std::endl;
    }
    detail::Timer graph_timer;
    resident_graph.build_hnsw(
      full_float, static_cast<int>(x.rows), static_cast<int>(x.cols),
      neighbors, options.metric, options.n_threads
    );
    global_graph = resident_graph.download();
    result.graph_index_type = KNNIndexType::NativeHNSW;
    result.graph_builds = 1;
    result.graph_seconds = graph_timer.seconds();
    host_graph_trimmed = true;
    if (options.compute_visual_init) {
      VisualizationInitOptions init_options;
      init_options.n_components = 2;
      init_options.n_threads = options.n_threads;
      init_options.seed = options.seed;
      init_options.gpu_device = options.knn.gpu_device;
      init_options.backend = options.backend;
      result.visual_init = KODAMAVisualizationPCAInit(x, init_options);
      result.has_visual_init = true;
      result.visual_init_seconds = result.visual_init.runtime_seconds;
    }
  } else if (direct_resident_ivf) {
    if (options.progress) {
      progress_stream() << "[kodama] building one resident accelerator IVF-Flat graph for all M runs"
                << std::endl;
    }
    detail::Timer graph_timer;
    resident_graph.build_ivf(
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      neighbors,
      options.metric,
      options,
      workers,
      &result.graph_ivf_nlist,
      &result.graph_ivf_nprobe,
      &result.graph_ivf_pilot_recall
    );
    result.graph_index_type = options.backend == Backend::CUDA ?
      KNNIndexType::CudaIVFFlat : KNNIndexType::MetalIVFFlat;
    result.graph_builds = 1;
    result.graph_seconds = graph_timer.seconds();
    host_graph_trimmed = true;
    if (options.compute_visual_init) {
      VisualizationInitOptions init_options;
      init_options.n_components = 2;
      init_options.n_threads = options.n_threads;
      init_options.seed = options.seed;
      init_options.gpu_device = options.knn.gpu_device;
      init_options.backend = options.backend;
      result.visual_init = KODAMAVisualizationPCAInit(x, init_options);
      result.has_visual_init = true;
      result.visual_init_seconds = result.visual_init.runtime_seconds;
    }
  } else {
    if (options.progress) {
      progress_stream() << "[kodama] running KODAMA.graph for " << x.rows
                << " samples, " << neighbors
                << " neighbors, and shared PCA initialization" << std::endl;
    }
    KODAMAGraphResult prepared = build_kodama_graph(
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      graph_options_from_matrix(options, neighbors),
      options.compute_visual_init,
      false
    );
    global_graph = std::move(prepared.knn);
    result.graph_builds = prepared.graph_builds;
    result.graph_seconds = prepared.graph_seconds;
    result.visual_init = std::move(prepared.visual_init);
    result.has_visual_init = options.compute_visual_init;
    result.visual_init_seconds = prepared.visual_init_seconds;
    result.graph_index_type = prepared.index_type;
    result.graph_ivf_nlist = prepared.ivf_nlist;
    result.graph_ivf_nprobe = prepared.ivf_nprobe;
    result.graph_ivf_pilot_recall = prepared.ivf_pilot_recall;
    host_graph_trimmed = true;
  }
  if (use_resident_graph && !resident_graph.valid()) {
    resident_graph.build(
      global_graph,
      static_cast<int>(x.rows),
      options.backend,
      options.knn.gpu_device,
      workers
    );
    if (!graph_is_input && resident_graph.valid() && options.classifier != CoreClassifier::KNN) {
      std::vector<int>().swap(global_graph.indices);
      std::vector<float>().swap(global_graph.distances);
    }
  }
  if (resident_graph.valid()) {
    resident_graph.prepare_results(options.runs);
  }
  execution_context.pca_initialization = result.has_visual_init ? &result.visual_init : nullptr;
  if (options.spatial.empty()) {
    const int coarse_k = std::max(
      2,
      std::min(options.splitting > 0 ? options.splitting :
        (static_cast<int>(x.rows) < 40000 ? 100 : 300), static_cast<int>(x.rows))
    );
#if defined(KODAMA_ENABLE_CUDA)
    if (options.backend == Backend::CUDA) {
      if (options.progress) {
        progress_stream() << "[kodama] uploading one resident CUDA k-means matrix for all M runs"
                  << std::endl;
      }
      execution_context.cuda_kmeans =
        std::make_unique<detail::NativeCudaKMeansContext>(
          detail::native_cuda_build_kmeans_context(
            full_float, static_cast<int>(x.rows), static_cast<int>(x.cols),
            workers, coarse_k, options.knn.gpu_device
          )
        );
    }
#endif
#if defined(KODAMA_ENABLE_METAL)
    if (options.backend == Backend::Metal) {
      if (options.progress) {
        progress_stream() << "[kodama] retaining one Metal k-means matrix for all M runs"
                  << std::endl;
      }
      execution_context.metal_kmeans =
        std::make_unique<detail::NativeMetalKMeansContext>(
          detail::metal_build_kmeans_context(
            full_float, static_cast<int>(x.rows), static_cast<int>(x.cols),
            workers, coarse_k
          )
        );
    }
#endif
    if (options.progress) {
      progress_stream() << "[kodama] building one shared k-means landmark atlas with "
                << coarse_k << " strata for all independent M runs" << std::endl;
    }
    detail::Timer shared_partition_timer;
    std::mt19937_64 shared_rng(options.seed + 1u);
    std::vector<int> shared_labels = kmeans_labels(
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      coarse_k,
      shared_rng,
      10,
      options.n_threads,
      options.backend,
      options.knn.gpu_device,
      0,
#if defined(KODAMA_ENABLE_CUDA)
      execution_context.cuda_kmeans.get(),
#else
      nullptr,
#endif
#if defined(KODAMA_ENABLE_METAL)
      execution_context.metal_kmeans.get()
#else
      nullptr
#endif
    );
    execution_context.shared_landmark_strata =
      index_strata(shared_labels, coarse_k);
    execution_context.shared_landmark_strata_count = coarse_k;
    execution_context.has_shared_landmark_strata = true;
    result.shared_landmark_partition_used = true;
    result.shared_landmark_partition_strata = coarse_k;
    result.shared_landmark_partition_seconds =
      shared_partition_timer.seconds();
  }
  result.kmeans_input_uploads = execution_context.kmeans_input_uploads();
  // A raw-data accelerator run queries the retained IVF index for each
  // landmark set. The full graph stays resident until materialization is
  // explicitly required; graph-only input remains host-backed by definition.
  detail::Timer optimization_timer;
  KODAMAMatrixOptions iteration_options = options;
  if (options.backend == Backend::CPU) {
    iteration_options.n_threads = std::max(1, options.n_threads / workers);
  } else if (options.backend == Backend::Metal && workers > 1) {
    // Independent M lanes already provide device concurrency. Letting every
    // lane also spawn one host worker per CV fold creates workers * folds
    // command queues and thread-local Metal workspaces for each proposal.
    // Traverse folds sequentially inside each lane so its resident fold cache
    // and command queue survive across Tcycles.
    iteration_options.n_threads = 1;
  }
  std::vector<double> iteration_runtime(static_cast<std::size_t>(options.runs), 0.0);
  std::vector<double> iteration_memory(static_cast<std::size_t>(options.runs), 0.0);
  std::vector<int> iteration_landmarks(static_cast<std::size_t>(options.runs), 0);
  std::vector<char> iteration_resident(static_cast<std::size_t>(options.runs), 0);
  std::atomic<std::uint64_t> projection_sparse_uploads{0};
  std::atomic<std::uint64_t> projection_full_downloads{0};
  std::atomic<std::uint64_t> result_row_uploads{0};

  auto execute_run = [&](int run_id, int worker_lane, IterationScratch& scratch) {
    if (options.progress) {
      progress_stream() << "[kodama] launch M " << run_id << "/" << options.runs << std::endl;
    }
    IterationResult iter = run_iteration(
      x,
      full_float,
      options.spatial,
      spatial_jitter,
      global_graph,
      (resident_graph.valid() || resident_graph.has_landmark_index()) ?
        &resident_graph : nullptr,
      &execution_context,
      worker_lane,
      graph_is_input,
      constrain,
      constrain_is_identity,
      fixed,
      starting_labels,
      iteration_options,
      run_id,
      scratch
    );
    if (options.progress) {
      progress_stream() << "[kodama] complete M " << run_id << "/" << options.runs
                << " acc=" << iter.accbest
                << " elapsed=" << timer.seconds() << "s" << std::endl;
    }
    const std::size_t row = static_cast<std::size_t>(run_id - 1);
    result.acc[row] = iter.accbest;
    result.run_diagnostics[row] = iter.run_diagnostic;
    for (std::size_t cycle = 0; cycle < iter.cycle_diagnostics.size(); ++cycle) {
      result.cycle_diagnostics[
        row * static_cast<std::size_t>(options.cycles) + cycle
      ] = iter.cycle_diagnostics[cycle];
    }
    result.landmark_occupied_strata[row] = iter.landmark_occupied_strata;
    result.landmark_represented_strata[row] = iter.landmark_represented_strata;
    result.landmark_grid_bins[row] = iter.landmark_grid_bins;
    result.landmark_seconds[row] = iter.landmark_seconds;
    if (!iter.res.empty()) {
      std::copy(iter.res.begin(), iter.res.end(), result.res.begin() + row * x.rows);
    }
    if (result.res_constrain_rows > 1 || row == 0) {
      const std::size_t constrain_row = result.res_constrain_rows > 1 ? row : 0;
      std::copy(
        iter.constrain.begin(),
        iter.constrain.end(),
        result.res_constrain.begin() + constrain_row * x.rows
      );
    }
    for (int c = 0; c < options.cycles && c < static_cast<int>(iter.acc.size()); ++c) {
      result.v[row * static_cast<std::size_t>(options.cycles) + static_cast<std::size_t>(c)] =
        iter.acc[static_cast<std::size_t>(c)];
    }
    iteration_runtime[row] = iter.runtime;
    iteration_memory[row] = iter.memory;
    iteration_landmarks[row] = iter.landmarks_used;
    iteration_resident[row] = iter.resident_result ? 1 : 0;
    if (iter.sparse_projection_upload) projection_sparse_uploads.fetch_add(1);
    if (iter.full_projection_download) projection_full_downloads.fetch_add(1);
    result_row_uploads.fetch_add(static_cast<std::uint64_t>(iter.resident_row_uploads));
  };

#ifdef KODAMA_ENABLE_CUDA
  if (options.backend == Backend::CUDA) {
    CudaMScheduler scheduler(workers, options.knn.gpu_device);
    if (options.progress) {
      progress_stream() << "[kodama] CUDA M scheduler using " << scheduler.lanes()
                << " independent lanes" << std::endl;
    }
    scheduler.run(options.runs, [&](int run_id, int lane, IterationScratch& scratch) {
      execute_run(run_id, lane, scratch);
    });
  } else
#endif
  {
    std::atomic<int> next_run{1};
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
      futures.emplace_back(std::async(std::launch::async, [&, worker]() {
        IterationScratch scratch;
        while (true) {
          const int run_id = next_run.fetch_add(1);
          if (run_id > options.runs) break;
          execute_run(run_id, worker, scratch);
        }
      }));
    }
    for (auto& future : futures) future.get();
  }
  result.optimization_wall_seconds = optimization_timer.seconds();

  const bool all_results_resident = resident_graph.valid() &&
    std::all_of(iteration_resident.begin(), iteration_resident.end(),
                [](char value) { return value != 0; });
  result.projection_sparse_uploads = projection_sparse_uploads.load();
  result.projection_full_downloads = projection_full_downloads.load();
  result.result_row_uploads = result_row_uploads.load();

  for (int run_id = 0; run_id < options.runs; ++run_id) {
    const std::size_t row = static_cast<std::size_t>(run_id);
    result.effective_landmarks = iteration_landmarks[row];
    result.optimization_sum_seconds += iteration_runtime[row];
    result.peak_memory_mb = std::max(result.peak_memory_mb, iteration_memory[row]);
  }

  if (!host_graph_trimmed) {
    trim_self_neighbors_in_place(
      global_graph,
      static_cast<int>(x.rows),
      std::min(neighbors, global_graph.neighbors)
    );
    if (resident_graph.valid()) {
      resident_graph.replace_graph(global_graph);
    }
  }
  if (!options.spatial.empty() && options.spatial_graph_mix) {
    if (options.progress) {
      progress_stream() << "[kodama] building spatial KNN graph for final KODAMA graph" << std::endl;
    }
    detail::Timer spatial_graph_timer;
    if (global_graph.indices.empty() && resident_graph.valid()) {
      // Spatial graph fusion currently operates on host graph storage.
      global_graph = resident_graph.download();
    }
    const bool reuse_spatial_graph = prepared_graph != nullptr &&
      prepared_graph->samples == static_cast<int>(x.rows) &&
      prepared_graph->spatial_dimensions == options.spatial_cols &&
      prepared_graph->spatial_knn.neighbors >= global_graph.neighbors;
    NeighborGraph spatial_graph;
    if (reuse_spatial_graph) {
      spatial_graph = normalize_external_graph(
        prepared_graph->spatial_knn,
        static_cast<int>(x.rows),
        global_graph.neighbors
      );
      result.spatial_graph_builds = 0;
    } else {
      spatial_graph = self_knn_graph(
        options.spatial,
        static_cast<int>(x.rows),
        options.spatial_cols,
        neighbors + 1,
        DistanceMetric::Euclidean,
        options.n_threads,
        options.backend,
        options.knn.gpu_device,
        true,
        options.knn.index_type,
        options.knn.ivf_nlist,
        options.knn.ivf_nprobe
      );
      trim_self_neighbors_in_place(
        spatial_graph,
        static_cast<int>(x.rows),
        global_graph.neighbors
      );
      ++result.spatial_graph_builds;
    }
    merge_feature_spatial_graphs_in_place(
      global_graph,
      spatial_graph,
      static_cast<int>(x.rows),
      global_graph.neighbors
    );
    if (resident_graph.valid()) {
      resident_graph.replace_graph(global_graph);
    }
    result.spatial_graph_seconds = spatial_graph_timer.seconds();
  }

  if (options.apply_kodama_dissimilarity && options.materialize_graph) {
    if (options.progress) {
      progress_stream() << "[kodama] applying KODAMA dissimilarity to KNN graph" << std::endl;
    }
    detail::Timer dissimilarity_timer;
#if defined(KODAMA_ENABLE_CUDA) || defined(KODAMA_ENABLE_METAL)
    if (resident_graph.valid()) {
      result.dissimilarity_backend = options.backend;
      resident_graph.apply_dissimilarity(
        result.runs,
        false,
        false
      );
      if (options.materialize_graph) global_graph = resident_graph.download();
    } else
#endif
#if defined(KODAMA_ENABLE_CUDA)
    if (options.backend == Backend::CUDA) {
      result.dissimilarity_backend = Backend::CUDA;
      detail::apply_kodama_dissimilarity_cuda(
        global_graph,
        result.res,
        result.runs,
        result.samples,
        options.knn.gpu_device,
        false,
        false
      );
    } else
#endif
    {
      apply_kodama_dissimilarity(
        global_graph,
        result.res,
        result.runs,
        result.samples,
        options.n_threads,
        false,
        false
      );
    }
    result.knn_is_kodama_corrected = true;
    result.dissimilarity_seconds = dissimilarity_timer.seconds();
  } else {
    if (options.materialize_graph && global_graph.indices.empty() && resident_graph.valid()) {
      global_graph = resident_graph.download();
    }
    if (options.progress && !options.apply_kodama_dissimilarity) {
      progress_stream() << "[kodama] retaining the base graph for lazy KODAMA dissimilarity"
                << std::endl;
    }
  }
  if (all_results_resident) {
    result.res = resident_graph.download_results(result.runs);
    ++result.result_matrix_downloads;
  }
  if (options.materialize_graph) {
    if (!global_graph.indices.empty()) make_graph_indices_one_based(global_graph);
    result.graph_storage_bytes =
      static_cast<std::uint64_t>(global_graph.indices.capacity()) * sizeof(int) +
      static_cast<std::uint64_t>(global_graph.distances.capacity()) * sizeof(float);
    result.knn = std::move(global_graph);
  } else {
    result.graph_storage_bytes = 0;
  }
  result.runtime_seconds = timer.seconds();
  const auto accumulated = [](const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
  };
  const double measured_wall =
    result.input_copy_seconds + result.spatial_precompute_seconds +
    result.graph_seconds + result.graph_feature_seconds +
    result.visual_init_seconds +
    result.shared_landmark_partition_seconds + result.optimization_wall_seconds +
    result.spatial_graph_seconds + result.dissimilarity_seconds;
  const double orchestration_seconds =
    std::max(0.0, result.runtime_seconds - measured_wall);
  result.timings = {
    {"input_copy", result.input_copy_seconds, result.input_copy_seconds},
    {"spatial_precompute", result.spatial_precompute_seconds,
      result.spatial_precompute_seconds},
    {"knn_graph", result.graph_seconds, result.graph_seconds},
    {"graph_features", result.graph_feature_seconds,
      result.graph_feature_seconds},
    {"visual_initialization", result.visual_init_seconds,
      result.visual_init_seconds},
    {"shared_landmark_partition", result.shared_landmark_partition_seconds,
      result.shared_landmark_partition_seconds},
    {"optimization", result.optimization_wall_seconds,
      result.optimization_sum_seconds},
    {"coarse_partition_per_run", 0.0,
      accumulated(result.coarse_partition_seconds)},
    {"landmark_sampling", 0.0,
      accumulated(result.landmark_sampling_seconds)},
    {"constraint_preparation", 0.0,
      accumulated(result.constraint_seconds)},
    {"landmark_matrix_preparation", 0.0,
      accumulated(result.landmark_prepare_seconds)},
    {"landmark_initialization", 0.0,
      accumulated(result.landmark_initialization_seconds)},
    {"landmark_graph", 0.0,
      accumulated(result.landmark_graph_seconds)},
    {"core_evolution", 0.0,
      accumulated(result.core_evolution_seconds)},
    {"label_projection", 0.0,
      accumulated(result.projection_seconds)},
    {"spatial_graph_fusion", result.spatial_graph_seconds,
      result.spatial_graph_seconds},
    {"kodama_dissimilarity", result.dissimilarity_seconds,
      result.dissimilarity_seconds},
    {"orchestration_and_materialization", orchestration_seconds,
      orchestration_seconds},
    {"total", result.runtime_seconds, result.runtime_seconds}
  };
  result.peak_memory_mb = std::max(result.peak_memory_mb, detail::peak_memory_mb());
  if (options.progress) {
    progress_stream() << "[kodama] finished KODAMA.matrix in " << result.runtime_seconds << "s" << std::endl;
  }
  return result;
}

}  // namespace

namespace {

KODAMAGraphResult run_public_kodama_graph(
  MatrixView x,
  const KODAMAGraphOptions& options,
  const MatrixView* spatial = nullptr
) {
  if (spatial == nullptr && !options.samples.empty()) {
    throw std::invalid_argument(
      "KODAMAGraphOptions::samples requires spatial coordinates."
    );
  }
  detail::validate_inputs(x, std::vector<int>(x.rows, 1), std::vector<int>());
  if (x.rows < 2 || x.cols < 1) {
    throw std::invalid_argument("KODAMAGraph requires at least two rows and one column.");
  }

  detail::Timer timer;
  detail::Timer input_copy_timer;
  const std::vector<float> data = copy_float32(x);
  const double input_copy_seconds = input_copy_timer.seconds();
  KODAMAGraphResult result;
  result.samples = static_cast<int>(x.rows);
  result.dimensions = static_cast<int>(x.cols);
  result.backend = options.backend;
  result.neighbors = std::max(1, std::min(options.neighbors, result.samples - 1));
  const double exact_work = static_cast<double>(x.rows) * x.rows * x.cols;
  const bool direct_ivf =
    (options.backend == Backend::CUDA || options.backend == Backend::Metal) &&
    !detail::should_use_spatial_grid_knn(result.samples, result.dimensions, options.metric) &&
    ((options.backend == Backend::CUDA && options.index_type == KNNIndexType::CudaIVFFlat) ||
     (options.backend == Backend::Metal && options.index_type == KNNIndexType::MetalIVFFlat) ||
     (x.rows > 5000 && exact_work > 2.0e8));
  NeighborGraph base_graph;
  std::shared_ptr<DeviceResidentKODAMAGraph> resident;
  if (direct_ivf) {
    resident = std::make_shared<DeviceResidentKODAMAGraph>();
    KODAMAMatrixOptions bridge;
    bridge.backend = options.backend;
    bridge.knn.index_type = options.index_type;
    bridge.knn.ivf_nlist = options.ivf_nlist;
    bridge.knn.ivf_nprobe = options.ivf_nprobe;
    bridge.knn.gpu_device = options.gpu_device;
    detail::Timer graph_timer;
    resident->build_ivf(
      data, result.samples, result.dimensions, result.neighbors, options.metric,
      bridge, std::max(1, options.n_threads), &result.ivf_nlist,
      &result.ivf_nprobe, &result.ivf_pilot_recall
    );
    result.index_type = options.backend == Backend::CUDA ?
      KNNIndexType::CudaIVFFlat : KNNIndexType::MetalIVFFlat;
    result.graph_seconds = graph_timer.seconds();
    result.graph_builds = 1;
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = std::max(1, options.n_threads);
    init_options.seed = options.seed;
    init_options.gpu_device = options.gpu_device;
    init_options.backend = options.backend;
    result.visual_init = KODAMAVisualizationPCAInit(x, init_options);
    result.visual_init_seconds = result.visual_init.runtime_seconds;
    result.runtime_seconds = result.graph_seconds + result.visual_init_seconds;
    result.graph_storage_bytes = static_cast<std::uint64_t>(result.samples) *
      static_cast<std::uint64_t>(result.neighbors) * (sizeof(int) + sizeof(float));
  } else {
    result = build_kodama_graph(
      data, result.samples, result.dimensions, options, true, false);
    result.neighbors = result.knn.neighbors;
    base_graph = std::move(result.knn);
    if (options.backend == Backend::CUDA || options.backend == Backend::Metal) {
      resident = std::make_shared<DeviceResidentKODAMAGraph>();
      resident->build(
        base_graph, result.samples, options.backend, options.gpu_device,
        std::max(1, options.n_threads));
    }
  }
  if (spatial != nullptr) {
    if (spatial->data == nullptr || spatial->rows != x.rows || spatial->cols < 1) {
      throw std::invalid_argument(
        "KODAMAGraph spatial coordinates must have one row per data sample."
      );
    }
    detail::Timer spatial_timer;
    const std::vector<float> spatial_data = KODAMASeparateSpatialSamples(
      copy_float32(*spatial),
      static_cast<int>(x.rows),
      static_cast<int>(spatial->cols),
      options.samples
    );
    const int spatial_neighbors = std::min(
      static_cast<int>(x.rows) - 1,
      std::max(20, result.neighbors)
    );
    NeighborGraph spatial_with_self = self_knn_graph(
      spatial_data,
      static_cast<int>(x.rows),
      static_cast<int>(spatial->cols),
      spatial_neighbors + 1,
      DistanceMetric::Euclidean,
      std::max(1, options.n_threads),
      options.backend,
      options.gpu_device,
      true,
      options.index_type,
      options.ivf_nlist,
      options.ivf_nprobe
    );
    result.spatial_jitter = spatial_jitter_from_precomputed_graph(
      spatial_data,
      spatial_with_self,
      static_cast<int>(x.rows),
      static_cast<int>(spatial->cols)
    );
    trim_self_neighbors_in_place(
      spatial_with_self,
      static_cast<int>(x.rows),
      spatial_neighbors
    );
    result.spatial_knn = std::move(spatial_with_self);
    make_graph_indices_one_based(result.spatial_knn);
    result.spatial_dimensions = static_cast<int>(spatial->cols);
    result.spatial_graph_builds = 1;
    result.spatial_graph_seconds = spatial_timer.seconds();
    result.graph_storage_bytes +=
      static_cast<std::uint64_t>(result.spatial_knn.indices.capacity()) * sizeof(int) +
      static_cast<std::uint64_t>(result.spatial_knn.distances.capacity()) * sizeof(float) +
      static_cast<std::uint64_t>(result.spatial_jitter.capacity()) * sizeof(float);
  }
  NeighborGraph retained_host;
  if (!base_graph.indices.empty()) retained_host = base_graph;
  result.handle = detail::KODAMAGraphHandleAccess::create(
    options.backend,
    static_cast<int>(x.rows),
    result.neighbors,
    std::move(resident),
    std::move(retained_host)
  );
  if (options.materialize_graph) {
    result.knn = KODAMAGraphMaterialize(result);
  }
  result.input_copy_seconds = input_copy_seconds;
  result.runtime_seconds = timer.seconds();
  const double measured_wall =
    result.input_copy_seconds + result.graph_seconds +
    result.spatial_graph_seconds + result.visual_init_seconds;
  const double orchestration_seconds =
    std::max(0.0, result.runtime_seconds - measured_wall);
  result.timings = {
    {"input_copy", result.input_copy_seconds, result.input_copy_seconds},
    {"knn_graph", result.graph_seconds, result.graph_seconds},
    {"spatial_graph", result.spatial_graph_seconds,
      result.spatial_graph_seconds},
    {"visual_initialization", result.visual_init_seconds,
      result.visual_init_seconds},
    {"orchestration_and_materialization", orchestration_seconds,
      orchestration_seconds},
    {"total", result.runtime_seconds, result.runtime_seconds}
  };
  return result;
}

}  // namespace

std::vector<float> KODAMASeparateSpatialSamples(
  const std::vector<float>& spatial,
  const int rows,
  const int columns,
  const std::vector<int>& samples
) {
  if (rows < 0 || columns < 1 ||
      spatial.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)) {
    throw std::invalid_argument(
      "Spatial coordinates must contain rows * columns values."
    );
  }
  if (samples.empty()) return spatial;
  if (samples.size() != static_cast<std::size_t>(rows)) {
    throw std::invalid_argument("samples size must match the number of spatial rows.");
  }

  std::vector<int> sample_names = samples;
  std::sort(sample_names.begin(), sample_names.end());
  sample_names.erase(std::unique(sample_names.begin(), sample_names.end()), sample_names.end());
  if (sample_names.size() <= 1u) return spatial;

  std::vector<float> separated = spatial;
  double offset = 0.0;
  for (const int sample : sample_names) {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    bool found = false;
    for (int row = 0; row < rows; ++row) {
      if (samples[static_cast<std::size_t>(row)] != sample) continue;
      float& coordinate = separated[static_cast<std::size_t>(row) * columns];
      coordinate = static_cast<float>(static_cast<double>(coordinate) + offset);
      minimum = std::min(minimum, coordinate);
      maximum = std::max(maximum, coordinate);
      found = true;
    }
    if (found) {
      const double range = static_cast<double>(maximum) - static_cast<double>(minimum);
      offset = static_cast<double>(maximum) + 0.5 * range;
    }
  }
  return separated;
}

KODAMAGraphResult KODAMAGraph_CPU(
  MatrixView x,
  const KODAMAGraphOptions& options
) {
  KODAMAGraphOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  return run_public_kodama_graph(x, cpu_options);
}

KODAMAGraphResult KODAMAGraph_CPU(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options
) {
  KODAMAGraphOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  return run_public_kodama_graph(x, cpu_options, &spatial);
}

KODAMAGraphResult KODAMAGraph_CUDA(
  MatrixView x,
  const KODAMAGraphOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAGraphOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_public_kodama_graph(x, cuda_options);
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAGraph_CUDA requires a CUDA build.");
#endif
}

KODAMAGraphResult KODAMAGraph_CUDA(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAGraphOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_public_kodama_graph(x, cuda_options, &spatial);
#else
  (void)x;
  (void)spatial;
  (void)options;
  throw std::runtime_error("KODAMAGraph_CUDA requires a CUDA build.");
#endif
}

KODAMAGraphResult KODAMAGraph_METAL(
  MatrixView x,
  const KODAMAGraphOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAGraphOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  return run_public_kodama_graph(x, metal_options);
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAGraph_METAL requires an Apple Metal build.");
#endif
}

KODAMAGraphResult KODAMAGraph_METAL(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAGraphOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  return run_public_kodama_graph(x, metal_options, &spatial);
#else
  (void)x;
  (void)spatial;
  (void)options;
  throw std::runtime_error("KODAMAGraph_METAL requires an Apple Metal build.");
#endif
}

KODAMAGraphResult KODAMAGraph(
  MatrixView x,
  const KODAMAGraphOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAGraph_CUDA(x, options);
  if (options.backend == Backend::Metal) return KODAMAGraph_METAL(x, options);
  return KODAMAGraph_CPU(x, options);
}

KODAMAGraphResult KODAMAGraph(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAGraph_CUDA(x, spatial, options);
  if (options.backend == Backend::Metal) return KODAMAGraph_METAL(x, spatial, options);
  return KODAMAGraph_CPU(x, spatial, options);
}

NeighborGraph KODAMAGraphMaterialize(const KODAMAGraphResult& graph) {
  if (!graph.knn.indices.empty()) return graph.knn;
  if (!graph.handle || !graph.handle->impl_) {
    throw std::invalid_argument("KODAMAGraphMaterialize requires a valid graph handle.");
  }
  std::lock_guard<std::mutex> lock(graph.handle->impl_->mutex);
  NeighborGraph materialized;
  if (!graph.handle->impl_->host_graph.indices.empty()) {
    materialized = graph.handle->impl_->host_graph;
  } else if (graph.handle->impl_->resident && graph.handle->impl_->resident->valid()) {
    graph.handle->impl_->resident->reset_graph();
    materialized = graph.handle->impl_->resident->download();
  } else {
    throw std::runtime_error("The graph handle has no materializable graph storage.");
  }
  make_graph_indices_one_based(materialized);
  return materialized;
}

void KODAMADissimilarityInPlace(
  NeighborGraph& graph,
  const std::vector<int>& run_labels,
  int runs,
  int samples,
  Backend backend,
  int n_threads,
  int gpu_device
) {
#if !defined(KODAMA_ENABLE_CUDA)
  (void)gpu_device;
#endif
  if (runs <= 0 || samples <= 0 || graph.neighbors <= 0) {
    throw std::invalid_argument("KODAMADissimilarityInPlace requires a non-empty graph and label matrix.");
  }
  const std::size_t graph_size =
    static_cast<std::size_t>(samples) * static_cast<std::size_t>(graph.neighbors);
  if (graph.indices.size() != graph_size || graph.distances.size() != graph_size) {
    throw std::invalid_argument("KODAMADissimilarityInPlace graph dimensions are inconsistent.");
  }
  if (run_labels.size() != static_cast<std::size_t>(runs) * static_cast<std::size_t>(samples)) {
    throw std::invalid_argument("KODAMADissimilarityInPlace label matrix dimensions are inconsistent.");
  }
  for (const int index : graph.indices) {
    if (index < 0) continue;
    if (index < 1 || index > samples) {
      throw std::invalid_argument("KODAMADissimilarityInPlace expects public one-based graph indices.");
    }
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (backend == Backend::CUDA) {
    detail::apply_kodama_dissimilarity_cuda(
      graph,
      run_labels,
      runs,
      samples,
      gpu_device,
      true,
      true
    );
    return;
  }
#else
  if (backend == Backend::CUDA) {
    throw std::runtime_error("KODAMADissimilarityInPlace CUDA backend is not enabled.");
  }
#endif
#if defined(KODAMA_ENABLE_METAL)
  if (backend == Backend::Metal) {
    detail::NativeMetalKODAMAGraph resident =
      detail::metal_build_resident_kodama_graph(graph, samples, 1);
    detail::metal_prepare_resident_results(resident, runs);
    for (int run = 0; run < runs; ++run) {
      const auto begin = run_labels.begin() + static_cast<std::size_t>(run) * samples;
      detail::metal_store_resident_result_row(
        resident, std::vector<int>(begin, begin + samples), run, 0);
    }
    detail::metal_apply_resident_kodama_dissimilarity(
      resident,
      runs,
      true,
      true
    );
    graph = detail::metal_download_resident_kodama_graph(resident);
    return;
  }
#else
  if (backend == Backend::Metal) {
    throw std::runtime_error("KODAMADissimilarityInPlace Metal backend is not enabled.");
  }
#endif
  apply_kodama_dissimilarity(
    graph,
    run_labels,
    runs,
    samples,
    std::max(1, n_threads),
    true,
    true
  );
}

KODAMAMatrixResult KODAMAMatrix_CPU(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, cpu_options);
}

KODAMAMatrixResult KODAMAMatrix_CUDA(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, cuda_options);
#else
  (void)x;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrix_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrix_METAL(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  metal_options.pls.backend = Backend::Metal;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, metal_options);
#else
  (void)x;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrix_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrix_CUDA(x, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrix_METAL(x, starting_labels, constrain, fixed, options);
  return KODAMAMatrix_CPU(x, starting_labels, constrain, fixed, options);
}

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const KODAMAGraphResult& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (graph.samples != static_cast<int>(x.rows)) {
    throw std::invalid_argument("KODAMAMatrix graph sample count must match the data rows.");
  }
  if (graph.dimensions > 0 && graph.dimensions != static_cast<int>(x.cols)) {
    throw std::invalid_argument("KODAMAMatrix graph dimensions must match the data columns.");
  }

  const bool reuse_visual_init =
    graph.visual_init.samples == static_cast<int>(x.rows) &&
    graph.visual_init.components == 2 &&
    graph.visual_init.backend == options.backend &&
    graph.visual_init.umap.size() == x.rows * 2 &&
    graph.visual_init.opentsne.size() == x.rows * 2;

  KODAMAMatrixOptions prepared_options = options;
  prepared_options.compute_visual_init = options.compute_visual_init && !reuse_visual_init;
  if (prepared_options.backend == Backend::Metal) {
    prepared_options.knn.backend = Backend::Metal;
    prepared_options.pls.backend = Backend::Metal;
  } else if (prepared_options.backend == Backend::CUDA) {
    prepared_options.knn.backend = Backend::CUDA;
    prepared_options.pls.backend = Backend::CUDA;
  }
  KODAMAMatrixResult result = run_kodama_matrix(
    x,
    starting_labels,
    constrain,
    fixed,
    prepared_options,
    graph.knn.indices.empty() ? nullptr : &graph.knn,
    &graph
  );
  result.graph_backend = graph.backend;
  result.graph_index_type = graph.index_type;
  result.graph_ivf_nlist = graph.ivf_nlist;
  result.graph_ivf_nprobe = graph.ivf_nprobe;
  result.graph_ivf_pilot_recall = graph.ivf_pilot_recall;
  if (reuse_visual_init) {
    result.visual_init = graph.visual_init;
    result.has_visual_init = true;
    result.visual_init_seconds = 0.0;
  }
  return result;
}

KODAMAMatrixResult KODAMAMatrix(
  const KODAMAGraphResult& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (graph.samples < 2) {
    throw std::invalid_argument("KODAMAMatrix requires a non-empty KODAMAGraphResult.");
  }
  const NeighborGraph materialized = KODAMAGraphMaterialize(graph);
  KODAMAMatrixOptions prepared_options = options;
  prepared_options.compute_visual_init = false;
  KODAMAMatrixResult result = KODAMAMatrixFromGraph(
    materialized,
    graph.samples,
    starting_labels,
    constrain,
    fixed,
    prepared_options
  );
  const bool reuse_visual_init =
    graph.visual_init.samples == graph.samples &&
    graph.visual_init.components == 2 &&
    graph.visual_init.backend == options.backend &&
    graph.visual_init.umap.size() == static_cast<std::size_t>(graph.samples) * 2 &&
    graph.visual_init.opentsne.size() == static_cast<std::size_t>(graph.samples) * 2;
  if (reuse_visual_init) {
    result.visual_init = graph.visual_init;
    result.has_visual_init = true;
    result.visual_init_seconds = 0.0;
  }
  return result;
}

std::vector<float> KODAMAGraphFeatures_CPU(
  const NeighborGraph& graph,
  int samples,
  const KODAMAMatrixOptions& options
) {
  const int components = std::max(
    1,
    options.graph_feature_components > 0 ? options.graph_feature_components : std::max(1, options.components)
  );
  return graph_laplacian_operator_features(
    graph,
    samples,
    components,
    std::max(8, options.graph_feature_steps),
    options.seed,
    options.n_threads
  );
}

KODAMAMatrixResult KODAMAMatrixFromGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, cpu_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, cpu_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_CPU(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, cpu_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_CUDA(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, cuda_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
#else
  (void)x;
  (void)graph;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraphData_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_METAL(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  metal_options.pls.backend = Backend::Metal;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, metal_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
#else
  (void)x;
  (void)graph;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraphData_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraphData(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrixFromGraphData_CUDA(x, graph, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrixFromGraphData_METAL(x, graph, starting_labels, constrain, fixed, options);
  return KODAMAMatrixFromGraphData_CPU(x, graph, starting_labels, constrain, fixed, options);
}

KODAMAMatrixResult KODAMAMatrixFromGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, cuda_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, cuda_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
#else
  (void)graph;
  (void)samples;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraph_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  metal_options.pls.backend = Backend::Metal;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, metal_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, metal_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
#else
  (void)graph;
  (void)samples;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraph_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraph(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrixFromGraph_CUDA(graph, samples, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrixFromGraph_METAL(graph, samples, starting_labels, constrain, fixed, options);
  return KODAMAMatrixFromGraph_CPU(graph, samples, starting_labels, constrain, fixed, options);
}

}  // namespace kodama
