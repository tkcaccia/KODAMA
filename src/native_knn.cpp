/*
 * SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Compact HNSW implementation distilled from the algorithmic organization in
 * FAISS 1.14.3, commit 0ca9df4792b173d573044ee14ca0704780176e82.
 *
 * Licensed under the MIT License. The FAISS copyright and MIT license must be
 * retained with redistributed derivatives of this file.
 */

#include "native_knn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && \
  (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KODAMA_NATIVE_AVX2_DISPATCH 1
#endif

namespace kodama::detail {
namespace {

#ifdef KODAMA_NATIVE_AVX2_DISPATCH
__attribute__((target("avx2,fma"))) float avx2_distance(
  const float* a,
  const float* b,
  int dimensions,
  bool euclidean
) {
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  int d = 0;
  for (; d + 31 < dimensions; d += 32) {
    const __m256 a0 = _mm256_loadu_ps(a + d);
    const __m256 a1 = _mm256_loadu_ps(a + d + 8);
    const __m256 a2 = _mm256_loadu_ps(a + d + 16);
    const __m256 a3 = _mm256_loadu_ps(a + d + 24);
    const __m256 b0 = _mm256_loadu_ps(b + d);
    const __m256 b1 = _mm256_loadu_ps(b + d + 8);
    const __m256 b2 = _mm256_loadu_ps(b + d + 16);
    const __m256 b3 = _mm256_loadu_ps(b + d + 24);
    if (euclidean) {
      const __m256 delta0 = _mm256_sub_ps(a0, b0);
      const __m256 delta1 = _mm256_sub_ps(a1, b1);
      const __m256 delta2 = _mm256_sub_ps(a2, b2);
      const __m256 delta3 = _mm256_sub_ps(a3, b3);
      sum0 = _mm256_fmadd_ps(delta0, delta0, sum0);
      sum1 = _mm256_fmadd_ps(delta1, delta1, sum1);
      sum2 = _mm256_fmadd_ps(delta2, delta2, sum2);
      sum3 = _mm256_fmadd_ps(delta3, delta3, sum3);
    } else {
      sum0 = _mm256_fmadd_ps(a0, b0, sum0);
      sum1 = _mm256_fmadd_ps(a1, b1, sum1);
      sum2 = _mm256_fmadd_ps(a2, b2, sum2);
      sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }
  }
  const __m256 sum01 = _mm256_add_ps(sum0, sum1);
  const __m256 sum23 = _mm256_add_ps(sum2, sum3);
  const __m256 sum = _mm256_add_ps(sum01, sum23);
  const __m128 low = _mm256_castps256_ps128(sum);
  const __m128 high = _mm256_extractf128_ps(sum, 1);
  const __m128 sum128 = _mm_add_ps(low, high);
  alignas(16) float lanes[4];
  _mm_store_ps(lanes, sum128);
  float result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
  if (euclidean) {
    for (; d < dimensions; ++d) {
      const float delta = a[d] - b[d];
      result += delta * delta;
    }
    return result;
  }
  for (; d < dimensions; ++d) result += a[d] * b[d];
  return -result;
}

bool cpu_has_avx2_fma() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

struct NodeDistance {
  float distance = 0.0f;
  int id = -1;
};

struct CloserFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance > b.distance;
    return a.id > b.id;
  }
};

struct FartherFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.id < b.id;
  }
};

bool closer(const NodeDistance& a, const NodeDistance& b) {
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

class CompactHNSW {
 public:
  CompactHNSW(
    const std::vector<float>& data,
    int n,
    int p,
    DistanceMetric metric,
    int m,
    int ef_construction,
    int ef_search
  ) :
      data_(data.data()),
      n_(n),
      p_(p),
      metric_(metric),
      m_(std::max(2, std::min(m, std::max(2, n - 1)))),
      ef_construction_(std::max(1, ef_construction)),
      ef_search_(std::max(1, ef_search)),
      node_mutexes_(std::make_unique<std::mutex[]>(static_cast<std::size_t>(std::max(0, n))))
#ifdef KODAMA_NATIVE_AVX2_DISPATCH
      , use_avx2_(cpu_has_avx2_fma())
#endif
  {
    generate_levels();
    allocate_graph();
  }

  void build(int n_threads) {
    if (n_ == 0) return;
    entry_point_ = 0;
    current_max_level_ = levels_[0];
    if (n_ == 1) {
      release_build_distances();
      return;
    }

    n_threads = std::max(1, std::min(n_threads, n_));
    const int max_neighbors = 2 * m_;
    if (n_threads == 1) {
      BuildScratch scratch(n_, ef_construction_, max_neighbors);
      ReciprocalScratch reciprocal(max_neighbors);
      for (int point_id = 1; point_id < n_; ++point_id) {
        add_point(point_id, scratch, reciprocal);
      }
      release_build_distances();
      return;
    }

    // Complete one base-layer neighborhood before concurrent insertion. This
    // gives every worker a stable connected entry graph without introducing a
    // dataset-dependent tuning threshold.
    const int parallel_begin = std::min(n_, std::max(n_threads + 1, 2 * m_ + 1));
    {
      BuildScratch scratch(n_, ef_construction_, max_neighbors);
      ReciprocalScratch reciprocal(max_neighbors);
      for (int point_id = 1; point_id < parallel_begin; ++point_id) {
        add_point(point_id, scratch, reciprocal);
      }
    }

    parallel_build_ = true;
    std::atomic<int> next{parallel_begin};
    auto worker = [&]() {
      BuildScratch scratch(n_, ef_construction_, max_neighbors);
      ReciprocalScratch reciprocal(max_neighbors);
      for (;;) {
        const int point_id = next.fetch_add(1, std::memory_order_relaxed);
        if (point_id >= n_) break;
        add_point(point_id, scratch, reciprocal);
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int worker_id = 1; worker_id < n_threads; ++worker_id) {
      workers.emplace_back(worker);
    }
    worker();
    for (std::thread& thread : workers) thread.join();
    parallel_build_ = false;
    release_build_distances();
  }

  NativeKNNResult search(
    const std::vector<float>& queries,
    int query_rows,
    int requested_k,
    int n_threads,
    const std::vector<int>& query_train_indices
  ) const {
    if (static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(p_) != queries.size()) {
      throw std::invalid_argument("native HNSW query matrix size mismatch.");
    }
    if (!query_train_indices.empty() && static_cast<int>(query_train_indices.size()) != query_rows) {
      throw std::invalid_argument("native HNSW query/train index size mismatch.");
    }

    const bool excludes_self = !query_train_indices.empty();
    const int available = n_ - (excludes_self ? 1 : 0);
    const int k = std::min(requested_k, std::max(0, available));
    NativeKNNResult output;
    output.queries = query_rows;
    output.neighbors = k;
    output.indices.assign(static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k), -1);
    output.distances.assign(
      static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k),
      std::numeric_limits<float>::infinity()
    );
    if (k == 0 || query_rows == 0) return output;

    constexpr int query_batch = 8;
    std::atomic<int> next{0};
    n_threads = std::max(1, std::min(n_threads, query_rows));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int worker_id = 0; worker_id < n_threads; ++worker_id) {
      workers.emplace_back([&]() {
        SearchScratch scratch(n_, std::max(k + 1, ef_search_), 2 * m_);
        for (;;) {
          const int begin = next.fetch_add(query_batch, std::memory_order_relaxed);
          if (begin >= query_rows) break;
          const int end = std::min(query_rows, begin + query_batch);
          for (int query_id = begin; query_id < end; ++query_id) {
            const float* query =
              queries.data() + static_cast<std::size_t>(query_id) * static_cast<std::size_t>(p_);
            const int excluded = query_train_indices.empty() ?
              -1 : query_train_indices[static_cast<std::size_t>(query_id)];
            search_query(query, std::max(k + 1, ef_search_), scratch);
            int used = 0;
            for (const NodeDistance& candidate : scratch.layer_candidates) {
              if (candidate.id == excluded) continue;
              const std::size_t pos =
                static_cast<std::size_t>(query_id) * static_cast<std::size_t>(k) +
                static_cast<std::size_t>(used);
              output.indices[pos] = candidate.id;
              output.distances[pos] = candidate.distance;
              if (++used == k) break;
            }
            if (used < k) exact_fill(query, query_id, excluded, k, output);
          }
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
    return output;
  }

  NativeKNNResult search_filtered(
    const std::vector<float>& queries,
    int query_rows,
    int requested_k,
    int n_threads,
    const std::vector<int>& query_train_indices,
    const std::vector<int>& allowed_local_ids
  ) const {
    if (queries.size() != static_cast<std::size_t>(query_rows) * p_ ||
        static_cast<int>(query_train_indices.size()) != query_rows ||
        static_cast<int>(allowed_local_ids.size()) != n_) {
      throw std::invalid_argument("native HNSW filtered-query size mismatch.");
    }
    const int allowed = static_cast<int>(std::count_if(
      allowed_local_ids.begin(), allowed_local_ids.end(), [](int id) { return id >= 0; }
    ));
    const int k = std::min(requested_k, std::max(0, allowed - 1));
    NativeKNNResult output;
    output.queries = query_rows;
    output.neighbors = k;
    output.indices.assign(static_cast<std::size_t>(query_rows) * k, -1);
    output.distances.assign(
      static_cast<std::size_t>(query_rows) * k,
      std::numeric_limits<float>::infinity()
    );
    if (query_rows == 0 || k == 0) return output;

    std::atomic<int> next{0};
    n_threads = std::max(1, std::min(n_threads, query_rows));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int worker = 0; worker < n_threads; ++worker) {
      workers.emplace_back([&]() {
        SearchScratch scratch(n_, std::max(k + 1, ef_search_), 2 * m_);
        std::vector<NodeDistance> exact_candidates;
        exact_candidates.reserve(static_cast<std::size_t>(allowed));
        for (;;) {
          const int query_id = next.fetch_add(1, std::memory_order_relaxed);
          if (query_id >= query_rows) break;
          const float* query = queries.data() + static_cast<std::size_t>(query_id) * p_;
          const int excluded = query_train_indices[static_cast<std::size_t>(query_id)];
          int ef = std::max(k + 1, ef_search_);
          int used = 0;
          for (;;) {
            search_query(query, ef, scratch);
            used = 0;
            for (const NodeDistance& candidate : scratch.layer_candidates) {
              if (candidate.id == excluded) continue;
              const int local = allowed_local_ids[static_cast<std::size_t>(candidate.id)];
              if (local < 0) continue;
              const std::size_t pos = static_cast<std::size_t>(query_id) * k + used;
              output.indices[pos] = local;
              output.distances[pos] = candidate.distance;
              if (++used == k) break;
            }
            if (used == k || ef >= n_) break;
            ef = std::min(n_, std::max(ef + 1, 2 * ef));
          }
          if (used == k) continue;

          exact_candidates.clear();
          for (int global = 0; global < n_; ++global) {
            const int local = allowed_local_ids[static_cast<std::size_t>(global)];
            if (local < 0 || global == excluded) continue;
            exact_candidates.push_back({distance(query, point(global)), local});
          }
          std::partial_sort(
            exact_candidates.begin(), exact_candidates.begin() + k,
            exact_candidates.end(), closer
          );
          for (int rank = 0; rank < k; ++rank) {
            const std::size_t pos = static_cast<std::size_t>(query_id) * k + rank;
            output.indices[pos] = exact_candidates[static_cast<std::size_t>(rank)].id;
            output.distances[pos] = exact_candidates[static_cast<std::size_t>(rank)].distance;
          }
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
    return output;
  }

 private:
  struct VisitTable {
    explicit VisitTable(int n) : marks(static_cast<std::size_t>(n), 0), generation(1) {}

    void reset() {
      if (++generation == 0) {
        std::fill(marks.begin(), marks.end(), 0);
        generation = 1;
      }
    }

    bool set(int id) {
      if (marks[static_cast<std::size_t>(id)] == generation) return false;
      marks[static_cast<std::size_t>(id)] = generation;
      return true;
    }

    std::vector<std::uint32_t> marks;
    std::uint32_t generation;
  };

  struct SearchScratch {
    SearchScratch(int n, int ef, int max_neighbors) : visited(n) {
      candidate_heap.reserve(static_cast<std::size_t>(ef) + 1u);
      result_heap.reserve(static_cast<std::size_t>(ef) + 1u);
      layer_candidates.reserve(static_cast<std::size_t>(ef));
      neighbor_ids.reserve(static_cast<std::size_t>(max_neighbors));
      expansion_ids.reserve(static_cast<std::size_t>(max_neighbors));
      expansion_distances.reserve(static_cast<std::size_t>(max_neighbors));
    }

    VisitTable visited;
    std::vector<NodeDistance> candidate_heap;
    std::vector<NodeDistance> result_heap;
    std::vector<NodeDistance> layer_candidates;
    std::vector<int> neighbor_ids;
    std::vector<int> expansion_ids;
    std::vector<float> expansion_distances;
  };

  struct BuildScratch {
    BuildScratch(int n, int ef, int max_neighbors) :
        search(n, ef, max_neighbors) {
      selected.reserve(static_cast<std::size_t>(max_neighbors));
      selected_distances.reserve(static_cast<std::size_t>(max_neighbors));
      rejected.reserve(static_cast<std::size_t>(ef));
    }

    SearchScratch search;
    std::vector<int> selected;
    std::vector<float> selected_distances;
    std::vector<NodeDistance> rejected;
  };

  struct ReciprocalScratch {
    explicit ReciprocalScratch(int max_neighbors) {
      candidates.reserve(static_cast<std::size_t>(max_neighbors) + 1u);
      selected.reserve(static_cast<std::size_t>(max_neighbors));
      selected_distances.reserve(static_cast<std::size_t>(max_neighbors));
      rejected.reserve(static_cast<std::size_t>(max_neighbors) + 1u);
    }

    std::vector<NodeDistance> candidates;
    std::vector<int> selected;
    std::vector<float> selected_distances;
    std::vector<NodeDistance> rejected;
  };

  const float* data_ = nullptr;
  int n_ = 0;
  int p_ = 0;
  DistanceMetric metric_ = DistanceMetric::Euclidean;
  int m_ = 16;
  int ef_construction_ = 80;
  int ef_search_ = 64;
  int entry_point_ = -1;
  int current_max_level_ = -1;
  std::vector<int> levels_;
  std::vector<std::size_t> node_offsets_;
  std::vector<std::size_t> level_offsets_;
  std::vector<std::uint16_t> counts_;
  std::vector<std::int32_t> neighbors_;
  std::vector<float> neighbor_distances_;
  mutable std::unique_ptr<std::mutex[]> node_mutexes_;
  mutable std::mutex entry_mutex_;
  bool parallel_build_ = false;
#ifdef KODAMA_NATIVE_AVX2_DISPATCH
  bool use_avx2_ = false;
#endif

  const float* point(int id) const {
    return data_ + static_cast<std::size_t>(id) * static_cast<std::size_t>(p_);
  }

  float distance(const float* a, const float* b) const {
#ifdef KODAMA_NATIVE_AVX2_DISPATCH
    if (use_avx2_) {
      return avx2_distance(a, b, p_, metric_ == DistanceMetric::Euclidean);
    }
#endif
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int d = 0;
    if (metric_ == DistanceMetric::Euclidean) {
      for (; d + 15 < p_; d += 16) {
        const float d0 = a[d] - b[d];
        const float d1 = a[d + 1] - b[d + 1];
        const float d2 = a[d + 2] - b[d + 2];
        const float d3 = a[d + 3] - b[d + 3];
        const float d4 = a[d + 4] - b[d + 4];
        const float d5 = a[d + 5] - b[d + 5];
        const float d6 = a[d + 6] - b[d + 6];
        const float d7 = a[d + 7] - b[d + 7];
        const float d8 = a[d + 8] - b[d + 8];
        const float d9 = a[d + 9] - b[d + 9];
        const float da = a[d + 10] - b[d + 10];
        const float db = a[d + 11] - b[d + 11];
        const float dc = a[d + 12] - b[d + 12];
        const float dd = a[d + 13] - b[d + 13];
        const float de = a[d + 14] - b[d + 14];
        const float df = a[d + 15] - b[d + 15];
        sum0 += d0 * d0 + d4 * d4 + d8 * d8 + dc * dc;
        sum1 += d1 * d1 + d5 * d5 + d9 * d9 + dd * dd;
        sum2 += d2 * d2 + d6 * d6 + da * da + de * de;
        sum3 += d3 * d3 + d7 * d7 + db * db + df * df;
      }
      float sum = (sum0 + sum1) + (sum2 + sum3);
      for (; d < p_; ++d) {
        const float delta = a[d] - b[d];
        sum += delta * delta;
      }
      return sum;
    }

    for (; d + 15 < p_; d += 16) {
      sum0 += a[d] * b[d] + a[d + 4] * b[d + 4] +
        a[d + 8] * b[d + 8] + a[d + 12] * b[d + 12];
      sum1 += a[d + 1] * b[d + 1] + a[d + 5] * b[d + 5] +
        a[d + 9] * b[d + 9] + a[d + 13] * b[d + 13];
      sum2 += a[d + 2] * b[d + 2] + a[d + 6] * b[d + 6] +
        a[d + 10] * b[d + 10] + a[d + 14] * b[d + 14];
      sum3 += a[d + 3] * b[d + 3] + a[d + 7] * b[d + 7] +
        a[d + 11] * b[d + 11] + a[d + 15] * b[d + 15];
    }
    float dot = (sum0 + sum1) + (sum2 + sum3);
    for (; d < p_; ++d) dot += a[d] * b[d];
    return -dot;
  }

  float distance(int a, int b) const {
    return distance(point(a), point(b));
  }

  bool distance_below(int a, int b, float threshold) const {
    if (metric_ != DistanceMetric::Euclidean) {
      return distance(a, b) < threshold;
    }
    const float* lhs = point(a);
    const float* rhs = point(b);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int d = 0;
    for (; d + 15 < p_; d += 16) {
      const float d0 = lhs[d] - rhs[d];
      const float d1 = lhs[d + 1] - rhs[d + 1];
      const float d2 = lhs[d + 2] - rhs[d + 2];
      const float d3 = lhs[d + 3] - rhs[d + 3];
      const float d4 = lhs[d + 4] - rhs[d + 4];
      const float d5 = lhs[d + 5] - rhs[d + 5];
      const float d6 = lhs[d + 6] - rhs[d + 6];
      const float d7 = lhs[d + 7] - rhs[d + 7];
      const float d8 = lhs[d + 8] - rhs[d + 8];
      const float d9 = lhs[d + 9] - rhs[d + 9];
      const float da = lhs[d + 10] - rhs[d + 10];
      const float db = lhs[d + 11] - rhs[d + 11];
      const float dc = lhs[d + 12] - rhs[d + 12];
      const float dd = lhs[d + 13] - rhs[d + 13];
      const float de = lhs[d + 14] - rhs[d + 14];
      const float df = lhs[d + 15] - rhs[d + 15];
      sum0 += d0 * d0 + d4 * d4 + d8 * d8 + dc * dc;
      sum1 += d1 * d1 + d5 * d5 + d9 * d9 + dd * dd;
      sum2 += d2 * d2 + d6 * d6 + da * da + de * de;
      sum3 += d3 * d3 + d7 * d7 + db * db + df * df;
      if ((sum0 + sum1) + (sum2 + sum3) >= threshold) return false;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; d < p_; ++d) {
      const float delta = lhs[d] - rhs[d];
      sum += delta * delta;
      if (sum >= threshold) return false;
    }
    return sum < threshold;
  }

  int capacity(int level) const {
    return level == 0 ? 2 * m_ : m_;
  }

  std::size_t level_index(int node, int level) const {
    return level_offsets_[static_cast<std::size_t>(node)] + static_cast<std::size_t>(level);
  }

  std::size_t neighbor_offset(int node, int level) const {
    return node_offsets_[static_cast<std::size_t>(node)] +
      (level == 0 ? 0u : static_cast<std::size_t>(2 * m_ + (level - 1) * m_));
  }

  void generate_levels() {
    levels_.resize(static_cast<std::size_t>(n_));
    std::mt19937 generator(12345u);
    std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
    const double level_multiplier = 1.0 / std::log(static_cast<double>(m_));
    for (int i = 0; i < n_; ++i) {
      levels_[static_cast<std::size_t>(i)] =
        static_cast<int>(-std::log(uniform(generator)) * level_multiplier);
    }
  }

  void allocate_graph() {
    node_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    level_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    for (int i = 0; i < n_; ++i) {
      node_offsets_[static_cast<std::size_t>(i + 1)] =
        node_offsets_[static_cast<std::size_t>(i)] +
        static_cast<std::size_t>(2 * m_ + levels_[static_cast<std::size_t>(i)] * m_);
      level_offsets_[static_cast<std::size_t>(i + 1)] =
        level_offsets_[static_cast<std::size_t>(i)] +
        static_cast<std::size_t>(levels_[static_cast<std::size_t>(i)] + 1);
    }
    neighbors_.assign(node_offsets_.back(), -1);
    neighbor_distances_.assign(node_offsets_.back(), std::numeric_limits<float>::infinity());
    counts_.assign(level_offsets_.back(), 0);
  }

  void release_build_distances() {
    std::vector<float>().swap(neighbor_distances_);
  }

  std::pair<const std::int32_t*, int> neighbor_range(int node, int level) const {
    const std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, counts_[level_index(node, level)]};
  }

  std::pair<std::int32_t*, std::uint16_t*> mutable_neighbor_range(int node, int level) {
    const std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, &counts_[level_index(node, level)]};
  }

  void copy_build_neighbors(int node, int level, std::vector<int>& output) const {
    std::unique_lock<std::mutex> lock(
      node_mutexes_[static_cast<std::size_t>(node)],
      std::defer_lock
    );
    if (parallel_build_) lock.lock();
    const auto range = neighbor_range(node, level);
    output.assign(range.first, range.first + range.second);
  }

  void compute_distances(
    const float* query,
    const std::vector<int>& ids,
    std::vector<float>& output
  ) const {
    output.resize(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      output[i] = distance(query, point(ids[i]));
    }
  }

  int greedy_search(
    const float* query,
    int entry,
    int level,
    float& entry_distance,
    SearchScratch& scratch
  ) const {
    bool changed = true;
    while (changed) {
      changed = false;
      if (parallel_build_) {
        copy_build_neighbors(entry, level, scratch.neighbor_ids);
        for (int candidate : scratch.neighbor_ids) {
          const float candidate_distance = distance(query, point(candidate));
          if (closer({candidate_distance, candidate}, {entry_distance, entry})) {
            entry = candidate;
            entry_distance = candidate_distance;
            changed = true;
          }
        }
      } else {
        const auto range = neighbor_range(entry, level);
        for (int j = 0; j < range.second; ++j) {
          const int candidate = range.first[j];
          const float candidate_distance = distance(query, point(candidate));
          if (closer({candidate_distance, candidate}, {entry_distance, entry})) {
            entry = candidate;
            entry_distance = candidate_distance;
            changed = true;
          }
        }
      }
    }
    return entry;
  }

  void search_layer(
    const float* query,
    int entry,
    int ef,
    int level,
    SearchScratch& scratch
  ) const {
    std::vector<NodeDistance>& candidates = scratch.candidate_heap;
    std::vector<NodeDistance>& results = scratch.result_heap;
    std::vector<NodeDistance>& output = scratch.layer_candidates;
    candidates.clear();
    results.clear();
    output.clear();
    scratch.visited.reset();

    const NodeDistance initial{distance(query, point(entry)), entry};
    candidates.push_back(initial);
    std::push_heap(candidates.begin(), candidates.end(), CloserFirst{});
    results.push_back(initial);
    std::push_heap(results.begin(), results.end(), FartherFirst{});
    scratch.visited.set(entry);

    while (!candidates.empty()) {
      const NodeDistance current = candidates.front();
      if (results.size() >= static_cast<std::size_t>(ef) &&
          closer(results.front(), current)) {
        break;
      }
      std::pop_heap(candidates.begin(), candidates.end(), CloserFirst{});
      candidates.pop_back();

      scratch.expansion_ids.clear();
      if (parallel_build_) {
        copy_build_neighbors(current.id, level, scratch.neighbor_ids);
        for (int candidate_id : scratch.neighbor_ids) {
          if (scratch.visited.set(candidate_id)) {
            scratch.expansion_ids.push_back(candidate_id);
          }
        }
      } else {
        const auto range = neighbor_range(current.id, level);
        for (int j = 0; j < range.second; ++j) {
          const int candidate_id = range.first[j];
          if (scratch.visited.set(candidate_id)) {
            scratch.expansion_ids.push_back(candidate_id);
          }
        }
      }

      compute_distances(query, scratch.expansion_ids, scratch.expansion_distances);
      for (std::size_t i = 0; i < scratch.expansion_ids.size(); ++i) {
        const NodeDistance candidate{
          scratch.expansion_distances[i],
          scratch.expansion_ids[i]
        };
        if (results.size() < static_cast<std::size_t>(ef) ||
            closer(candidate, results.front())) {
          candidates.push_back(candidate);
          std::push_heap(candidates.begin(), candidates.end(), CloserFirst{});
          results.push_back(candidate);
          std::push_heap(results.begin(), results.end(), FartherFirst{});
          if (results.size() > static_cast<std::size_t>(ef)) {
            std::pop_heap(results.begin(), results.end(), FartherFirst{});
            results.pop_back();
          }
        }
      }
    }

    output.reserve(results.size());
    while (!results.empty()) {
      output.push_back(results.front());
      std::pop_heap(results.begin(), results.end(), FartherFirst{});
      results.pop_back();
    }
    std::sort(output.begin(), output.end(), closer);
  }

  void select_diverse(
    int query_id,
    const std::vector<NodeDistance>& candidates,
    int max_size,
    std::vector<int>& selected,
    std::vector<float>& selected_distances,
    std::vector<NodeDistance>& rejected
  ) const {
    selected.clear();
    selected_distances.clear();
    rejected.clear();
    for (const NodeDistance& candidate : candidates) {
      if (candidate.id == query_id) continue;
      bool good = true;
      for (int other : selected) {
        if (distance_below(candidate.id, other, candidate.distance)) {
          good = false;
          break;
        }
      }
      if (good) {
        selected.push_back(candidate.id);
        selected_distances.push_back(candidate.distance);
        if (static_cast<int>(selected.size()) == max_size) break;
      } else {
        rejected.push_back(candidate);
      }
    }
    for (const NodeDistance& candidate : rejected) {
      if (static_cast<int>(selected.size()) == max_size) break;
      selected.push_back(candidate.id);
      selected_distances.push_back(candidate.distance);
    }
  }

  void replace_neighbors_unlocked(
    int node,
    int level,
    const std::vector<int>& selected,
    const std::vector<float>& selected_distances
  ) {
    auto range = mutable_neighbor_range(node, level);
    const std::size_t offset = neighbor_offset(node, level);
    const int cap = capacity(level);
    const int size = std::min(cap, static_cast<int>(selected.size()));
    for (int i = 0; i < size; ++i) {
      range.first[i] = selected[static_cast<std::size_t>(i)];
      neighbor_distances_[offset + static_cast<std::size_t>(i)] =
        selected_distances[static_cast<std::size_t>(i)];
    }
    for (int i = size; i < cap; ++i) {
      range.first[i] = -1;
      neighbor_distances_[offset + static_cast<std::size_t>(i)] =
        std::numeric_limits<float>::infinity();
    }
    *range.second = static_cast<std::uint16_t>(size);
  }

  void replace_neighbors(
    int node,
    int level,
    const std::vector<int>& selected,
    const std::vector<float>& selected_distances
  ) {
    std::unique_lock<std::mutex> lock(
      node_mutexes_[static_cast<std::size_t>(node)],
      std::defer_lock
    );
    if (parallel_build_) lock.lock();
    replace_neighbors_unlocked(node, level, selected, selected_distances);
  }

  void add_reciprocal(
    int node,
    int other,
    int level,
    ReciprocalScratch& scratch
  ) {
    std::unique_lock<std::mutex> lock(
      node_mutexes_[static_cast<std::size_t>(node)],
      std::defer_lock
    );
    if (parallel_build_) lock.lock();
    auto range = mutable_neighbor_range(node, level);
    const int count = *range.second;
    for (int i = 0; i < count; ++i) {
      if (range.first[i] == other) return;
    }
    const int cap = capacity(level);
    const std::size_t offset = neighbor_offset(node, level);
    if (count < cap) {
      range.first[count] = other;
      neighbor_distances_[offset + static_cast<std::size_t>(count)] =
        distance(node, other);
      *range.second = static_cast<std::uint16_t>(count + 1);
      return;
    }

    scratch.candidates.clear();
    for (int i = 0; i < count; ++i) {
      scratch.candidates.push_back({
        neighbor_distances_[offset + static_cast<std::size_t>(i)],
        range.first[i]
      });
    }
    scratch.candidates.push_back({distance(node, other), other});
    std::sort(scratch.candidates.begin(), scratch.candidates.end(), closer);
    select_diverse(
      node,
      scratch.candidates,
      cap,
      scratch.selected,
      scratch.selected_distances,
      scratch.rejected
    );
    replace_neighbors_unlocked(
      node,
      level,
      scratch.selected,
      scratch.selected_distances
    );
  }

  void add_point(
    int point_id,
    BuildScratch& scratch,
    ReciprocalScratch& reciprocal
  ) {
    int nearest = -1;
    int max_level = -1;
    if (parallel_build_) {
      std::lock_guard<std::mutex> lock(entry_mutex_);
      nearest = entry_point_;
      max_level = current_max_level_;
    } else {
      nearest = entry_point_;
      max_level = current_max_level_;
    }

    float nearest_distance = distance(point_id, nearest);
    for (int level = max_level;
         level > levels_[static_cast<std::size_t>(point_id)];
         --level) {
      nearest = greedy_search(
        point(point_id),
        nearest,
        level,
        nearest_distance,
        scratch.search
      );
    }
    for (int level = std::min(
           levels_[static_cast<std::size_t>(point_id)],
           max_level
         );
         level >= 0;
         --level) {
      search_layer(
        point(point_id),
        nearest,
        ef_construction_,
        level,
        scratch.search
      );
      select_diverse(
        point_id,
        scratch.search.layer_candidates,
        capacity(level),
        scratch.selected,
        scratch.selected_distances,
        scratch.rejected
      );
      replace_neighbors(
        point_id,
        level,
        scratch.selected,
        scratch.selected_distances
      );
      for (int other : scratch.selected) {
        add_reciprocal(other, point_id, level, reciprocal);
      }
      if (!scratch.search.layer_candidates.empty()) {
        nearest = scratch.search.layer_candidates.front().id;
      }
    }

    const int point_level = levels_[static_cast<std::size_t>(point_id)];
    if (parallel_build_) {
      std::lock_guard<std::mutex> lock(entry_mutex_);
      if (point_level > current_max_level_ ||
          (point_level == current_max_level_ && point_id < entry_point_)) {
        entry_point_ = point_id;
        current_max_level_ = point_level;
      }
    } else if (point_level > current_max_level_) {
      entry_point_ = point_id;
      current_max_level_ = point_level;
    }
  }

  void search_query(const float* query, int ef, SearchScratch& scratch) const {
    int nearest = entry_point_;
    float nearest_distance = distance(query, point(nearest));
    for (int level = current_max_level_; level >= 1; --level) {
      nearest = greedy_search(query, nearest, level, nearest_distance, scratch);
    }
    search_layer(query, nearest, ef, 0, scratch);
  }

  void exact_fill(
    const float* query,
    int query_id,
    int excluded,
    int k,
    NativeKNNResult& output
  ) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> heap;
    for (int candidate = 0; candidate < n_; ++candidate) {
      if (candidate == excluded) continue;
      const NodeDistance value{distance(query, point(candidate)), candidate};
      if (heap.size() < static_cast<std::size_t>(k) || closer(value, heap.top())) {
        heap.push(value);
        if (heap.size() > static_cast<std::size_t>(k)) heap.pop();
      }
    }
    std::vector<NodeDistance> exact;
    exact.reserve(heap.size());
    while (!heap.empty()) {
      exact.push_back(heap.top());
      heap.pop();
    }
    std::sort(exact.begin(), exact.end(), closer);
    for (int rank = 0; rank < k; ++rank) {
      const std::size_t pos =
        static_cast<std::size_t>(query_id) * static_cast<std::size_t>(k) +
        static_cast<std::size_t>(rank);
      output.indices[pos] = exact[static_cast<std::size_t>(rank)].id;
      output.distances[pos] = exact[static_cast<std::size_t>(rank)].distance;
    }
  }
};

}  // namespace

struct NativeHNSWIndex::Impl {
  std::vector<float> train;
  int rows = 0;
  int dimensions = 0;
  DistanceMetric metric = DistanceMetric::Euclidean;
  std::unique_ptr<CompactHNSW> index;
};

NativeHNSWIndex::NativeHNSWIndex() = default;
NativeHNSWIndex::~NativeHNSWIndex() = default;
NativeHNSWIndex::NativeHNSWIndex(NativeHNSWIndex&&) noexcept = default;
NativeHNSWIndex& NativeHNSWIndex::operator=(NativeHNSWIndex&&) noexcept = default;
NativeHNSWIndex::NativeHNSWIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
bool NativeHNSWIndex::valid() const noexcept { return impl_ != nullptr && impl_->index != nullptr; }
int NativeHNSWIndex::rows() const noexcept { return impl_ == nullptr ? 0 : impl_->rows; }
int NativeHNSWIndex::dimensions() const noexcept { return impl_ == nullptr ? 0 : impl_->dimensions; }
DistanceMetric NativeHNSWIndex::metric() const noexcept {
  return impl_ == nullptr ? DistanceMetric::Euclidean : impl_->metric;
}

std::vector<float> prepare_native_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
) {
  std::vector<float> output(rows.size() * x.cols, 0.0f);
  for (std::size_t row_pos = 0; row_pos < rows.size(); ++row_pos) {
    const int row = rows[row_pos];
    if (row < 0 || static_cast<std::size_t>(row) >= x.rows) {
      throw std::out_of_range("native KNN row index is outside the input matrix.");
    }
    float squared_norm = 0.0f;
    if (metric == DistanceMetric::Cosine) {
      for (std::size_t j = 0; j < x.cols; ++j) {
        const float value = x.value_float(static_cast<std::size_t>(row), j);
        squared_norm += value * value;
      }
    }
    const float scale =
      metric == DistanceMetric::Cosine &&
      squared_norm > 0.0f &&
      std::isfinite(squared_norm) ?
      1.0f / std::sqrt(squared_norm) :
      1.0f;
    for (std::size_t j = 0; j < x.cols; ++j) {
      output[row_pos * x.cols + j] =
        x.value_float(static_cast<std::size_t>(row), j) * scale;
    }
  }
  return output;
}

std::vector<float> prepare_native_matrix(MatrixView x, DistanceMetric metric) {
  std::vector<int> rows(x.rows);
  std::iota(rows.begin(), rows.end(), 0);
  return prepare_native_matrix(x, rows, metric);
}

NativeHNSWIndex native_build_hnsw_index(
  std::vector<float> train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  const NativeHNSWParameters& parameters,
  int n_threads
) {
  if (train_rows < 1 || dimensions < 1 ||
      train.size() != static_cast<std::size_t>(train_rows) * dimensions) {
    throw std::invalid_argument("invalid native HNSW index matrix.");
  }
  auto impl = std::make_unique<NativeHNSWIndex::Impl>();
  impl->train = std::move(train);
  impl->rows = train_rows;
  impl->dimensions = dimensions;
  impl->metric = metric;
  impl->index = std::make_unique<CompactHNSW>(
    impl->train, train_rows, dimensions, metric,
    parameters.m, parameters.ef_construction, parameters.ef_search
  );
  impl->index->build(n_threads);
  return NativeHNSWIndex(std::move(impl));
}

NativeKNNResult native_hnsw_index_search(
  const NativeHNSWIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int n_threads,
  const std::vector<int>& query_train_indices
) {
  if (!index.valid()) throw std::invalid_argument("native HNSW index is empty.");
  return index.impl_->index->search(
    query, query_rows, k, n_threads, query_train_indices
  );
}

NativeKNNResult native_hnsw_index_filtered_search(
  const NativeHNSWIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int n_threads,
  const std::vector<int>& query_train_indices,
  const std::vector<int>& allowed_local_ids
) {
  if (!index.valid()) throw std::invalid_argument("native HNSW index is empty.");
  return index.impl_->index->search_filtered(
    query, query_rows, k, n_threads, query_train_indices, allowed_local_ids
  );
}

NativeKNNResult native_hnsw_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const NativeHNSWParameters& parameters,
  int n_threads,
  const std::vector<int>& query_train_indices
) {
  if (train_rows < 1 || query_rows < 0 || dimensions < 1 || k < 1) {
    throw std::invalid_argument("invalid native HNSW dimensions or neighbor count.");
  }
  if (train.size() !=
      static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("native HNSW training matrix size mismatch.");
  }
  if (query.size() !=
      static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("native HNSW query matrix size mismatch.");
  }
  if (train_rows == 1) {
    NativeKNNResult output;
    output.queries = query_rows;
    output.neighbors = query_train_indices.empty() ? 1 : 0;
    if (output.neighbors == 1) {
      output.indices.assign(static_cast<std::size_t>(query_rows), 0);
      output.distances.resize(static_cast<std::size_t>(query_rows));
      CompactHNSW singleton(train, train_rows, dimensions, metric, 2, 2, 2);
      singleton.build(n_threads);
      return singleton.search(
        query,
        query_rows,
        1,
        n_threads,
        query_train_indices
      );
    }
    return output;
  }

  CompactHNSW index(
    train,
    train_rows,
    dimensions,
    metric,
    parameters.m,
    parameters.ef_construction,
    parameters.ef_search
  );
  index.build(n_threads);
  return index.search(query, query_rows, k, n_threads, query_train_indices);
}

float native_knn_score(float internal_distance, DistanceMetric) {
  return -internal_distance;
}

float native_knn_output_distance(float internal_distance, DistanceMetric metric) {
  if (metric == DistanceMetric::Euclidean) {
    return std::sqrt(std::max(0.0f, internal_distance));
  }
  return metric == DistanceMetric::Cosine ?
    std::max(0.0f, 1.0f + internal_distance) :
    1.0f + internal_distance;
}

}  // namespace kodama::detail
