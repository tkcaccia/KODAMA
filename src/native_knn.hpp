// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

struct NativeHNSWParameters {
  int m = 16;
  int ef_construction = 80;
  int ef_search = 64;
};

struct NativeKNNResult {
  int queries = 0;
  int neighbors = 0;
  std::vector<int> indices;
  // Internal smaller-is-better values: squared L2 or negative inner product.
  std::vector<float> distances;
};

class NativeHNSWIndex {
 public:
  NativeHNSWIndex();
  ~NativeHNSWIndex();
  NativeHNSWIndex(NativeHNSWIndex&&) noexcept;
  NativeHNSWIndex& operator=(NativeHNSWIndex&&) noexcept;
  NativeHNSWIndex(const NativeHNSWIndex&) = delete;
  NativeHNSWIndex& operator=(const NativeHNSWIndex&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  DistanceMetric metric() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit NativeHNSWIndex(std::unique_ptr<Impl> impl);

  friend NativeHNSWIndex native_build_hnsw_index(
    std::vector<float>, int, int, DistanceMetric,
    const NativeHNSWParameters&, int
  );
  friend NativeKNNResult native_hnsw_index_search(
    const NativeHNSWIndex&, const std::vector<float>&, int, int, int,
    const std::vector<int>&
  );
  friend NativeKNNResult native_hnsw_index_filtered_search(
    const NativeHNSWIndex&, const std::vector<float>&, int, int, int,
    const std::vector<int>&, const std::vector<int>&
  );
};

std::vector<float> prepare_native_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
);

std::vector<float> prepare_native_matrix(MatrixView x, DistanceMetric metric);

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
  const std::vector<int>& query_train_indices = {}
);

NativeHNSWIndex native_build_hnsw_index(
  std::vector<float> train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  const NativeHNSWParameters& parameters,
  int n_threads
);

NativeKNNResult native_hnsw_index_search(
  const NativeHNSWIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int n_threads,
  const std::vector<int>& query_train_indices = {}
);

NativeKNNResult native_hnsw_index_filtered_search(
  const NativeHNSWIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int n_threads,
  const std::vector<int>& query_train_indices,
  const std::vector<int>& allowed_local_ids
);

float native_knn_score(float internal_distance, DistanceMetric metric);
float native_knn_output_distance(float internal_distance, DistanceMetric metric);

}  // namespace kodama::detail
