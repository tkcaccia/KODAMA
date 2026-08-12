// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "native_knn.hpp"

namespace kodama::detail {

struct NativeCudaIVFStats {
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
};

class NativeCudaIVFIndex {
 public:
  NativeCudaIVFIndex();
  ~NativeCudaIVFIndex();
  NativeCudaIVFIndex(NativeCudaIVFIndex&&) noexcept;
  NativeCudaIVFIndex& operator=(NativeCudaIVFIndex&&) noexcept;

  NativeCudaIVFIndex(const NativeCudaIVFIndex&) = delete;
  NativeCudaIVFIndex& operator=(const NativeCudaIVFIndex&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int nlist() const noexcept;
  int device() const noexcept;
  DistanceMetric metric() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeCudaIVFIndex(std::unique_ptr<Impl> impl);

  friend NativeCudaIVFIndex native_cuda_build_ivf_index(
    const std::vector<float>&,
    int,
    int,
    DistanceMetric,
    int,
    int
  );
  friend NativeKNNResult native_cuda_ivf_index_search(
    const NativeCudaIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    NativeCudaIVFStats*
  );
  friend NativeKNNResult native_cuda_ivf_index_filtered_search(
    const NativeCudaIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    const std::vector<int>&,
    NativeCudaIVFStats*
  );
  friend NativeKNNResult native_cuda_ivf_index_self_search(
    const NativeCudaIVFIndex&,
    int,
    int,
    double,
    const std::vector<int>&,
    NativeCudaIVFStats*
  );
  friend void native_cuda_ivf_index_self_search_device(
    const NativeCudaIVFIndex&,
    int,
    int,
    double,
    const std::vector<int>&,
    int*,
    float*,
    NativeCudaIVFStats*
  );
};

class NativeCudaKNNVoteGraph {
 public:
  NativeCudaKNNVoteGraph();
  ~NativeCudaKNNVoteGraph();
  NativeCudaKNNVoteGraph(NativeCudaKNNVoteGraph&&) noexcept;
  NativeCudaKNNVoteGraph& operator=(NativeCudaKNNVoteGraph&&) noexcept;

  NativeCudaKNNVoteGraph(const NativeCudaKNNVoteGraph&) = delete;
  NativeCudaKNNVoteGraph& operator=(const NativeCudaKNNVoteGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  int device() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeCudaKNNVoteGraph(std::unique_ptr<Impl> impl);

  friend NativeCudaKNNVoteGraph native_cuda_build_knn_vote_graph(
    const std::vector<int>&,
    const std::vector<float>&,
    int,
    int,
    int
  );
  friend std::vector<int> native_cuda_knn_vote_predict(
    const NativeCudaKNNVoteGraph&,
    const std::vector<int>&,
    int
  );
  friend void native_cuda_knn_vote_predict_into(
    const NativeCudaKNNVoteGraph&,
    const std::vector<int>&,
    int,
    std::vector<int>&
  );
};

class NativeCudaKMeansContext {
 public:
  NativeCudaKMeansContext();
  ~NativeCudaKMeansContext();
  NativeCudaKMeansContext(NativeCudaKMeansContext&&) noexcept;
  NativeCudaKMeansContext& operator=(NativeCudaKMeansContext&&) noexcept;

  NativeCudaKMeansContext(const NativeCudaKMeansContext&) = delete;
  NativeCudaKMeansContext& operator=(const NativeCudaKMeansContext&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int lanes() const noexcept;
  std::uint64_t input_uploads() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeCudaKMeansContext(std::unique_ptr<Impl> impl);

  friend NativeCudaKMeansContext native_cuda_build_kmeans_context(
    const std::vector<float>&, int, int, int, int, int
  );
  friend std::vector<int> native_cuda_kmeans_context_labels(
    NativeCudaKMeansContext&, int, int, int, std::uint64_t
  );
};

bool native_cuda_backend_available(int device = 0);

NativeKNNResult native_cuda_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int device,
  const std::vector<int>& query_train_indices = {}
);

NativeKNNResult native_cuda_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  double target_recall,
  int device,
  const std::vector<int>& query_train_indices = {},
  NativeCudaIVFStats* stats = nullptr
);

NativeCudaIVFIndex native_cuda_build_ivf_index(
  const std::vector<float>& train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  int requested_nlist,
  int device
);

NativeKNNResult native_cuda_ivf_index_search(
  const NativeCudaIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  NativeCudaIVFStats* stats = nullptr
);

// Queries a resident full-data index while retaining only rows selected by
// allowed_local_ids. Non-negative entries are returned as landmark-local IDs;
// negative entries are excluded from both the exact pilot and IVF candidates.
NativeKNNResult native_cuda_ivf_index_filtered_search(
  const NativeCudaIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  const std::vector<int>& allowed_local_ids,
  NativeCudaIVFStats* stats = nullptr
);

NativeKNNResult native_cuda_ivf_index_self_search(
  const NativeCudaIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  NativeCudaIVFStats* stats = nullptr
);

// Executes self-search into caller-owned device buffers. Distances use the
// native internal representation (squared L2 or negative inner product).
void native_cuda_ivf_index_self_search_device(
  const NativeCudaIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  int* device_output_ids,
  float* device_output_distances,
  NativeCudaIVFStats* stats = nullptr
);

std::vector<int> native_cuda_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  int max_iterations,
  std::uint64_t seed,
  int device
);

NativeCudaKMeansContext native_cuda_build_kmeans_context(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int lanes,
  int max_clusters,
  int device
);

std::vector<int> native_cuda_kmeans_context_labels(
  NativeCudaKMeansContext& context,
  int lane,
  int clusters,
  int max_iterations,
  std::uint64_t seed
);

NativeCudaKNNVoteGraph native_cuda_build_knn_vote_graph(
  const std::vector<int>& neighbor_rows,
  const std::vector<float>& scores,
  int samples,
  int neighbors,
  int device
);

std::vector<int> native_cuda_knn_vote_predict(
  const NativeCudaKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label
);

void native_cuda_knn_vote_predict_into(
  const NativeCudaKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label,
  std::vector<int>& predictions
);

}  // namespace kodama::detail
