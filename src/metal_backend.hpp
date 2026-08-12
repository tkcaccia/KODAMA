// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "native_knn.hpp"

namespace kodama::detail {

bool metal_backend_available();
int metal_recommended_worker_count(std::size_t estimated_worker_bytes, int max_workers);
void metal_set_pls_residency_epoch(std::uint64_t epoch);

NativeKNNResult metal_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const std::vector<int>& query_train_indices = {}
);

NativeKNNResult metal_spatial_grid_self_knn(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int k,
  bool include_self
);

struct MetalIVFStats {
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
};

class NativeMetalKODAMAGraph;

class NativeMetalIVFIndex {
 public:
  NativeMetalIVFIndex();
  ~NativeMetalIVFIndex();
  NativeMetalIVFIndex(NativeMetalIVFIndex&&) noexcept;
  NativeMetalIVFIndex& operator=(NativeMetalIVFIndex&&) noexcept;

  NativeMetalIVFIndex(const NativeMetalIVFIndex&) = delete;
  NativeMetalIVFIndex& operator=(const NativeMetalIVFIndex&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int nlist() const noexcept;
  DistanceMetric metric() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalIVFIndex(std::unique_ptr<Impl> impl);

  friend NativeMetalIVFIndex metal_build_ivf_index(
    const std::vector<float>&,
    int,
    int,
    DistanceMetric,
    int
  );
  friend NativeKNNResult metal_ivf_index_search(
    const NativeMetalIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    MetalIVFStats*
  );
  friend NativeKNNResult metal_ivf_index_filtered_search(
    const NativeMetalIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    const std::vector<int>&,
    MetalIVFStats*
  );
  friend NativeKNNResult metal_ivf_index_self_search(
    const NativeMetalIVFIndex&,
    int,
    int,
    double,
    const std::vector<int>&,
    MetalIVFStats*
  );
  friend NativeMetalKODAMAGraph metal_build_resident_kodama_graph_ivf(
    const std::vector<float>&,
    int,
    int,
    int,
    DistanceMetric,
    int,
    int,
    int,
    MetalIVFStats*
  );
};

class NativeMetalKNNVoteGraph {
 public:
  NativeMetalKNNVoteGraph();
  ~NativeMetalKNNVoteGraph();
  NativeMetalKNNVoteGraph(NativeMetalKNNVoteGraph&&) noexcept;
  NativeMetalKNNVoteGraph& operator=(NativeMetalKNNVoteGraph&&) noexcept;

  NativeMetalKNNVoteGraph(const NativeMetalKNNVoteGraph&) = delete;
  NativeMetalKNNVoteGraph& operator=(const NativeMetalKNNVoteGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalKNNVoteGraph(std::unique_ptr<Impl> impl);

  friend NativeMetalKNNVoteGraph metal_build_knn_vote_graph(
    const std::vector<int>&,
    const std::vector<float>&,
    int,
    int
  );
  friend std::vector<int> metal_knn_vote_predict(
    const NativeMetalKNNVoteGraph&,
    const std::vector<int>&,
    int
  );
  friend void metal_knn_vote_predict_into(
    const NativeMetalKNNVoteGraph&,
    const std::vector<int>&,
    int,
    std::vector<int>&
  );
};

class NativeMetalKODAMAGraph {
 public:
  NativeMetalKODAMAGraph();
  ~NativeMetalKODAMAGraph();
  NativeMetalKODAMAGraph(NativeMetalKODAMAGraph&&) noexcept;
  NativeMetalKODAMAGraph& operator=(NativeMetalKODAMAGraph&&) noexcept;

  NativeMetalKODAMAGraph(const NativeMetalKODAMAGraph&) = delete;
  NativeMetalKODAMAGraph& operator=(const NativeMetalKODAMAGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  int lanes() const noexcept;
  bool has_landmark_index() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalKODAMAGraph(std::unique_ptr<Impl> impl);

  friend NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
    const NeighborGraph&,
    int,
    int
  );
  friend NativeMetalKODAMAGraph metal_build_resident_kodama_graph_ivf(
    const std::vector<float>&,
    int,
    int,
    int,
    DistanceMetric,
    int,
    int,
    int,
    MetalIVFStats*
  );
  friend void metal_prepare_resident_results(NativeMetalKODAMAGraph&, int);
  friend void metal_project_landmark_labels_to_result(
    NativeMetalKODAMAGraph&,
    const std::vector<int>&,
    const std::vector<int>&,
    int,
    int,
    int,
    int
  );
  friend void metal_store_resident_result_row(
    NativeMetalKODAMAGraph&,
    const std::vector<int>&,
    int,
    int
  );
  friend void metal_constrain_resident_result_row(
    NativeMetalKODAMAGraph&,
    const std::vector<int>&,
    int,
    int,
    int
  );
  friend std::vector<int> metal_download_resident_results(
    const NativeMetalKODAMAGraph&,
    int
  );
  friend std::vector<int> metal_download_resident_result_row(
    const NativeMetalKODAMAGraph&,
    int,
    int
  );
  friend NeighborGraph metal_resident_landmark_knn_graph(
    const NativeMetalKODAMAGraph&,
    const std::vector<float>&,
    const std::vector<int>&,
    int,
    int,
    double
  );
  friend void metal_apply_resident_kodama_dissimilarity(
    NativeMetalKODAMAGraph&,
    int,
    bool,
    bool
  );
  friend NeighborGraph metal_download_resident_kodama_graph(
    const NativeMetalKODAMAGraph&
  );
  friend void metal_replace_resident_kodama_graph(
    NativeMetalKODAMAGraph&,
    const NeighborGraph&
  );
  friend void metal_reset_resident_kodama_graph(NativeMetalKODAMAGraph&);
};

class NativeMetalKMeansContext {
 public:
  NativeMetalKMeansContext();
  ~NativeMetalKMeansContext();
  NativeMetalKMeansContext(NativeMetalKMeansContext&&) noexcept;
  NativeMetalKMeansContext& operator=(NativeMetalKMeansContext&&) noexcept;

  NativeMetalKMeansContext(const NativeMetalKMeansContext&) = delete;
  NativeMetalKMeansContext& operator=(const NativeMetalKMeansContext&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int lanes() const noexcept;
  std::uint64_t input_uploads() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalKMeansContext(std::unique_ptr<Impl> impl);

  friend NativeMetalKMeansContext metal_build_kmeans_context(
    const std::vector<float>&, int, int, int, int
  );
  friend std::vector<int> metal_kmeans_context_labels(
    NativeMetalKMeansContext&, int, int, const std::vector<int>&, int
  );
};

NativeKNNResult metal_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

NativeMetalIVFIndex metal_build_ivf_index(
  const std::vector<float>& train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  int requested_nlist
);

NativeKNNResult metal_ivf_index_search(
  const NativeMetalIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

NativeKNNResult metal_ivf_index_filtered_search(
  const NativeMetalIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  const std::vector<int>& allowed_local_ids,
  MetalIVFStats* stats = nullptr
);

NativeKNNResult metal_ivf_index_self_search(
  const NativeMetalIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

std::vector<int> metal_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  const std::vector<int>& initial_point_indices,
  int max_iterations
);

NativeMetalKMeansContext metal_build_kmeans_context(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int lanes,
  int max_clusters
);

std::vector<int> metal_kmeans_context_labels(
  NativeMetalKMeansContext& context,
  int lane,
  int clusters,
  const std::vector<int>& initial_point_indices,
  int max_iterations
);

NativeMetalKNNVoteGraph metal_build_knn_vote_graph(
  const std::vector<int>& neighbor_rows,
  const std::vector<float>& scores,
  int samples,
  int neighbors
);

std::vector<int> metal_knn_vote_predict(
  const NativeMetalKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label
);

void metal_knn_vote_predict_into(
  const NativeMetalKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label,
  std::vector<int>& predictions
);

NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int lanes
);

NativeMetalKODAMAGraph metal_build_resident_kodama_graph_ivf(
  const std::vector<float>& data,
  int samples,
  int dimensions,
  int neighbors,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  int lanes,
  MetalIVFStats* stats
);

NeighborGraph metal_resident_landmark_knn_graph(
  const NativeMetalKODAMAGraph& graph,
  const std::vector<float>& landmark_data,
  const std::vector<int>& landmark_rows,
  int k,
  int requested_nprobe,
  double target_recall
);

void metal_prepare_resident_results(NativeMetalKODAMAGraph& graph, int runs);

void metal_project_landmark_labels_to_result(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& landmark_rows,
  const std::vector<int>& landmark_labels,
  int projection_k,
  int fallback_label,
  int run,
  int lane
);

void metal_store_resident_result_row(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& labels,
  int run,
  int lane
);

void metal_constrain_resident_result_row(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& constrain,
  int max_label,
  int run,
  int lane
);

std::vector<int> metal_download_resident_results(
  const NativeMetalKODAMAGraph& graph,
  int runs
);

std::vector<int> metal_download_resident_result_row(
  const NativeMetalKODAMAGraph& graph,
  int run,
  int lane
);

void metal_apply_resident_kodama_dissimilarity(
  NativeMetalKODAMAGraph& graph,
  int runs,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph metal_download_resident_kodama_graph(
  const NativeMetalKODAMAGraph& graph
);

void metal_replace_resident_kodama_graph(
  NativeMetalKODAMAGraph& graph,
  const NeighborGraph& replacement
);
void metal_reset_resident_kodama_graph(NativeMetalKODAMAGraph& graph);

std::vector<float> metal_matrix_multiply(
  const std::vector<float>& left,
  int left_rows,
  int left_cols,
  const std::vector<float>& right,
  int right_rows,
  int right_cols,
  bool transpose_left = false,
  bool transpose_right = false,
  bool reuse_left = false
);

struct MetalPLSScoreStatistics {
  int classes = 0;
  int components = 0;
  std::vector<float> class_sums;
  std::vector<float> score_crossprod;
};

MetalPLSScoreStatistics metal_pls_score_statistics(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& weights,
  int components,
  const std::vector<int>& encoded_labels,
  int classes
);

std::vector<int> metal_pls_lda_predict(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& weights,
  int components,
  const std::vector<float>& linear,
  const std::vector<float>& constants,
  const std::vector<int>& class_labels
);

std::vector<float> metal_umap_optimize(
  const std::vector<int>& neighbors,
  const std::vector<float>& weights,
  const std::vector<float>& initialization,
  int samples,
  int width,
  int epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float max_weight,
  float repulsion_strength,
  std::uint32_t seed
);

std::vector<float> metal_opentsne_optimize(
  const std::vector<int>& row_ptr,
  const std::vector<int>& columns,
  const std::vector<float>& probabilities,
  const std::vector<float>& initialization,
  int samples,
  int early_exaggeration_iter,
  int normal_iter,
  float early_exaggeration,
  float exaggeration,
  float learning_rate,
  bool learning_rate_auto,
  float initial_momentum,
  float final_momentum,
  float min_gain,
  float max_step_norm,
  std::uint32_t seed
);

struct MetalSIMPLSResult {
  int predictors = 0;
  int responses = 0;
  int components = 0;
  std::vector<float> weights;
  std::vector<float> y_weights;
};

MetalSIMPLSResult metal_simpls_fit(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& cross_product,
  int responses,
  int max_components
);

}  // namespace kodama::detail
