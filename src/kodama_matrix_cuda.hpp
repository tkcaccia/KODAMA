// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

struct NativeCudaIVFStats;

class CudaResidentKODAMAGraph {
 public:
  CudaResidentKODAMAGraph();
  ~CudaResidentKODAMAGraph();
  CudaResidentKODAMAGraph(CudaResidentKODAMAGraph&&) noexcept;
  CudaResidentKODAMAGraph& operator=(CudaResidentKODAMAGraph&&) noexcept;

  CudaResidentKODAMAGraph(const CudaResidentKODAMAGraph&) = delete;
  CudaResidentKODAMAGraph& operator=(const CudaResidentKODAMAGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  int lanes() const noexcept;
  bool has_landmark_index() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit CudaResidentKODAMAGraph(std::unique_ptr<Impl> impl);

  friend CudaResidentKODAMAGraph make_cuda_resident_kodama_graph(
    const NeighborGraph&,
    int,
    int,
    int
  );
  friend CudaResidentKODAMAGraph make_cuda_resident_kodama_graph_ivf(
    const std::vector<float>&,
    int,
    int,
    int,
    DistanceMetric,
    int,
    int,
    int,
    int,
    NativeCudaIVFStats*
  );
  friend void cuda_resident_prepare_results(
    CudaResidentKODAMAGraph&,
    int
  );
  friend void cuda_resident_project_landmark_labels_to_result(
    CudaResidentKODAMAGraph&,
    const std::vector<int>&,
    const std::vector<int>&,
    int,
    int,
    int,
    int
  );
  friend void cuda_resident_store_result_row(
    CudaResidentKODAMAGraph&,
    const std::vector<int>&,
    int,
    int
  );
  friend void cuda_resident_constrain_result_row(
    CudaResidentKODAMAGraph&,
    const std::vector<int>&,
    int,
    int,
    int
  );
  friend std::vector<int> cuda_resident_download_results(
    const CudaResidentKODAMAGraph&,
    int
  );
  friend std::vector<int> cuda_resident_download_result_row(
    const CudaResidentKODAMAGraph&,
    int,
    int
  );
  friend NeighborGraph cuda_resident_landmark_knn_graph(
    const CudaResidentKODAMAGraph&,
    const std::vector<float>&,
    const std::vector<int>&,
    int,
    int,
    double
  );
  friend void cuda_resident_apply_kodama_dissimilarity(
    CudaResidentKODAMAGraph&,
    int,
    bool,
    bool
  );
  friend NeighborGraph download_cuda_resident_kodama_graph(
    const CudaResidentKODAMAGraph&
  );
  friend void cuda_resident_replace_graph(
    CudaResidentKODAMAGraph&,
    const NeighborGraph&
  );
  friend void cuda_resident_reset_graph(CudaResidentKODAMAGraph&);
};

CudaResidentKODAMAGraph make_cuda_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int gpu_device,
  int lanes
);

CudaResidentKODAMAGraph make_cuda_resident_kodama_graph_ivf(
  const std::vector<float>& data,
  int samples,
  int dimensions,
  int neighbors,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  int gpu_device,
  int lanes,
  NativeCudaIVFStats* stats
);

void cuda_resident_prepare_results(
  CudaResidentKODAMAGraph& graph,
  int runs
);

void cuda_resident_project_landmark_labels_to_result(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& landmark_rows,
  const std::vector<int>& landmark_labels,
  int projection_k,
  int fallback_label,
  int run,
  int lane
);

void cuda_resident_store_result_row(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& labels,
  int run,
  int lane
);

void cuda_resident_constrain_result_row(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& constrain,
  int max_label,
  int run,
  int lane
);

std::vector<int> cuda_resident_download_results(
  const CudaResidentKODAMAGraph& graph,
  int runs
);

std::vector<int> cuda_resident_download_result_row(
  const CudaResidentKODAMAGraph& graph,
  int run,
  int lane
);

NeighborGraph cuda_resident_landmark_knn_graph(
  const CudaResidentKODAMAGraph& graph,
  const std::vector<float>& landmark_data,
  const std::vector<int>& landmark_rows,
  int k,
  int requested_nprobe,
  double target_recall
);

void cuda_resident_apply_kodama_dissimilarity(
  CudaResidentKODAMAGraph& graph,
  int runs,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph download_cuda_resident_kodama_graph(
  const CudaResidentKODAMAGraph& graph
);

void cuda_resident_replace_graph(
  CudaResidentKODAMAGraph& graph,
  const NeighborGraph& replacement
);
void cuda_resident_reset_graph(CudaResidentKODAMAGraph& graph);

void apply_kodama_dissimilarity_cuda(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int gpu_device,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph spatial_grid_self_knn_cuda(
  const std::vector<float>& data,
  int n,
  int dims,
  int neighbors,
  int gpu_device,
  bool one_based_indices = false,
  bool include_self = false
);

}  // namespace kodama::detail
