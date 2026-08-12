// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "metal_backend.hpp"

#include <stdexcept>
#include <utility>

namespace kodama::detail {

struct NativeMetalIVFIndex::Impl {};
struct NativeMetalKODAMAGraph::Impl {};

NativeMetalIVFIndex::NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::~NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex& NativeMetalIVFIndex::operator=(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool NativeMetalIVFIndex::valid() const noexcept { return false; }
int NativeMetalIVFIndex::rows() const noexcept { return 0; }
int NativeMetalIVFIndex::dimensions() const noexcept { return 0; }
int NativeMetalIVFIndex::nlist() const noexcept { return 0; }
DistanceMetric NativeMetalIVFIndex::metric() const noexcept {
  return DistanceMetric::Euclidean;
}

NativeMetalKODAMAGraph::NativeMetalKODAMAGraph() = default;
NativeMetalKODAMAGraph::~NativeMetalKODAMAGraph() = default;
NativeMetalKODAMAGraph::NativeMetalKODAMAGraph(
  NativeMetalKODAMAGraph&&
) noexcept = default;
NativeMetalKODAMAGraph& NativeMetalKODAMAGraph::operator=(
  NativeMetalKODAMAGraph&&
) noexcept = default;
NativeMetalKODAMAGraph::NativeMetalKODAMAGraph(
  std::unique_ptr<Impl> impl
) : impl_(std::move(impl)) {}

bool NativeMetalKODAMAGraph::valid() const noexcept { return false; }
int NativeMetalKODAMAGraph::samples() const noexcept { return 0; }
int NativeMetalKODAMAGraph::neighbors() const noexcept { return 0; }
int NativeMetalKODAMAGraph::lanes() const noexcept { return 0; }

bool metal_backend_available() {
  return false;
}

std::vector<float> metal_umap_optimize(
  const std::vector<int>&,
  const std::vector<float>&,
  const std::vector<float>&,
  int,
  int,
  int,
  int,
  float,
  float,
  float,
  float,
  float,
  std::uint32_t
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<float> metal_opentsne_optimize(
  const std::vector<int>&,
  const std::vector<int>&,
  const std::vector<float>&,
  const std::vector<float>&,
  int,
  int,
  int,
  float,
  float,
  float,
  bool,
  float,
  float,
  float,
  float,
  std::uint32_t
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

int metal_recommended_worker_count(std::size_t, int) {
  return 1;
}

void metal_set_pls_residency_epoch(std::uint64_t) {}

NativeKNNResult metal_exact_knn_search(
  const std::vector<float>&,
  int,
  const std::vector<float>&,
  int,
  int,
  int,
  DistanceMetric,
  const std::vector<int>&
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_spatial_grid_self_knn(
  const std::vector<float>&,
  int,
  int,
  int,
  bool
) {
  throw std::runtime_error("Metal spatial grid KNN is unavailable in this build.");
}

NativeKNNResult metal_ivf_knn_search(
  const std::vector<float>&,
  int,
  const std::vector<float>&,
  int,
  int,
  int,
  DistanceMetric,
  int,
  int,
  const std::vector<int>&,
  MetalIVFStats*
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeMetalIVFIndex metal_build_ivf_index(
  const std::vector<float>&,
  int,
  int,
  DistanceMetric,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_ivf_index_search(
  const NativeMetalIVFIndex&,
  const std::vector<float>&,
  int,
  int,
  int,
  double,
  const std::vector<int>&,
  MetalIVFStats*
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_ivf_index_self_search(
  const NativeMetalIVFIndex&,
  int,
  int,
  double,
  const std::vector<int>&,
  MetalIVFStats*
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<int> metal_kmeans_labels(
  const std::vector<float>&,
  int,
  int,
  int,
  const std::vector<int>&,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
  const NeighborGraph&,
  int,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeMetalKODAMAGraph metal_build_resident_kodama_graph_ivf(
  const std::vector<float>&,
  int,
  int,
  int,
  DistanceMetric,
  int,
  int,
  int,
  MetalIVFStats*
) {
  throw std::runtime_error("Metal backend is unavailable in this build.");
}

void metal_prepare_resident_results(NativeMetalKODAMAGraph&, int) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_project_landmark_labels_to_result(
  NativeMetalKODAMAGraph&,
  const std::vector<int>&,
  const std::vector<int>&,
  int,
  int,
  int,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_store_resident_result_row(
  NativeMetalKODAMAGraph&, const std::vector<int>&, int, int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_constrain_resident_result_row(
  NativeMetalKODAMAGraph&, const std::vector<int>&, int, int, int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<int> metal_download_resident_results(
  const NativeMetalKODAMAGraph&, int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<int> metal_download_resident_result_row(
  const NativeMetalKODAMAGraph&, int, int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_apply_resident_kodama_dissimilarity(
  NativeMetalKODAMAGraph&,
  int,
  bool,
  bool
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NeighborGraph metal_download_resident_kodama_graph(
  const NativeMetalKODAMAGraph&
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_replace_resident_kodama_graph(
  NativeMetalKODAMAGraph&, const NeighborGraph&
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

void metal_reset_resident_kodama_graph(NativeMetalKODAMAGraph&) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<float> metal_matrix_multiply(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  int,
  bool,
  bool,
  bool
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

MetalPLSScoreStatistics metal_pls_score_statistics(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  const std::vector<int>&,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<int> metal_pls_lda_predict(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  const std::vector<float>&,
  const std::vector<float>&,
  const std::vector<int>&
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

MetalSIMPLSResult metal_simpls_fit(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

}  // namespace kodama::detail
