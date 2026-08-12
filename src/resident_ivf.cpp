// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common.hpp"
#include "metal_backend.hpp"
#include "native_knn.hpp"

#if defined(KODAMA_ENABLE_CUDA)
#include "native_cuda_backend.hpp"
#endif

namespace kodama {

struct ResidentIVFIndex::Impl {
  Backend backend = Backend::CPU;
  DistanceMetric metric = DistanceMetric::Euclidean;
  int rows = 0;
  int dimensions = 0;
  int nlist = 0;
  int requested_nprobe = 0;
  int gpu_device = 0;
  double target_recall = 0.99;
  double build_seconds = 0.0;
#if defined(KODAMA_ENABLE_CUDA)
  std::unique_ptr<detail::NativeCudaIVFIndex> cuda;
#endif
#if defined(KODAMA_ENABLE_METAL)
  std::unique_ptr<detail::NativeMetalIVFIndex> metal;
#endif
};

namespace {

Backend resolve_resident_backend(Backend backend, int gpu_device) {
  if (backend != Backend::Auto) return backend;
#if !defined(KODAMA_ENABLE_CUDA)
  (void)gpu_device;
#endif
#if defined(KODAMA_ENABLE_CUDA)
  if (detail::native_cuda_backend_available(gpu_device)) return Backend::CUDA;
#endif
#if defined(KODAMA_ENABLE_METAL)
  if (detail::metal_backend_available()) return Backend::Metal;
#endif
  return Backend::CPU;
}

NeighborGraph public_graph(
  const detail::NativeKNNResult& native,
  DistanceMetric metric
) {
  NeighborGraph graph;
  graph.neighbors = native.neighbors;
  graph.index_base = GraphIndexBase::One;
  graph.indices = native.indices;
  graph.distances.resize(native.distances.size());
  for (std::size_t i = 0; i < graph.indices.size(); ++i) {
    if (graph.indices[i] >= 0) ++graph.indices[i];
    graph.distances[i] = detail::native_knn_output_distance(native.distances[i], metric);
  }
  return graph;
}

#if defined(KODAMA_ENABLE_CUDA) || defined(KODAMA_ENABLE_METAL)
void set_stats(
  ResidentIVFSearchStats* output,
  Backend backend,
  int nlist,
  int nprobe,
  double pilot_recall,
  double seconds
) {
  if (output == nullptr) return;
  output->backend = backend;
  output->nlist = nlist;
  output->nprobe = nprobe;
  output->pilot_recall = pilot_recall;
  output->search_seconds = seconds;
}
#endif

}  // namespace

ResidentIVFIndex::ResidentIVFIndex() = default;
ResidentIVFIndex::~ResidentIVFIndex() = default;
ResidentIVFIndex::ResidentIVFIndex(ResidentIVFIndex&&) noexcept = default;
ResidentIVFIndex& ResidentIVFIndex::operator=(ResidentIVFIndex&&) noexcept = default;
ResidentIVFIndex::ResidentIVFIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool ResidentIVFIndex::valid() const noexcept { return impl_ != nullptr; }
Backend ResidentIVFIndex::backend() const noexcept {
  return impl_ == nullptr ? Backend::CPU : impl_->backend;
}
DistanceMetric ResidentIVFIndex::metric() const noexcept {
  return impl_ == nullptr ? DistanceMetric::Euclidean : impl_->metric;
}
int ResidentIVFIndex::rows() const noexcept { return impl_ == nullptr ? 0 : impl_->rows; }
int ResidentIVFIndex::dimensions() const noexcept {
  return impl_ == nullptr ? 0 : impl_->dimensions;
}
int ResidentIVFIndex::nlist() const noexcept { return impl_ == nullptr ? 0 : impl_->nlist; }
double ResidentIVFIndex::build_seconds() const noexcept {
  return impl_ == nullptr ? 0.0 : impl_->build_seconds;
}

ResidentIVFIndex BuildResidentIVFIndex(
  MatrixView train,
  const KNNOptions& options
) {
  if (train.data == nullptr || train.rows < 1 || train.cols < 1) {
    throw std::invalid_argument("BuildResidentIVFIndex requires a non-empty matrix.");
  }
  const Backend backend = resolve_resident_backend(options.backend, options.gpu_device);
  if (backend != Backend::CUDA && backend != Backend::Metal) {
    throw std::invalid_argument(
      "BuildResidentIVFIndex requires backend CUDA or Metal."
    );
  }
  detail::Timer timer;
  std::vector<float> prepared = detail::prepare_native_matrix(train, options.metric);
  auto impl = std::make_unique<ResidentIVFIndex::Impl>();
  impl->backend = backend;
  impl->metric = options.metric;
  impl->rows = static_cast<int>(train.rows);
  impl->dimensions = static_cast<int>(train.cols);
  impl->requested_nprobe = options.ivf_nprobe;
  impl->gpu_device = options.gpu_device;
  impl->target_recall = std::max(0.0, std::min(1.0, options.hnsw_target_recall));

  if (backend == Backend::CUDA) {
#if defined(KODAMA_ENABLE_CUDA)
    impl->cuda = std::make_unique<detail::NativeCudaIVFIndex>(
      detail::native_cuda_build_ivf_index(
        prepared,
        impl->rows,
        impl->dimensions,
        impl->metric,
        options.ivf_nlist,
        impl->gpu_device
      )
    );
    impl->nlist = impl->cuda->nlist();
#else
    throw std::runtime_error("The CUDA backend is not available in this build.");
#endif
  } else {
#if defined(KODAMA_ENABLE_METAL)
    impl->metal = std::make_unique<detail::NativeMetalIVFIndex>(
      detail::metal_build_ivf_index(
        prepared,
        impl->rows,
        impl->dimensions,
        impl->metric,
        options.ivf_nlist
      )
    );
    impl->nlist = impl->metal->nlist();
#else
    throw std::runtime_error("The Metal backend is not available in this build.");
#endif
  }
  impl->build_seconds = timer.seconds();
  return ResidentIVFIndex(std::move(impl));
}

NeighborGraph SearchResidentIVFIndex(
  const ResidentIVFIndex& index,
  MatrixView query,
  int k,
  ResidentIVFSearchStats* stats
) {
  (void)stats;
  if (!index.valid()) throw std::invalid_argument("Resident IVF index is empty.");
  if (query.data == nullptr || query.rows < 1 || query.cols < 1) {
    throw std::invalid_argument("Resident IVF search requires a non-empty query matrix.");
  }
  if (k < 1) throw std::invalid_argument("Resident IVF search requires k >= 1.");
  const ResidentIVFIndex::Impl& impl = *index.impl_;
  if (query.cols != static_cast<std::size_t>(impl.dimensions)) {
    throw std::invalid_argument("Resident IVF query dimensions do not match the index.");
  }
  detail::Timer timer;
  std::vector<float> prepared = detail::prepare_native_matrix(query, impl.metric);
  detail::NativeKNNResult native;
  if (impl.backend == Backend::CUDA) {
#if defined(KODAMA_ENABLE_CUDA)
    detail::NativeCudaIVFStats native_stats;
    native = detail::native_cuda_ivf_index_search(
      *impl.cuda,
      prepared,
      static_cast<int>(query.rows),
      k,
      impl.requested_nprobe,
      impl.target_recall,
      {},
      &native_stats
    );
    set_stats(
      stats,
      impl.backend,
      native_stats.nlist,
      native_stats.nprobe,
      native_stats.pilot_recall,
      timer.seconds()
    );
#else
    throw std::runtime_error("The CUDA backend is not available in this build.");
#endif
  } else {
#if defined(KODAMA_ENABLE_METAL)
    detail::MetalIVFStats native_stats;
    native = detail::metal_ivf_index_search(
      *impl.metal,
      prepared,
      static_cast<int>(query.rows),
      k,
      impl.requested_nprobe,
      impl.target_recall,
      {},
      &native_stats
    );
    set_stats(
      stats,
      impl.backend,
      native_stats.nlist,
      native_stats.nprobe,
      native_stats.pilot_recall,
      timer.seconds()
    );
#else
    throw std::runtime_error("The Metal backend is not available in this build.");
#endif
  }
  return public_graph(native, impl.metric);
}

NeighborGraph SearchResidentIVFIndexSelf(
  const ResidentIVFIndex& index,
  int k,
  bool exclude_self,
  ResidentIVFSearchStats* stats
) {
  (void)stats;
  if (!index.valid()) throw std::invalid_argument("Resident IVF index is empty.");
  if (k < 1) throw std::invalid_argument("Resident IVF self-search requires k >= 1.");
  const ResidentIVFIndex::Impl& impl = *index.impl_;
  std::vector<int> exclusions;
  if (exclude_self) {
    exclusions.resize(static_cast<std::size_t>(impl.rows));
    std::iota(exclusions.begin(), exclusions.end(), 0);
  }
  detail::Timer timer;
  detail::NativeKNNResult native;
  if (impl.backend == Backend::CUDA) {
#if defined(KODAMA_ENABLE_CUDA)
    detail::NativeCudaIVFStats native_stats;
    native = detail::native_cuda_ivf_index_self_search(
      *impl.cuda,
      k,
      impl.requested_nprobe,
      impl.target_recall,
      exclusions,
      &native_stats
    );
    set_stats(
      stats,
      impl.backend,
      native_stats.nlist,
      native_stats.nprobe,
      native_stats.pilot_recall,
      timer.seconds()
    );
#else
    throw std::runtime_error("The CUDA backend is not available in this build.");
#endif
  } else {
#if defined(KODAMA_ENABLE_METAL)
    detail::MetalIVFStats native_stats;
    native = detail::metal_ivf_index_self_search(
      *impl.metal,
      k,
      impl.requested_nprobe,
      impl.target_recall,
      exclusions,
      &native_stats
    );
    set_stats(
      stats,
      impl.backend,
      native_stats.nlist,
      native_stats.nprobe,
      native_stats.pilot_recall,
      timer.seconds()
    );
#else
    throw std::runtime_error("The Metal backend is not available in this build.");
#endif
  }
  return public_graph(native, impl.metric);
}

}  // namespace kodama
