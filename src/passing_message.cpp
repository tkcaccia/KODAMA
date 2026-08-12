// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT
// Reimplements the maintainer-authored KODAMAextra passing.message formula.

#include "kodama/kodama.hpp"

#include "metal_backend.hpp"
#include "native_knn.hpp"
#include "spatial_grid_knn.hpp"
#if defined(KODAMA_ENABLE_CUDA)
#include "kodama_matrix_cuda.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;

struct SampleGroup {
  int label = 0;
  std::vector<int> rows;
};

template <class Function>
void parallel_for(const std::size_t count, const int requested_threads, Function function) {
  const std::size_t workers = std::min<std::size_t>(
    count, static_cast<std::size_t>(std::max(1, requested_threads))
  );
  if (workers <= 1 || count < 2) {
    for (std::size_t i = 0; i < count; ++i) function(i);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([=, &function]() {
      for (std::size_t i = worker; i < count; i += workers) function(i);
    });
  }
  for (auto& thread : threads) thread.join();
}

void validate_inputs(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options
) {
  if (data.data == nullptr || data.rows == 0 || data.cols == 0) {
    throw std::invalid_argument("The expression matrix must be non-empty.");
  }
  if (spatial.data == nullptr || spatial.rows != data.rows ||
      (spatial.cols != 2 && spatial.cols != 3)) {
    throw std::invalid_argument(
      "spatial must have the same rows as data and contain two or three coordinates."
    );
  }
  if (!samples.empty() && samples.size() != data.rows) {
    throw std::invalid_argument("samples must be empty or contain one value per row.");
  }
  if (options.neighbors < 1) {
    throw std::invalid_argument("neighbors must be positive.");
  }
  for (std::size_t row = 0; row < spatial.rows; ++row) {
    for (std::size_t column = 0; column < spatial.cols; ++column) {
      if (!std::isfinite(spatial.value_float(row, column))) {
        throw std::invalid_argument("spatial coordinates must be finite.");
      }
    }
  }
}

std::vector<SampleGroup> make_sample_groups(
  const std::size_t rows,
  const std::vector<int>& samples
) {
  if (samples.empty()) {
    SampleGroup group;
    group.rows.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) group.rows[row] = static_cast<int>(row);
    return {std::move(group)};
  }
  std::vector<SampleGroup> groups;
  std::unordered_map<int, std::size_t> positions;
  positions.reserve(samples.size());
  for (std::size_t row = 0; row < samples.size(); ++row) {
    const int label = samples[row];
    const auto inserted = positions.emplace(label, groups.size());
    if (inserted.second) groups.push_back(SampleGroup{label, {}});
    groups[inserted.first->second].rows.push_back(static_cast<int>(row));
  }
  return groups;
}

NeighborGraph build_grid_graph(
  const std::vector<float>& coordinates,
  const int rows,
  const int dimensions,
  const PassingMessageOptions& options,
  const Backend backend
) {
  if (backend == Backend::CPU) {
    return detail::spatial_grid_self_knn(
      coordinates.data(), rows, dimensions, options.neighbors,
      options.n_threads, false, true
    );
  }
  if (backend == Backend::CUDA) {
#if defined(KODAMA_ENABLE_CUDA)
    return detail::spatial_grid_self_knn_cuda(
      coordinates, rows, dimensions, options.neighbors,
      options.gpu_device, false, true
    );
#else
    throw std::runtime_error("PassingMessage_CUDA requires a CUDA-enabled build.");
#endif
  }
  if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
    const detail::NativeKNNResult search = detail::metal_spatial_grid_self_knn(
      coordinates, rows, dimensions, options.neighbors, true
    );
    NeighborGraph graph;
    graph.neighbors = options.neighbors;
    graph.indices = search.indices;
    graph.distances.resize(search.distances.size());
    for (std::size_t i = 0; i < search.distances.size(); ++i) {
      graph.distances[i] = detail::native_knn_output_distance(
        search.distances[i], DistanceMetric::Euclidean
      );
    }
    return graph;
#else
    throw std::runtime_error("PassingMessage_METAL requires a Metal-enabled build.");
#endif
  }
  throw std::invalid_argument("PassingMessage requires an explicit CPU, CUDA, or Metal backend.");
}

PassingMessageResult passing_message_impl(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options,
  const Backend backend
) {
  validate_inputs(data, spatial, samples, options);
  const auto started = Clock::now();
  const auto groups = make_sample_groups(data.rows, samples);

  PassingMessageResult result;
  result.values.assign(data.rows * data.cols, 0.0f);
  result.sample_max_distances.reserve(groups.size());
  result.samples = data.rows;
  result.variables = data.cols;
  result.sample_groups = groups.size();
  result.neighbors = options.neighbors;
  result.backend = backend;

  for (const SampleGroup& group : groups) {
    if (group.rows.size() < static_cast<std::size_t>(options.neighbors)) {
      throw std::invalid_argument(
        "Each sample/slide must contain at least neighbors rows."
      );
    }
    std::vector<float> coordinates(group.rows.size() * spatial.cols);
    for (std::size_t local_row = 0; local_row < group.rows.size(); ++local_row) {
      const std::size_t global_row = static_cast<std::size_t>(group.rows[local_row]);
      for (std::size_t column = 0; column < spatial.cols; ++column) {
        coordinates[local_row * spatial.cols + column] = spatial.value_float(global_row, column);
      }
    }

    const auto graph_started = Clock::now();
    const NeighborGraph graph = build_grid_graph(
      coordinates, static_cast<int>(group.rows.size()), static_cast<int>(spatial.cols),
      options, backend
    );
    result.graph_seconds += std::chrono::duration<double>(Clock::now() - graph_started).count();
    const float max_distance = graph.distances.empty() ? 0.0f :
      *std::max_element(graph.distances.begin(), graph.distances.end());
    result.sample_max_distances.push_back(max_distance);
    for (const int index : graph.indices) {
      if (index < 0 || index >= static_cast<int>(group.rows.size())) {
        throw std::runtime_error("The spatial grid returned an invalid neighbor index.");
      }
    }

    const auto aggregation_started = Clock::now();
    parallel_for(group.rows.size(), options.n_threads, [&](const std::size_t local_row) {
      const std::size_t output_row = static_cast<std::size_t>(group.rows[local_row]);
      float* output = result.values.data() + output_row * data.cols;
      for (int neighbor = 0; neighbor < options.neighbors; ++neighbor) {
        const std::size_t edge = local_row * static_cast<std::size_t>(options.neighbors) + neighbor;
        const int local_neighbor = graph.indices[edge];
        const std::size_t input_row = static_cast<std::size_t>(group.rows[static_cast<std::size_t>(local_neighbor)]);
        const float scaled_distance = max_distance > 0.0f ? graph.distances[edge] / max_distance : 0.0f;
        const float weight = std::exp(-scaled_distance);
        for (std::size_t variable = 0; variable < data.cols; ++variable) {
          output[variable] += data.value_float(input_row, variable) * weight;
        }
      }
    });
    result.aggregation_seconds +=
      std::chrono::duration<double>(Clock::now() - aggregation_started).count();
  }
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

}  // namespace

PassingMessageResult PassingMessage_CPU(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options
) {
  return passing_message_impl(data, spatial, samples, options, Backend::CPU);
}

PassingMessageResult PassingMessage_CUDA(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options
) {
  return passing_message_impl(data, spatial, samples, options, Backend::CUDA);
}

PassingMessageResult PassingMessage_METAL(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options
) {
  return passing_message_impl(data, spatial, samples, options, Backend::Metal);
}

PassingMessageResult PassingMessage(
  const MatrixView data,
  const MatrixView spatial,
  const std::vector<int>& samples,
  const PassingMessageOptions& options
) {
  switch (options.backend) {
    case Backend::CUDA: return PassingMessage_CUDA(data, spatial, samples, options);
    case Backend::Metal: return PassingMessage_METAL(data, spatial, samples, options);
    case Backend::Auto:
#if defined(KODAMA_ENABLE_CUDA)
      return PassingMessage_CUDA(data, spatial, samples, options);
#elif defined(KODAMA_ENABLE_METAL)
      if (detail::metal_backend_available()) {
        return PassingMessage_METAL(data, spatial, samples, options);
      }
#endif
      return PassingMessage_CPU(data, spatial, samples, options);
    case Backend::CPU: return PassingMessage_CPU(data, spatial, samples, options);
  }
  throw std::invalid_argument("Unsupported passing-message backend.");
}

}  // namespace kodama
