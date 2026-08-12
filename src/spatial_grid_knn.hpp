// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kodama/kodama.hpp"

namespace kodama::detail {

struct GridCandidate {
  float distance = std::numeric_limits<float>::infinity();
  int index = -1;
};

inline bool grid_candidate_better(float lhs_distance, int lhs_index, const GridCandidate& rhs) {
  return lhs_distance < rhs.distance || (lhs_distance == rhs.distance && lhs_index < rhs.index);
}

inline void grid_insert_candidate(std::vector<GridCandidate>& top, int k, int index, float distance) {
  if (index < 0) return;
  if (static_cast<int>(top.size()) == k && !grid_candidate_better(distance, index, top.back())) return;
  GridCandidate candidate{distance, index};
  auto pos = std::lower_bound(
    top.begin(),
    top.end(),
    candidate,
    [](const GridCandidate& lhs, const GridCandidate& rhs) {
      return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.index < rhs.index);
    }
  );
  top.insert(pos, candidate);
  if (static_cast<int>(top.size()) > k) top.pop_back();
}

inline int spatial_grid_bins_per_dim(int n, int k, int dims) {
  const double kk = static_cast<double>(std::max(1, k));
  const double target_occupancy = dims == 3 ?
    std::max(1.5, std::min(8.0, kk / 25.0)) :
    std::max(4.0, std::min(16.0, kk / 10.0));
  const double bins = dims == 3 ?
    std::ceil(std::pow(static_cast<double>(n) / target_occupancy, 1.0 / 3.0)) :
    std::ceil(std::sqrt(static_cast<double>(n) / target_occupancy));
  return std::max(4, std::min(4096, static_cast<int>(bins)));
}

inline int grid_coord(float value, float min_value, float cell_size, int bins) {
  int out = static_cast<int>((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= bins) out = bins - 1;
  return out;
}

inline int grid_cell_id_2d(int ix, int iy, int bins) {
  return iy * bins + ix;
}

inline int grid_cell_id_3d(int ix, int iy, int iz, int bins) {
  return (iz * bins + iy) * bins + ix;
}

struct SpatialGridIndex {
  int dims = 2;
  int bins = 1;
  int n_cells = 1;
  float min_x = 0.0f;
  float min_y = 0.0f;
  float min_z = 0.0f;
  float cell_x = 1.0f;
  float cell_y = 1.0f;
  float cell_z = 1.0f;
  std::vector<int> offsets;
  std::vector<int> rows;
};

inline SpatialGridIndex build_spatial_grid_index(const float* data, int n, int dims, int bins) {
  if (n < 2 || (dims != 2 && dims != 3) || bins < 1) {
    throw std::invalid_argument("Spatial grid KNN requires n >= 2, dims 2 or 3, and positive bins.");
  }
  SpatialGridIndex grid;
  grid.dims = dims;
  grid.bins = bins;
  const long long n_cells_ll = dims == 3 ?
    static_cast<long long>(bins) * bins * bins :
    static_cast<long long>(bins) * bins;
  if (n_cells_ll > static_cast<long long>(std::numeric_limits<int>::max() - 1)) {
    throw std::invalid_argument("Spatial grid KNN requested too many grid cells.");
  }
  grid.n_cells = static_cast<int>(n_cells_ll);

  grid.min_x = data[0];
  grid.min_y = data[1];
  grid.min_z = dims == 3 ? data[2] : 0.0f;
  float max_x = grid.min_x;
  float max_y = grid.min_y;
  float max_z = grid.min_z;
  for (int i = 1; i < n; ++i) {
    const float x = data[static_cast<std::size_t>(i) * dims];
    const float y = data[static_cast<std::size_t>(i) * dims + 1];
    grid.min_x = std::min(grid.min_x, x);
    grid.min_y = std::min(grid.min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    if (dims == 3) {
      const float z = data[static_cast<std::size_t>(i) * dims + 2];
      grid.min_z = std::min(grid.min_z, z);
      max_z = std::max(max_z, z);
    }
  }
  grid.cell_x = std::nextafter(std::max(max_x - grid.min_x, std::numeric_limits<float>::epsilon()), std::numeric_limits<float>::infinity()) /
    static_cast<float>(bins);
  grid.cell_y = std::nextafter(std::max(max_y - grid.min_y, std::numeric_limits<float>::epsilon()), std::numeric_limits<float>::infinity()) /
    static_cast<float>(bins);
  grid.cell_z = dims == 3 ?
    std::nextafter(std::max(max_z - grid.min_z, std::numeric_limits<float>::epsilon()), std::numeric_limits<float>::infinity()) /
      static_cast<float>(bins) :
    1.0f;

  grid.offsets.assign(static_cast<std::size_t>(grid.n_cells + 1), 0);
  std::vector<int> cell_ids(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    const int ix = grid_coord(data[base], grid.min_x, grid.cell_x, bins);
    const int iy = grid_coord(data[base + 1], grid.min_y, grid.cell_y, bins);
    const int iz = dims == 3 ? grid_coord(data[base + 2], grid.min_z, grid.cell_z, bins) : 0;
    const int cell = dims == 3 ? grid_cell_id_3d(ix, iy, iz, bins) : grid_cell_id_2d(ix, iy, bins);
    cell_ids[static_cast<std::size_t>(i)] = cell;
    ++grid.offsets[static_cast<std::size_t>(cell + 1)];
  }
  for (int c = 1; c <= grid.n_cells; ++c) {
    grid.offsets[static_cast<std::size_t>(c)] += grid.offsets[static_cast<std::size_t>(c - 1)];
  }
  grid.rows.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> cursor = grid.offsets;
  for (int i = 0; i < n; ++i) {
    const int cell = cell_ids[static_cast<std::size_t>(i)];
    grid.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cell)]++)] = i;
  }
  return grid;
}

inline void add_grid_cell_candidates(
  const float* data,
  const SpatialGridIndex& grid,
  int n,
  int query,
  int ix,
  int iy,
  int iz,
  int k,
  std::vector<GridCandidate>& top
) {
  if (ix < 0 || iy < 0 || ix >= grid.bins || iy >= grid.bins) return;
  if (grid.dims == 3 && (iz < 0 || iz >= grid.bins)) return;
  const int cell = grid.dims == 3 ?
    grid_cell_id_3d(ix, iy, iz, grid.bins) :
    grid_cell_id_2d(ix, iy, grid.bins);
  const int start = grid.offsets[static_cast<std::size_t>(cell)];
  const int end = grid.offsets[static_cast<std::size_t>(cell + 1)];
  const std::size_t qbase = static_cast<std::size_t>(query) * grid.dims;
  const float qx = data[qbase];
  const float qy = data[qbase + 1];
  const float qz = grid.dims == 3 ? data[qbase + 2] : 0.0f;
  for (int pos = start; pos < end; ++pos) {
    const int candidate = grid.rows[static_cast<std::size_t>(pos)];
    if (candidate == query || candidate < 0 || candidate >= n) continue;
    const std::size_t cbase = static_cast<std::size_t>(candidate) * grid.dims;
    const float dx = qx - data[cbase];
    const float dy = qy - data[cbase + 1];
    float dist = dx * dx + dy * dy;
    if (grid.dims == 3) {
      const float dz = qz - data[cbase + 2];
      dist += dz * dz;
    }
    grid_insert_candidate(top, k, candidate, dist);
  }
}

inline float grid_lower_bound_outside(
  float x,
  float y,
  float z,
  const SpatialGridIndex& grid,
  int x0,
  int x1,
  int y0,
  int y1,
  int z0,
  int z1
) {
  float best = std::numeric_limits<float>::infinity();
  if (x0 > 0) {
    const float border = grid.min_x + static_cast<float>(x0) * grid.cell_x;
    const float dx = std::max(0.0f, x - border);
    best = std::min(best, dx * dx);
  }
  if (x1 + 1 < grid.bins) {
    const float border = grid.min_x + static_cast<float>(x1 + 1) * grid.cell_x;
    const float dx = std::max(0.0f, border - x);
    best = std::min(best, dx * dx);
  }
  if (y0 > 0) {
    const float border = grid.min_y + static_cast<float>(y0) * grid.cell_y;
    const float dy = std::max(0.0f, y - border);
    best = std::min(best, dy * dy);
  }
  if (y1 + 1 < grid.bins) {
    const float border = grid.min_y + static_cast<float>(y1 + 1) * grid.cell_y;
    const float dy = std::max(0.0f, border - y);
    best = std::min(best, dy * dy);
  }
  if (grid.dims == 3) {
    if (z0 > 0) {
      const float border = grid.min_z + static_cast<float>(z0) * grid.cell_z;
      const float dz = std::max(0.0f, z - border);
      best = std::min(best, dz * dz);
    }
    if (z1 + 1 < grid.bins) {
      const float border = grid.min_z + static_cast<float>(z1 + 1) * grid.cell_z;
      const float dz = std::max(0.0f, border - z);
      best = std::min(best, dz * dz);
    }
  }
  return best;
}

inline void search_spatial_grid_exact(
  const float* data,
  const SpatialGridIndex& grid,
  int n,
  int query,
  int k,
  std::vector<GridCandidate>& top
) {
  top.clear();
  const std::size_t qbase = static_cast<std::size_t>(query) * grid.dims;
  const float qx = data[qbase];
  const float qy = data[qbase + 1];
  const float qz = grid.dims == 3 ? data[qbase + 2] : 0.0f;
  const int cx = grid_coord(qx, grid.min_x, grid.cell_x, grid.bins);
  const int cy = grid_coord(qy, grid.min_y, grid.cell_y, grid.bins);
  const int cz = grid.dims == 3 ? grid_coord(qz, grid.min_z, grid.cell_z, grid.bins) : 0;

  for (int radius = 0; radius <= grid.bins; ++radius) {
    const int raw_x0 = cx - radius;
    const int raw_x1 = cx + radius;
    const int raw_y0 = cy - radius;
    const int raw_y1 = cy + radius;
    const int raw_z0 = cz - radius;
    const int raw_z1 = cz + radius;
    const int x0 = std::max(0, raw_x0);
    const int x1 = std::min(grid.bins - 1, raw_x1);
    const int y0 = std::max(0, raw_y0);
    const int y1 = std::min(grid.bins - 1, raw_y1);
    const int z0 = grid.dims == 3 ? std::max(0, raw_z0) : 0;
    const int z1 = grid.dims == 3 ? std::min(grid.bins - 1, raw_z1) : 0;

    if (grid.dims == 2) {
      if (radius == 0) {
        add_grid_cell_candidates(data, grid, n, query, cx, cy, 0, k, top);
      } else {
        for (int ix = raw_x0; ix <= raw_x1; ++ix) {
          if (raw_y0 >= 0 && raw_y0 < grid.bins) add_grid_cell_candidates(data, grid, n, query, ix, raw_y0, 0, k, top);
          if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < grid.bins) add_grid_cell_candidates(data, grid, n, query, ix, raw_y1, 0, k, top);
        }
        for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
          if (raw_x0 >= 0 && raw_x0 < grid.bins) add_grid_cell_candidates(data, grid, n, query, raw_x0, iy, 0, k, top);
          if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < grid.bins) add_grid_cell_candidates(data, grid, n, query, raw_x1, iy, 0, k, top);
        }
      }
    } else {
      if (radius == 0) {
        add_grid_cell_candidates(data, grid, n, query, cx, cy, cz, k, top);
      } else {
        for (int iz = raw_z0; iz <= raw_z1; ++iz) {
          if (iz < 0 || iz >= grid.bins) continue;
          for (int iy = raw_y0; iy <= raw_y1; ++iy) {
            if (iy < 0 || iy >= grid.bins) continue;
            for (int ix = raw_x0; ix <= raw_x1; ++ix) {
              if (ix < 0 || ix >= grid.bins) continue;
              if (ix != raw_x0 && ix != raw_x1 && iy != raw_y0 && iy != raw_y1 && iz != raw_z0 && iz != raw_z1) continue;
              add_grid_cell_candidates(data, grid, n, query, ix, iy, iz, k, top);
            }
          }
        }
      }
    }

    if (static_cast<int>(top.size()) == k) {
      const float kth = top.back().distance;
      const float lower = grid_lower_bound_outside(qx, qy, qz, grid, x0, x1, y0, y1, z0, z1);
      if (lower > kth) break;
    }
  }
}

inline void add_grid_cell_nearest_query(
  const float* base,
  const float* query,
  const SpatialGridIndex& grid,
  int query_row,
  int ix,
  int iy,
  int iz,
  GridCandidate& best
) {
  if (ix < 0 || iy < 0 || ix >= grid.bins || iy >= grid.bins) return;
  if (grid.dims == 3 && (iz < 0 || iz >= grid.bins)) return;
  const int cell = grid.dims == 3 ?
    grid_cell_id_3d(ix, iy, iz, grid.bins) :
    grid_cell_id_2d(ix, iy, grid.bins);
  const int start = grid.offsets[static_cast<std::size_t>(cell)];
  const int end = grid.offsets[static_cast<std::size_t>(cell + 1)];
  const std::size_t query_base = static_cast<std::size_t>(query_row) * grid.dims;
  for (int position = start; position < end; ++position) {
    const int candidate = grid.rows[static_cast<std::size_t>(position)];
    const std::size_t candidate_base = static_cast<std::size_t>(candidate) * grid.dims;
    const float dx = query[query_base] - base[candidate_base];
    const float dy = query[query_base + 1] - base[candidate_base + 1];
    float distance = dx * dx + dy * dy;
    if (grid.dims == 3) {
      const float dz = query[query_base + 2] - base[candidate_base + 2];
      distance += dz * dz;
    }
    if (grid_candidate_better(distance, candidate, best)) {
      best = GridCandidate{distance, candidate};
    }
  }
}

inline GridCandidate search_spatial_grid_nearest_query(
  const float* base,
  const float* query,
  const SpatialGridIndex& grid,
  int query_row
) {
  const std::size_t query_base = static_cast<std::size_t>(query_row) * grid.dims;
  const float qx = query[query_base];
  const float qy = query[query_base + 1];
  const float qz = grid.dims == 3 ? query[query_base + 2] : 0.0f;
  const int cx = grid_coord(qx, grid.min_x, grid.cell_x, grid.bins);
  const int cy = grid_coord(qy, grid.min_y, grid.cell_y, grid.bins);
  const int cz = grid.dims == 3 ? grid_coord(qz, grid.min_z, grid.cell_z, grid.bins) : 0;
  GridCandidate best;

  for (int radius = 0; radius <= grid.bins; ++radius) {
    const int raw_x0 = cx - radius;
    const int raw_x1 = cx + radius;
    const int raw_y0 = cy - radius;
    const int raw_y1 = cy + radius;
    const int raw_z0 = cz - radius;
    const int raw_z1 = cz + radius;
    if (grid.dims == 2) {
      if (radius == 0) {
        add_grid_cell_nearest_query(base, query, grid, query_row, cx, cy, 0, best);
      } else {
        for (int ix = raw_x0; ix <= raw_x1; ++ix) {
          add_grid_cell_nearest_query(base, query, grid, query_row, ix, raw_y0, 0, best);
          if (raw_y1 != raw_y0) {
            add_grid_cell_nearest_query(base, query, grid, query_row, ix, raw_y1, 0, best);
          }
        }
        for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
          add_grid_cell_nearest_query(base, query, grid, query_row, raw_x0, iy, 0, best);
          if (raw_x1 != raw_x0) {
            add_grid_cell_nearest_query(base, query, grid, query_row, raw_x1, iy, 0, best);
          }
        }
      }
    } else {
      for (int iz = raw_z0; iz <= raw_z1; ++iz) {
        for (int iy = raw_y0; iy <= raw_y1; ++iy) {
          for (int ix = raw_x0; ix <= raw_x1; ++ix) {
            if (radius > 0 && ix != raw_x0 && ix != raw_x1 &&
                iy != raw_y0 && iy != raw_y1 && iz != raw_z0 && iz != raw_z1) {
              continue;
            }
            add_grid_cell_nearest_query(base, query, grid, query_row, ix, iy, iz, best);
          }
        }
      }
    }

    if (best.index >= 0) {
      const int x0 = std::max(0, raw_x0);
      const int x1 = std::min(grid.bins - 1, raw_x1);
      const int y0 = std::max(0, raw_y0);
      const int y1 = std::min(grid.bins - 1, raw_y1);
      const int z0 = grid.dims == 3 ? std::max(0, raw_z0) : 0;
      const int z1 = grid.dims == 3 ? std::min(grid.bins - 1, raw_z1) : 0;
      if (grid_lower_bound_outside(qx, qy, qz, grid, x0, x1, y0, y1, z0, z1) > best.distance) {
        break;
      }
    }
  }
  return best;
}

inline NeighborGraph spatial_grid_query_nearest(
  const float* base,
  int base_rows,
  const float* query,
  int query_rows,
  int dims,
  int n_threads
) {
  if (base == nullptr || query == nullptr || base_rows < 1 || query_rows < 1 ||
      (dims != 2 && dims != 3)) {
    throw std::invalid_argument("Spatial grid query supports non-empty 2D/3D matrices.");
  }
  const int bins = spatial_grid_bins_per_dim(base_rows, 1, dims);
  const SpatialGridIndex grid = build_spatial_grid_index(base, base_rows, dims, bins);
  NeighborGraph graph;
  graph.neighbors = 1;
  graph.indices.assign(static_cast<std::size_t>(query_rows), -1);
  graph.distances.assign(static_cast<std::size_t>(query_rows), std::numeric_limits<float>::infinity());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
#else
  (void)n_threads;
#endif
  for (int row = 0; row < query_rows; ++row) {
    const GridCandidate best = search_spatial_grid_nearest_query(base, query, grid, row);
    graph.indices[static_cast<std::size_t>(row)] = best.index;
    graph.distances[static_cast<std::size_t>(row)] =
      std::sqrt(std::max(0.0f, best.distance));
  }
  return graph;
}

inline NeighborGraph spatial_grid_self_knn(
  const float* data,
  int n,
  int dims,
  int neighbors,
  int n_threads,
  bool one_based_indices = false,
  bool include_self = false
) {
  if (data == nullptr || n < 2 || (dims != 2 && dims != 3)) {
    throw std::invalid_argument("Spatial grid KNN supports only 2D/3D matrices with at least two rows.");
  }
  const int k = include_self ?
    std::max(1, std::min(neighbors, n)) :
    std::max(1, std::min(neighbors, n - 1));
  const int nonself_k = include_self ? std::max(0, k - 1) : k;
  const int bins = spatial_grid_bins_per_dim(n, std::max(1, nonself_k), dims);
  const SpatialGridIndex grid = build_spatial_grid_index(data, n, dims, bins);

  NeighborGraph graph;
  graph.neighbors = k;
  graph.indices.assign(static_cast<std::size_t>(n) * k, -1);
  graph.distances.assign(static_cast<std::size_t>(n) * k, std::numeric_limits<float>::infinity());

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads > 0 ? n_threads : 1)
  {
    std::vector<GridCandidate> top;
    top.reserve(static_cast<std::size_t>(std::max(1, nonself_k)));
#pragma omp for schedule(static)
    for (int q = 0; q < n; ++q) {
      if (nonself_k > 0) search_spatial_grid_exact(data, grid, n, q, nonself_k, top);
      else top.clear();
      if (static_cast<int>(top.size()) < nonself_k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (candidate == q) continue;
          const std::size_t qb = static_cast<std::size_t>(q) * dims;
          const std::size_t cb = static_cast<std::size_t>(candidate) * dims;
          const float dx = data[qb] - data[cb];
          const float dy = data[qb + 1] - data[cb + 1];
          float dist = dx * dx + dy * dy;
          if (dims == 3) {
            const float dz = data[qb + 2] - data[cb + 2];
            dist += dz * dz;
          }
          grid_insert_candidate(top, nonself_k, candidate, dist);
        }
      }
      int out_col = 0;
      if (include_self) {
        const std::size_t offset = static_cast<std::size_t>(q) * k;
        graph.indices[offset] = q + (one_based_indices ? 1 : 0);
        graph.distances[offset] = 0.0f;
        out_col = 1;
      }
      for (int j = 0; out_col < k && j < static_cast<int>(top.size()); ++j, ++out_col) {
        const std::size_t offset = static_cast<std::size_t>(q) * k + static_cast<std::size_t>(out_col);
        graph.indices[offset] = top[static_cast<std::size_t>(j)].index + (one_based_indices ? 1 : 0);
        graph.distances[offset] = std::sqrt(std::max(top[static_cast<std::size_t>(j)].distance, 0.0f));
        }
    }
  }
#else
  (void)n_threads;
  std::vector<GridCandidate> top;
  top.reserve(static_cast<std::size_t>(std::max(1, nonself_k)));
  for (int q = 0; q < n; ++q) {
    if (nonself_k > 0) search_spatial_grid_exact(data, grid, n, q, nonself_k, top);
    else top.clear();
    if (static_cast<int>(top.size()) < nonself_k) {
      for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == q) continue;
        const std::size_t qb = static_cast<std::size_t>(q) * dims;
        const std::size_t cb = static_cast<std::size_t>(candidate) * dims;
        const float dx = data[qb] - data[cb];
        const float dy = data[qb + 1] - data[cb + 1];
        float dist = dx * dx + dy * dy;
        if (dims == 3) {
          const float dz = data[qb + 2] - data[cb + 2];
          dist += dz * dz;
        }
        grid_insert_candidate(top, nonself_k, candidate, dist);
      }
    }
    int out_col = 0;
    if (include_self) {
      const std::size_t offset = static_cast<std::size_t>(q) * k;
      graph.indices[offset] = q + (one_based_indices ? 1 : 0);
      graph.distances[offset] = 0.0f;
      out_col = 1;
    }
    for (int j = 0; out_col < k && j < static_cast<int>(top.size()); ++j, ++out_col) {
      const std::size_t offset = static_cast<std::size_t>(q) * k + static_cast<std::size_t>(out_col);
      graph.indices[offset] = top[static_cast<std::size_t>(j)].index + (one_based_indices ? 1 : 0);
      graph.distances[offset] = std::sqrt(std::max(top[static_cast<std::size_t>(j)].distance, 0.0f));
    }
  }
#endif
  return graph;
}

inline bool should_use_spatial_grid_knn(int n, int dims, DistanceMetric metric) {
  return n >= 10000 && (dims == 2 || dims == 3) && metric == DistanceMetric::Euclidean;
}

}  // namespace kodama::detail
