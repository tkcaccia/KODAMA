// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"
#endif

namespace kodama {
namespace {

struct Timer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
};

std::vector<float> copy_float32(MatrixView x) {
  std::vector<float> out(x.rows * x.cols);
  for (std::size_t i = 0; i < x.rows; ++i) {
    for (std::size_t j = 0; j < x.cols; ++j) out[i * x.cols + j] = x.value_float(i, j);
  }
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
void normalize_rows(std::vector<float>& x, std::size_t rows, std::size_t cols) {
  for (std::size_t i = 0; i < rows; ++i) {
    double norm2 = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      const double v = x[i * cols + j];
      norm2 += v * v;
    }
    const double norm = std::sqrt(norm2);
    if (norm <= 0.0) continue;
    for (std::size_t j = 0; j < cols; ++j) x[i * cols + j] = static_cast<float>(x[i * cols + j] / norm);
  }
}
#endif

NeighborGraph build_hnsw_graph(MatrixView x, const GraphClusterOptions& options) {
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  const std::vector<float> data = detail::prepare_native_matrix(x, options.metric);
  std::vector<int> self_indices(static_cast<std::size_t>(n));
  std::iota(self_indices.begin(), self_indices.end(), 0);
  const detail::NativeKNNResult search = detail::native_hnsw_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    detail::NativeHNSWParameters{
      std::min(32, std::max(2, n - 1)),
      200,
      std::max(150, k + 1)
    },
    options.n_threads,
    self_indices
  );

  NeighborGraph out;
  out.neighbors = k;
  out.index_base = GraphIndexBase::One;
  out.indices.assign(static_cast<std::size_t>(n) * k, -1);
  out.distances.assign(static_cast<std::size_t>(n) * k, std::numeric_limits<float>::infinity());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < search.neighbors; ++j) {
      const std::size_t pos = static_cast<std::size_t>(i) * static_cast<std::size_t>(search.neighbors) + static_cast<std::size_t>(j);
      const int nb = search.indices[pos];
      if (nb < 0 || nb == i) continue;
      const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j);
      out.indices[off] = nb + 1;
      out.distances[off] = detail::native_knn_output_distance(search.distances[pos], options.metric);
    }
  }
  return out;
}

#if defined(KODAMA_ENABLE_METAL)
NeighborGraph build_metal_graph(MatrixView x, const GraphClusterOptions& options) {
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  const std::vector<float> data = detail::prepare_native_matrix(x, options.metric);
  std::vector<int> self_indices(static_cast<std::size_t>(n));
  std::iota(self_indices.begin(), self_indices.end(), 0);
  const detail::NativeKNNResult search = detail::metal_exact_knn_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    self_indices
  );

  NeighborGraph out;
  out.neighbors = search.neighbors;
  out.index_base = GraphIndexBase::One;
  out.indices.resize(search.indices.size(), -1);
  out.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.indices.size(); ++i) {
    if (search.indices[i] >= 0) out.indices[i] = search.indices[i] + 1;
    out.distances[i] = detail::native_knn_output_distance(search.distances[i], options.metric);
  }
  return out;
}
#endif

std::uint64_t edge_key(int a, int b) {
  const std::uint32_t u = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t v = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

struct Edge {
  std::uint64_t key = 0;
  double weight = 0.0;
};

int edge_from(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key >> 32)); }
int edge_to(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key & 0xffffffffULL)); }

int normalize_index(int idx, int n, bool one_based) {
  if (idx < 0) return -1;
  const int z = one_based ? idx - 1 : idx;
  return z >= 0 && z < n ? z : -1;
}

bool graph_is_one_based(const NeighborGraph& graph, int n) {
  if (graph.index_base == GraphIndexBase::One) return true;
  if (graph.index_base == GraphIndexBase::Zero) return false;
  for (int idx : graph.indices) {
    if (idx == 0) return false;
    if (idx == n) return true;
  }
  return true;
}

bool contains_neighbor(const NeighborGraph& graph, int row, int target0, int n, bool one_based) {
  for (int col = 0; col < graph.neighbors; ++col) {
    const int nb = normalize_index(graph.indices[static_cast<std::size_t>(row) * graph.neighbors + col], n, one_based);
    if (nb == target0) return true;
  }
  return false;
}

void push_edge(std::vector<Edge>& edges, int u0, int v0, double weight, double prune) {
  if (u0 == v0 || !std::isfinite(weight) || weight <= prune) return;
  edges.push_back({edge_key(u0, v0), weight});
}

std::vector<double> local_sigmas(const NeighborGraph& graph, int n) {
  std::vector<double> sigma(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    double last = 0.0;
    double sum = 0.0;
    int count = 0;
    for (int j = 0; j < graph.neighbors; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(i) * graph.neighbors + j];
      if (std::isfinite(d) && d > 0.0f) {
        last = d;
        sum += d;
        ++count;
      }
    }
    if (last > 0.0) sigma[static_cast<std::size_t>(i)] = last;
    else if (count > 0 && sum > 0.0) sigma[static_cast<std::size_t>(i)] = sum / count;
  }
  return sigma;
}

struct EdgeList {
  int n = 0;
  std::vector<int> from;
  std::vector<int> to;
  std::vector<double> weight;
};

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    int root = x;
    while (parent[static_cast<std::size_t>(root)] != root) root = parent[static_cast<std::size_t>(root)];
    while (parent[static_cast<std::size_t>(x)] != x) {
      const int next = parent[static_cast<std::size_t>(x)];
      parent[static_cast<std::size_t>(x)] = root;
      x = next;
    }
    return root;
  }

  bool unite(int a, int b) {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) return false;
    if (rank[static_cast<std::size_t>(ra)] < rank[static_cast<std::size_t>(rb)]) std::swap(ra, rb);
    parent[static_cast<std::size_t>(rb)] = ra;
    if (rank[static_cast<std::size_t>(ra)] == rank[static_cast<std::size_t>(rb)]) ++rank[static_cast<std::size_t>(ra)];
    return true;
  }
};

EdgeList edge_list_from_graph(const NeighborGraph& graph, int n, const GraphClusterOptions& options) {
  if (n < 2 || graph.neighbors < 1) throw std::invalid_argument("graph clustering requires a non-empty graph.");
  if (graph.indices.size() != static_cast<std::size_t>(n) * graph.neighbors ||
      graph.distances.size() != static_cast<std::size_t>(n) * graph.neighbors) {
    throw std::invalid_argument("NeighborGraph size does not match sample count.");
  }
  const bool one_based = graph_is_one_based(graph, n);
  const double prune = std::max(0.0, options.prune);
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * graph.neighbors);

  if (options.weight_type == GraphWeightType::SNN) {
    std::vector<int> valid(static_cast<std::size_t>(n), 0);
    std::vector<int> reverse_count(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb >= 0 && nb != i) {
          ++valid[static_cast<std::size_t>(i)];
          ++reverse_count[static_cast<std::size_t>(nb)];
        }
      }
    }
    std::vector<int> ptr(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) ptr[static_cast<std::size_t>(i + 1)] = ptr[static_cast<std::size_t>(i)] + reverse_count[static_cast<std::size_t>(i)];
    std::vector<int> rows(static_cast<std::size_t>(ptr.back()), 0);
    std::fill(reverse_count.begin(), reverse_count.end(), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb >= 0 && nb != i) rows[static_cast<std::size_t>(ptr[static_cast<std::size_t>(nb)] + reverse_count[static_cast<std::size_t>(nb)]++)] = i;
      }
    }
    std::vector<int> shared(static_cast<std::size_t>(n), 0);
    std::vector<int> touched;
    touched.reserve(static_cast<std::size_t>(graph.neighbors) * graph.neighbors);
    for (int i = 0; i < n; ++i) {
      touched.clear();
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb < 0 || nb == i) continue;
        if (options.mutual && !contains_neighbor(graph, nb, i, n, one_based)) continue;
        for (int pos = ptr[static_cast<std::size_t>(nb)]; pos < ptr[static_cast<std::size_t>(nb + 1)]; ++pos) {
          const int other = rows[static_cast<std::size_t>(pos)];
          if (other <= i) continue;
          int& s = shared[static_cast<std::size_t>(other)];
          if (s == 0) touched.push_back(other);
          ++s;
        }
      }
      for (int other : touched) {
        const int s = shared[static_cast<std::size_t>(other)];
        shared[static_cast<std::size_t>(other)] = 0;
        const int denom = valid[static_cast<std::size_t>(i)] + valid[static_cast<std::size_t>(other)] - s;
        if (s > 0 && denom > 0) push_edge(edges, i, other, static_cast<double>(s) / denom, prune);
      }
    }
  } else {
    const std::vector<double> sigma = options.weight_type == GraphWeightType::Adaptive ? local_sigmas(graph, n) : std::vector<double>();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const std::size_t off = static_cast<std::size_t>(i) * graph.neighbors + j;
        const int nb = normalize_index(graph.indices[off], n, one_based);
        if (nb < 0 || nb == i) continue;
        if (options.mutual && !contains_neighbor(graph, nb, i, n, one_based)) continue;
        double w = 1.0;
        if (options.weight_type == GraphWeightType::Distance) {
          const float d = graph.distances[off];
          if (!std::isfinite(d) || d < 0.0f) continue;
          w = 1.0 / (1.0 + static_cast<double>(d));
        } else if (options.weight_type == GraphWeightType::Adaptive) {
          const float d = graph.distances[off];
          if (!std::isfinite(d) || d < 0.0f) continue;
          const double scale = std::max(1e-12, sigma[static_cast<std::size_t>(i)] * sigma[static_cast<std::size_t>(nb)]);
          w = std::exp(-(static_cast<double>(d) * d) / scale);
        }
        push_edge(edges, i, nb, w, prune);
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.key < b.key; });
  std::size_t unique_count = 0;
  for (std::size_t p = 0; p < edges.size();) {
    const std::uint64_t key = edges[p].key;
    double w = edges[p].weight;
    ++p;
    while (p < edges.size() && edges[p].key == key) {
      w = std::max(w, edges[p].weight);
      ++p;
    }
    edges[unique_count++] = {key, w};
  }
  EdgeList out;
  out.n = n;
  out.from.resize(unique_count);
  out.to.resize(unique_count);
  out.weight.resize(unique_count);
  for (std::size_t i = 0; i < unique_count; ++i) {
    out.from[i] = edge_from(edges[i].key);
    out.to[i] = edge_to(edges[i].key);
    out.weight[i] = edges[i].weight;
  }
  return out;
}

std::vector<int> components_from_edges(const EdgeList& edges) {
  DisjointSet dsu(edges.n);
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u >= 0 && u < edges.n && v >= 0 && v < edges.n && u != v && std::isfinite(w) && w > 0.0) dsu.unite(u, v);
  }
  std::unordered_map<int, int> remap;
  remap.reserve(static_cast<std::size_t>(edges.n));
  std::vector<int> comp(static_cast<std::size_t>(edges.n), 0);
  int next = 0;
  for (int i = 0; i < edges.n; ++i) {
    const int root = dsu.find(i);
    auto it = remap.find(root);
    if (it == remap.end()) it = remap.emplace(root, next++).first;
    comp[static_cast<std::size_t>(i)] = it->second;
  }
  return comp;
}

int component_count(const std::vector<int>& comp) {
  int count = 0;
  for (int c : comp) count = std::max(count, c + 1);
  return count;
}

double robust_edge_weight(const EdgeList& edges) {
  std::vector<double> weights;
  weights.reserve(edges.weight.size());
  for (double w : edges.weight) {
    if (std::isfinite(w) && w > 0.0) weights.push_back(w);
  }
  if (weights.empty()) return 1.0;
  const std::size_t mid = weights.size() / 2;
  std::nth_element(weights.begin(), weights.begin() + static_cast<std::ptrdiff_t>(mid), weights.end());
  return std::max(1e-12, weights[mid]);
}

double squared_distance(MatrixView x, int a, int b) {
  double out = 0.0;
  for (std::size_t j = 0; j < x.cols; ++j) {
    const double d = static_cast<double>(x.value_float(static_cast<std::size_t>(a), j)) -
                     static_cast<double>(x.value_float(static_cast<std::size_t>(b), j));
    out += d * d;
  }
  return out;
}

double squared_centroid_distance(const std::vector<double>& centroids, int dims, int a, int b) {
  double out = 0.0;
  const std::size_t ao = static_cast<std::size_t>(a) * dims;
  const std::size_t bo = static_cast<std::size_t>(b) * dims;
  for (int j = 0; j < dims; ++j) {
    const double d = centroids[ao + static_cast<std::size_t>(j)] - centroids[bo + static_cast<std::size_t>(j)];
    out += d * d;
  }
  return out;
}

void bridge_embedding_components(EdgeList& edges, MatrixView embedding) {
  if (edges.n < 2) return;
  if (embedding.rows != static_cast<std::size_t>(edges.n)) throw std::invalid_argument("Embedding row count does not match graph vertices.");
  if (embedding.cols < 1) throw std::invalid_argument("Embedding clustering requires at least one column.");

  const std::vector<int> comp = components_from_edges(edges);
  const int n_comp = component_count(comp);
  if (n_comp <= 1) return;

  const int dims = static_cast<int>(embedding.cols);
  std::vector<int> counts(static_cast<std::size_t>(n_comp), 0);
  std::vector<double> centroids(static_cast<std::size_t>(n_comp) * dims, 0.0);
  for (int i = 0; i < edges.n; ++i) {
    const int c = comp[static_cast<std::size_t>(i)];
    ++counts[static_cast<std::size_t>(c)];
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) centroids[off + static_cast<std::size_t>(j)] += embedding(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
  }
  for (int c = 0; c < n_comp; ++c) {
    const double denom = static_cast<double>(std::max(1, counts[static_cast<std::size_t>(c)]));
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) centroids[off + static_cast<std::size_t>(j)] /= denom;
  }

  std::vector<int> representative(static_cast<std::size_t>(n_comp), -1);
  std::vector<double> rep_dist(static_cast<std::size_t>(n_comp), std::numeric_limits<double>::infinity());
  for (int i = 0; i < edges.n; ++i) {
    const int c = comp[static_cast<std::size_t>(i)];
    double dist = 0.0;
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) {
      const double d = embedding(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) - centroids[off + static_cast<std::size_t>(j)];
      dist += d * d;
    }
    if (dist < rep_dist[static_cast<std::size_t>(c)] || (dist == rep_dist[static_cast<std::size_t>(c)] && i < representative[static_cast<std::size_t>(c)])) {
      rep_dist[static_cast<std::size_t>(c)] = dist;
      representative[static_cast<std::size_t>(c)] = i;
    }
  }

  const double weight = robust_edge_weight(edges);
  std::vector<double> best_dist(static_cast<std::size_t>(n_comp), std::numeric_limits<double>::infinity());
  std::vector<int> parent(static_cast<std::size_t>(n_comp), -1);
  std::vector<char> in_tree(static_cast<std::size_t>(n_comp), 0);
  best_dist[0] = 0.0;
  for (int iter = 0; iter < n_comp; ++iter) {
    int u = -1;
    double best = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_comp; ++c) {
      if (!in_tree[static_cast<std::size_t>(c)] && best_dist[static_cast<std::size_t>(c)] < best) {
        best = best_dist[static_cast<std::size_t>(c)];
        u = c;
      }
    }
    if (u < 0) break;
    in_tree[static_cast<std::size_t>(u)] = 1;
    if (parent[static_cast<std::size_t>(u)] >= 0) {
      const int a = representative[static_cast<std::size_t>(u)];
      const int b = representative[static_cast<std::size_t>(parent[static_cast<std::size_t>(u)])];
      if (a >= 0 && b >= 0 && a != b) {
        edges.from.push_back(std::min(a, b));
        edges.to.push_back(std::max(a, b));
        edges.weight.push_back(weight);
      }
    }
    for (int v = 0; v < n_comp; ++v) {
      if (in_tree[static_cast<std::size_t>(v)] || v == u) continue;
      double dist = squared_centroid_distance(centroids, dims, u, v);
      const int ru = representative[static_cast<std::size_t>(u)];
      const int rv = representative[static_cast<std::size_t>(v)];
      if (ru >= 0 && rv >= 0) dist = std::min(dist, squared_distance(embedding, ru, rv));
      if (dist < best_dist[static_cast<std::size_t>(v)] ||
          (dist == best_dist[static_cast<std::size_t>(v)] && u < parent[static_cast<std::size_t>(v)])) {
        best_dist[static_cast<std::size_t>(v)] = dist;
        parent[static_cast<std::size_t>(v)] = u;
      }
    }
  }
}

struct CsrGraph {
  int n = 0;
  std::vector<int> ptr;
  std::vector<int> to;
  std::vector<double> weight;
  std::vector<double> degree;
  double total_weight = 0.0;
};

CsrGraph csr_from_edges(const EdgeList& edges) {
  CsrGraph g;
  g.n = edges.n;
  g.ptr.assign(static_cast<std::size_t>(g.n) + 1, 0);
  g.degree.assign(static_cast<std::size_t>(g.n), 0.0);
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u < 0 || u >= g.n || v < 0 || v >= g.n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    ++g.ptr[static_cast<std::size_t>(u + 1)];
    ++g.ptr[static_cast<std::size_t>(v + 1)];
    g.degree[static_cast<std::size_t>(u)] += w;
    g.degree[static_cast<std::size_t>(v)] += w;
    g.total_weight += w;
  }
  for (int i = 1; i <= g.n; ++i) g.ptr[static_cast<std::size_t>(i)] += g.ptr[static_cast<std::size_t>(i - 1)];
  g.to.assign(static_cast<std::size_t>(g.ptr.back()), 0);
  g.weight.assign(g.to.size(), 0.0);
  std::vector<int> fill = g.ptr;
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u < 0 || u >= g.n || v < 0 || v >= g.n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    int pos = fill[static_cast<std::size_t>(u)]++;
    g.to[static_cast<std::size_t>(pos)] = v;
    g.weight[static_cast<std::size_t>(pos)] = w;
    pos = fill[static_cast<std::size_t>(v)]++;
    g.to[static_cast<std::size_t>(pos)] = u;
    g.weight[static_cast<std::size_t>(pos)] = w;
  }
  return g;
}

std::vector<int> compact(std::vector<int> x) {
  std::unordered_map<int, int> map;
  map.reserve(x.size());
  int next = 0;
  for (int& v : x) {
    auto it = map.find(v);
    if (it == map.end()) it = map.emplace(v, next++).first;
    v = it->second;
  }
  return x;
}

std::vector<double> community_degree(const CsrGraph& g, const std::vector<int>& membership) {
  int max_comm = 0;
  for (int c : membership) max_comm = std::max(max_comm, c);
  std::vector<double> deg(static_cast<std::size_t>(max_comm) + 1, 0.0);
  for (int i = 0; i < g.n; ++i) deg[static_cast<std::size_t>(membership[static_cast<std::size_t>(i)])] += g.degree[static_cast<std::size_t>(i)];
  return deg;
}

double modularity(const CsrGraph& g, const std::vector<int>& membership) {
  if (g.n == 0 || g.total_weight <= 0.0) return 0.0;
  const double two_m = 2.0 * g.total_weight;
  double internal = 0.0;
  for (int u = 0; u < g.n; ++u) {
    const int cu = membership[static_cast<std::size_t>(u)];
    for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
      const int v = g.to[static_cast<std::size_t>(p)];
      if (cu == membership[static_cast<std::size_t>(v)]) internal += g.weight[static_cast<std::size_t>(p)];
    }
  }
  const std::vector<double> cdeg = community_degree(g, membership);
  double expected = 0.0;
  for (double d : cdeg) expected += d * d;
  return internal / two_m - expected / (two_m * two_m);
}

std::vector<int> split_disconnected(const CsrGraph& g, const std::vector<int>& membership) {
  std::vector<int> refined(static_cast<std::size_t>(g.n), -1);
  std::vector<char> seen(static_cast<std::size_t>(g.n), 0);
  std::queue<int> q;
  int next = 0;
  for (int start = 0; start < g.n; ++start) {
    if (seen[static_cast<std::size_t>(start)]) continue;
    const int comm = membership[static_cast<std::size_t>(start)];
    seen[static_cast<std::size_t>(start)] = 1;
    refined[static_cast<std::size_t>(start)] = next;
    q.push(start);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int v = g.to[static_cast<std::size_t>(p)];
        if (!seen[static_cast<std::size_t>(v)] && membership[static_cast<std::size_t>(v)] == comm) {
          seen[static_cast<std::size_t>(v)] = 1;
          refined[static_cast<std::size_t>(v)] = next;
          q.push(v);
        }
      }
    }
    ++next;
  }
  return compact(std::move(refined));
}

std::vector<int> random_walk_cluster(const CsrGraph& g, int steps, int max_iter, int n_threads) {
  std::vector<int> labels(static_cast<std::size_t>(g.n));
  std::iota(labels.begin(), labels.end(), 0);
  if (g.total_weight <= 0.0) return labels;
  steps = std::max(1, steps);
  max_iter = std::max(1, max_iter);
  n_threads = std::max(1, n_threads);
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<int> proposed = labels;
    int changed = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256) reduction(+:changed)
#endif
    for (int u = 0; u < g.n; ++u) {
      std::unordered_map<int, double> score;
      std::unordered_map<int, double> frontier;
      frontier[u] = 1.0;
      for (int step = 0; step < steps; ++step) {
        std::unordered_map<int, double> next;
        next.reserve(frontier.size() * 2 + 1);
        for (const auto& item : frontier) {
          const int x = item.first;
          const double mass = item.second;
          const double deg = g.degree[static_cast<std::size_t>(x)];
          if (deg <= 0.0) continue;
          for (int p = g.ptr[static_cast<std::size_t>(x)]; p < g.ptr[static_cast<std::size_t>(x + 1)]; ++p) {
            next[g.to[static_cast<std::size_t>(p)]] += mass * g.weight[static_cast<std::size_t>(p)] / deg;
          }
        }
        frontier.swap(next);
        for (const auto& item : frontier) score[labels[static_cast<std::size_t>(item.first)]] += item.second;
      }
      int best = labels[static_cast<std::size_t>(u)];
      double best_score = -1.0;
      for (const auto& kv : score) {
        if (kv.second > best_score + 1e-15 || (std::abs(kv.second - best_score) <= 1e-15 && kv.first < best)) {
          best_score = kv.second;
          best = kv.first;
        }
      }
      proposed[static_cast<std::size_t>(u)] = best;
      if (best != labels[static_cast<std::size_t>(u)]) ++changed;
    }
    labels = compact(std::move(proposed));
    if (changed == 0) break;
  }
  return split_disconnected(g, labels);
}

std::vector<int> merge_communities_to_target(
  const CsrGraph& g,
  std::vector<int> membership,
  int target
) {
  membership = compact(std::move(membership));
  int communities = 0;
  for (int label : membership) communities = std::max(communities, label + 1);
  if (communities == target) return membership;

  // A target cut is a graph coarsening operation. If random-walk refinement
  // has already crossed below the requested cut, restart from singleton
  // communities so that the target remains exact and deterministic.
  if (communities < target) {
    membership.resize(static_cast<std::size_t>(g.n));
    std::iota(membership.begin(), membership.end(), 0);
    communities = g.n;
  }

  while (communities > target) {
    std::vector<double> volume(static_cast<std::size_t>(communities), 0.0);
    std::unordered_map<std::uint64_t, double> between;
    between.reserve(g.weight.size() / 2 + 1);
    for (int u = 0; u < g.n; ++u) {
      const int cu = membership[static_cast<std::size_t>(u)];
      volume[static_cast<std::size_t>(cu)] += g.degree[static_cast<std::size_t>(u)];
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int cv = membership[static_cast<std::size_t>(g.to[static_cast<std::size_t>(p)])];
        if (cu == cv) continue;
        const std::uint32_t lo = static_cast<std::uint32_t>(std::min(cu, cv));
        const std::uint32_t hi = static_cast<std::uint32_t>(std::max(cu, cv));
        const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | hi;
        between[key] += g.weight[static_cast<std::size_t>(p)];
      }
    }

    int merge_a = -1;
    int merge_b = -1;
    double best = -1.0;
    for (const auto& item : between) {
      const int a = static_cast<int>(item.first >> 32);
      const int b = static_cast<int>(item.first & 0xffffffffULL);
      const double denom = std::sqrt(
        std::max(volume[static_cast<std::size_t>(a)] *
                 volume[static_cast<std::size_t>(b)], 1e-30)
      );
      const double score = item.second / denom;
      if (score > best + 1e-15 ||
          (std::abs(score - best) <= 1e-15 &&
           (a < merge_a || (a == merge_a && b < merge_b)))) {
        best = score;
        merge_a = a;
        merge_b = b;
      }
    }

    if (merge_a < 0) {
      // Disconnected components have no cross-edge. Join the two smallest
      // communities only when an exact user-requested cut requires it.
      std::vector<int> size(static_cast<std::size_t>(communities), 0);
      for (int label : membership) ++size[static_cast<std::size_t>(label)];
      for (int c = 0; c < communities; ++c) {
        if (merge_a < 0 || size[static_cast<std::size_t>(c)] < size[static_cast<std::size_t>(merge_a)]) {
          merge_b = merge_a;
          merge_a = c;
        } else if (merge_b < 0 || size[static_cast<std::size_t>(c)] < size[static_cast<std::size_t>(merge_b)]) {
          merge_b = c;
        }
      }
    }

    for (int& label : membership) {
      if (label == merge_b) label = merge_a;
    }
    membership = compact(std::move(membership));
    --communities;
  }
  return membership;
}

GraphClusterResult make_result(const CsrGraph& g, const std::vector<int>& membership0, const GraphClusterOptions& options, int n_edges, Backend backend, double elapsed) {
  GraphClusterResult out;
  out.membership.resize(static_cast<std::size_t>(g.n));
  int max_comm = 0;
  for (int i = 0; i < g.n; ++i) {
    out.membership[static_cast<std::size_t>(i)] = membership0[static_cast<std::size_t>(i)] + 1;
    max_comm = std::max(max_comm, membership0[static_cast<std::size_t>(i)] + 1);
  }
  out.modularity = modularity(g, membership0);
  out.n_communities = max_comm;
  out.n_vertices = g.n;
  out.n_edges = n_edges;
  out.target_clusters = options.target_clusters;
  out.target_gap = options.target_clusters > 0 ? std::abs(out.n_communities - options.target_clusters) : 0;
  out.target_exact = options.target_clusters == 0 || out.target_gap == 0;
  out.backend = backend;
  out.runtime_seconds = elapsed;
  return out;
}

GraphClusterResult run_cpu_cluster(const EdgeList& edge_list, const GraphClusterOptions& options, double elapsed_start = 0.0) {
  Timer timer;
  CsrGraph g = csr_from_edges(edge_list);
  if (g.n == 0) throw std::invalid_argument("empty graph.");
  if (options.target_clusters > g.n) {
    throw std::invalid_argument("target_clusters cannot exceed number of graph vertices.");
  }

  GraphClusterOptions opts = options;
  opts.n_iterations = std::max(1, opts.n_iterations);
  opts.random_walk_steps = std::max(1, opts.random_walk_steps);
  opts.n_threads = std::max(1, opts.n_threads);
  std::vector<int> membership = random_walk_cluster(
    g,
    opts.random_walk_steps,
    opts.n_iterations,
    opts.n_threads
  );
  if (opts.target_clusters > 0) {
    membership = merge_communities_to_target(g, std::move(membership), opts.target_clusters);
  }
  GraphClusterResult out = make_result(
    g,
    membership,
    opts,
    static_cast<int>(edge_list.from.size()),
    Backend::CPU,
    elapsed_start + timer.seconds()
  );
  return out;
}

}  // namespace

NeighborGraph KODAMAKNNGraph_CPU(MatrixView x, const GraphClusterOptions& options) {
  GraphClusterOptions opts = options;
  opts.backend = Backend::CPU;
  if (detail::should_use_spatial_grid_knn(static_cast<int>(x.rows), static_cast<int>(x.cols), opts.metric)) {
    const std::vector<float> data = copy_float32(x);
    return detail::spatial_grid_self_knn(
      data.data(),
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      opts.k,
      opts.n_threads,
      true
    );
  }
  return build_hnsw_graph(x, opts);
}

NeighborGraph KODAMAKNNGraph_CUDA(MatrixView x, const GraphClusterOptions& options) {
#ifdef KODAMA_ENABLE_CUDA
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  std::vector<float> data = copy_float32(x);
  if (detail::should_use_spatial_grid_knn(n, d, options.metric)) {
    if (k <= 256) {
      return detail::spatial_grid_self_knn_cuda(data, n, d, k, options.gpu_device, true);
    }
    return detail::spatial_grid_self_knn(data.data(), n, d, k, options.n_threads, true);
  }
  if (options.metric == DistanceMetric::Cosine) normalize_rows(data, x.rows, x.cols);
  std::vector<int> exclusions(static_cast<std::size_t>(n));
  std::iota(exclusions.begin(), exclusions.end(), 0);
  const detail::NativeKNNResult search = detail::native_cuda_exact_knn_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    options.gpu_device,
    exclusions
  );
  NeighborGraph out;
  out.neighbors = search.neighbors;
  out.index_base = GraphIndexBase::One;
  out.indices = search.indices;
  out.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < out.indices.size(); ++i) {
    if (out.indices[i] >= 0) ++out.indices[i];
    out.distances[i] = detail::native_knn_output_distance(search.distances[i], options.metric);
  }
  return out;
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAKNNGraph_CUDA requires a CUDA build.");
#endif
}

NeighborGraph KODAMAKNNGraph_METAL(MatrixView x, const GraphClusterOptions& options) {
#ifdef KODAMA_ENABLE_METAL
  GraphClusterOptions opts = options;
  opts.backend = Backend::Metal;
  return build_metal_graph(x, opts);
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAKNNGraph_METAL requires an Apple Metal build.");
#endif
}

NeighborGraph KODAMAKNNGraph(MatrixView x, const GraphClusterOptions& options) {
  if (options.backend == Backend::CUDA) return KODAMAKNNGraph_CUDA(x, options);
  if (options.backend == Backend::Metal) return KODAMAKNNGraph_METAL(x, options);
  return KODAMAKNNGraph_CPU(x, options);
}

GraphClusterResult KODAMAGraphCluster_CPU(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  Timer timer;
  GraphClusterOptions opts = options;
  opts.backend = Backend::CPU;
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  return run_cpu_cluster(edges, opts, timer.seconds());
}

GraphClusterResult KODAMAGraphCluster(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  if (options.backend != Backend::CPU) {
    throw std::runtime_error(
      "Random-walk clustering is CPU-only. Construct the graph with the requested "
      "accelerator, then call KODAMAGraphCluster_CPU explicitly."
    );
  }
  return KODAMAGraphCluster_CPU(graph, samples, options);
}

GraphClusterResult KODAMAEmbeddingGraphCluster(MatrixView embedding, const NeighborGraph& graph, const GraphClusterOptions& options) {
  Timer timer;
  if (embedding.rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Embedding has too many rows for graph clustering.");
  }
  const int samples = static_cast<int>(embedding.rows);
  if (samples < 2) throw std::invalid_argument("KODAMAEmbeddingCluster requires at least two rows.");
  GraphClusterOptions opts = options;
  if (opts.backend != Backend::CPU) {
    throw std::runtime_error(
      "Random-walk clustering is CPU-only. Construct the graph with the requested "
      "accelerator, then call KODAMAEmbeddingGraphCluster with Backend::CPU."
    );
  }
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  if (opts.target_clusters > 0) bridge_embedding_components(edges, embedding);
  GraphClusterResult out = run_cpu_cluster(edges, opts, timer.seconds());
  out.runtime_seconds = timer.seconds();
  return out;
}

GraphClusterResult KODAMAEmbeddingCluster(MatrixView embedding, const GraphClusterOptions& options) {
  Timer timer;
  if (options.backend != Backend::CPU) {
    throw std::runtime_error(
      "KODAMAEmbeddingCluster provides CPU random-walk clustering only. Use "
      "KODAMAKNNGraph_CUDA or KODAMAKNNGraph_METAL separately when accelerated "
      "graph construction is required."
    );
  }
  GraphClusterOptions graph_options = options;
  NeighborGraph graph = KODAMAKNNGraph(embedding, graph_options);
  GraphClusterResult out = KODAMAEmbeddingGraphCluster(embedding, graph, options);
  out.runtime_seconds = timer.seconds();
  return out;
}

}  // namespace kodama
