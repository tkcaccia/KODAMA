// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace kodama {

const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::Auto: return "auto";
    case Backend::CPU: return "cpu";
    case Backend::CUDA: return "cuda";
    case Backend::Metal: return "metal";
  }
  return "unknown";
}

const char* to_string(DistanceMetric metric) {
  switch (metric) {
    case DistanceMetric::Cosine: return "cosine";
    case DistanceMetric::InnerProduct: return "inner_product";
    case DistanceMetric::Euclidean: return "euclidean";
  }
  return "unknown";
}

const char* to_string(KNNIndexType index_type) {
  switch (index_type) {
    case KNNIndexType::PrecomputedGraph: return "precomputed_graph";
    case KNNIndexType::NativeHNSW: return "native_hnsw";
    case KNNIndexType::CudaExact: return "cuda_exact";
    case KNNIndexType::CudaIVFFlat: return "cuda_ivf_flat";
    case KNNIndexType::MetalExact: return "metal_exact";
    case KNNIndexType::MetalIVFFlat: return "metal_ivf_flat";
  }
  return "unknown";
}

const char* to_string(PLSMode mode) {
  switch (mode) {
    case PLSMode::PLS_DA: return "pls_da";
    case PLSMode::PLS_LDA: return "pls_lda";
  }
  return "unknown";
}

const char* to_string(CoreClassifier classifier) {
  switch (classifier) {
    case CoreClassifier::PLS_LDA: return "pls_lda";
    case CoreClassifier::KNN: return "knn";
  }
  return "unknown";
}

const char* to_string(GraphWeightType weight_type) {
  switch (weight_type) {
    case GraphWeightType::SNN: return "snn";
    case GraphWeightType::Distance: return "distance";
    case GraphWeightType::Adaptive: return "adaptive";
    case GraphWeightType::Binary: return "binary";
  }
  return "unknown";
}

const char* to_string(GraphFeatureMode mode) {
  switch (mode) {
    case GraphFeatureMode::LaplacianSelfTuning: return "laplacian_self_tuning";
  }
  return "unknown";
}

}  // namespace kodama

namespace kodama::detail {

void validate_inputs(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain) {
  if (x.data == nullptr) {
    throw std::invalid_argument("MatrixView data pointer is null.");
  }
  if (x.rows == 0 || x.cols == 0) {
    throw std::invalid_argument("Input matrix must have at least one row and one column.");
  }
  if (labels.size() != x.rows) {
    throw std::invalid_argument("labels size must match number of rows.");
  }
  if (!constrain.empty() && constrain.size() != x.rows) {
    throw std::invalid_argument("constrain size must be zero or match number of rows.");
  }
}

std::vector<int> unique_labels(const std::vector<int>& labels) {
  std::vector<int> out = labels;
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<int> make_folds(
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const FoldOptions& options
) {
  if (options.folds < 2) {
    throw std::invalid_argument("Number of folds must be at least 2.");
  }
  const int n = static_cast<int>(labels.size());
  std::mt19937_64 rng(options.seed);
  if (constrain.empty()) {
    std::vector<int> folds(static_cast<std::size_t>(n), 0);
    if (options.stratified) {
      std::map<int, std::vector<int>> by_label;
      for (int i = 0; i < n; ++i) {
        by_label[labels[static_cast<std::size_t>(i)]].push_back(i);
      }
      std::vector<int> fold_load(static_cast<std::size_t>(options.folds), 0);
      for (auto& kv : by_label) {
        auto& ids = kv.second;
        std::shuffle(ids.begin(), ids.end(), rng);
        std::fill(fold_load.begin(), fold_load.end(), 0);
        for (int idx : ids) {
          const int fold = static_cast<int>(
            std::min_element(fold_load.begin(), fold_load.end()) - fold_load.begin()
          );
          folds[static_cast<std::size_t>(idx)] = fold;
          fold_load[static_cast<std::size_t>(fold)] += 1;
        }
      }
    } else {
      std::vector<int> order(static_cast<std::size_t>(n));
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      for (std::size_t i = 0; i < order.size(); ++i) {
        folds[static_cast<std::size_t>(order[i])] =
          static_cast<int>(i % static_cast<std::size_t>(options.folds));
      }
    }
    return folds;
  }

  struct Group {
    int id = 0;
    int label = 0;
    int size = 0;
  };
  std::vector<Group> groups;
  std::map<int, std::vector<int>> members;
  for (int i = 0; i < n; ++i) {
    members[constrain[static_cast<std::size_t>(i)]].push_back(i);
  }

  groups.reserve(members.size());
  for (const auto& kv : members) {
    std::map<int, int> counts;
    for (int idx : kv.second) counts[labels[static_cast<std::size_t>(idx)]]++;
    int best_label = counts.begin()->first;
    int best_count = counts.begin()->second;
    for (const auto& lc : counts) {
      if (lc.second > best_count) {
        best_label = lc.first;
        best_count = lc.second;
      }
    }
    groups.push_back(Group{kv.first, best_label, static_cast<int>(kv.second.size())});
  }

  std::vector<int> group_fold(groups.size(), 0);
  std::unordered_map<int, std::size_t> group_pos;
  group_pos.reserve(groups.size());
  for (std::size_t i = 0; i < groups.size(); ++i) group_pos[groups[i].id] = i;

  if (options.stratified) {
    std::map<int, std::vector<std::size_t>> by_label;
    for (std::size_t i = 0; i < groups.size(); ++i) {
      by_label[groups[i].label].push_back(i);
    }
    std::vector<int> fold_load(static_cast<std::size_t>(options.folds), 0);
    for (auto& kv : by_label) {
      auto& ids = kv.second;
      std::shuffle(ids.begin(), ids.end(), rng);
      std::fill(fold_load.begin(), fold_load.end(), 0);
      for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        const int fold = static_cast<int>(
          std::min_element(fold_load.begin(), fold_load.end()) - fold_load.begin()
        );
        group_fold[ids[pos]] = fold;
        fold_load[static_cast<std::size_t>(fold)] += groups[ids[pos]].size;
      }
    }
  } else {
    std::vector<std::size_t> order(groups.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t i = 0; i < order.size(); ++i) {
      group_fold[order[i]] = static_cast<int>(i % static_cast<std::size_t>(options.folds));
    }
  }

  std::vector<int> folds(static_cast<std::size_t>(n), 0);
  for (const auto& kv : members) {
    const int f = group_fold[group_pos[kv.first]];
    for (int idx : kv.second) folds[static_cast<std::size_t>(idx)] = f;
  }
  return folds;
}

double accuracy(const std::vector<int>& truth, const std::vector<int>& pred) {
  if (truth.size() != pred.size()) throw std::invalid_argument("truth and pred size mismatch.");
  if (truth.empty()) return 0.0;
  std::size_t ok = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    ok += truth[i] == pred[i] ? 1u : 0u;
  }
  return static_cast<double>(ok) / static_cast<double>(truth.size());
}

double accuracy_on_indices(
  const std::vector<int>& truth,
  const std::vector<int>& pred,
  const std::vector<int>& indices
) {
  if (indices.empty()) return 0.0;
  std::size_t ok = 0;
  for (int idx : indices) {
    const auto u = static_cast<std::size_t>(idx);
    ok += truth[u] == pred[u] ? 1u : 0u;
  }
  return static_cast<double>(ok) / static_cast<double>(indices.size());
}

ConfusionMatrix make_confusion(const std::vector<int>& truth, const std::vector<int>& pred) {
  const std::vector<int> labels = unique_labels(truth);
  std::unordered_map<int, std::size_t> pos;
  for (std::size_t i = 0; i < labels.size(); ++i) pos[labels[i]] = i;
  ConfusionMatrix cm;
  cm.labels = labels;
  cm.n_labels = labels.size();
  cm.counts.assign(labels.size() * labels.size(), 0);
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const auto ti = pos.at(truth[i]);
    const auto pi_it = pos.find(pred[i]);
    if (pi_it == pos.end()) continue;
    cm.counts[ti * labels.size() + pi_it->second]++;
  }
  return cm;
}

double peak_memory_mb() {
#if defined(__linux__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#if defined(__APPLE__)
  // Darwin reports ru_maxrss in bytes.
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  // Linux reports ru_maxrss in KiB.
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
  return 0.0;
#endif
}

std::vector<int> indices_where_fold(const std::vector<int>& folds, int fold, bool equal) {
  std::vector<int> out;
  for (std::size_t i = 0; i < folds.size(); ++i) {
    if ((folds[i] == fold) == equal) out.push_back(static_cast<int>(i));
  }
  return out;
}

std::vector<int> sorted_unique_folds(const std::vector<int>& folds) {
  std::vector<int> out = folds;
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace kodama::detail
