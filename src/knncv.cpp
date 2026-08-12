// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kodama {

namespace {

struct Neighbor {
  double score;
  int label;
};

struct HNSWParameters {
  int m = 32;
  int ef_construction = 200;
  int ef_search = 150;
  int tune_k = 50;
  double target_recall = 0.99;
};

int clamp_int(int value, int fallback, int lo, int hi) {
  const int x = value > 0 ? value : fallback;
  return std::min(hi, std::max(lo, x));
}

double hnsw_target_recall(double target_recall) {
  if (!std::isfinite(target_recall)) return 0.99;
  if (target_recall <= 0.925) return 0.90;
  if (target_recall <= 0.975) return 0.95;
  return 0.99;
}

HNSWParameters tune_hnsw_parameters(int n, int p, int k, DistanceMetric metric, const KNNOptions& options) {
  HNSWParameters out;
  out.tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  out.target_recall = hnsw_target_recall(options.hnsw_target_recall);

  const int tune_k = std::max(1, out.tune_k);
  const bool high_dim = p >= 256;
  const bool large_n = n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = tune_k <= 15;
  const bool large_k = tune_k >= 100;
  const bool non_euclidean = metric == DistanceMetric::Cosine || metric == DistanceMetric::InnerProduct;

  if (very_large_high_dim && large_k) {
    out.m = 48;
    out.ef_construction = 240;
    out.ef_search = std::max(220, 3 * tune_k);
  } else if (non_euclidean && small_k) {
    out.m = 32;
    out.ef_construction = 160;
    out.ef_search = std::max(120, 4 * tune_k);
  } else if (non_euclidean && (large_k || high_dim)) {
    out.m = 48;
    out.ef_construction = 240;
    out.ef_search = std::max(220, 3 * tune_k);
  } else {
    out.m = 32;
    out.ef_construction = 200;
    out.ef_search = std::max(150, 3 * tune_k);
  }

  out.m = clamp_int(options.hnsw_m, out.m, 1, 256);
  out.m = std::min(out.m, std::max(1, n > 1 ? n - 1 : 1));
  out.ef_construction = clamp_int(options.hnsw_ef_construction, out.ef_construction, out.m, 4096);
  out.ef_search = clamp_int(options.hnsw_ef_search, out.ef_search, std::max(k, tune_k), 4096);
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<float> make_search_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
) {
  std::vector<float> out(rows.size() * x.cols, 0.0f);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    double squared_norm = 0.0;
    const int row = rows[r];
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float value = x.value_float(static_cast<std::size_t>(row), j);
      squared_norm += static_cast<double>(value) * static_cast<double>(value);
    }
    const double norm = std::sqrt(squared_norm);
    const double scale = metric == DistanceMetric::Cosine && norm > 0.0 && std::isfinite(norm) ?
      1.0 / norm : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(
        static_cast<double>(x.value_float(static_cast<std::size_t>(row), j)) * scale
      );
    }
  }
  return out;
}
#endif

int majority_vote(const std::vector<Neighbor>& neighbors) {
  std::map<int, std::pair<int, double>> votes;
  for (const auto& nb : neighbors) {
    auto& v = votes[nb.label];
    v.first += 1;
    v.second += nb.score;
  }
  int best_label = votes.begin()->first;
  int best_count = votes.begin()->second.first;
  double best_score = votes.begin()->second.second;
  for (const auto& kv : votes) {
    if (kv.second.first > best_count ||
        (kv.second.first == best_count && kv.second.second > best_score) ||
        (kv.second.first == best_count && kv.second.second == best_score && kv.first < best_label)) {
      best_label = kv.first;
      best_count = kv.second.first;
      best_score = kv.second.second;
    }
  }
  return best_label;
}

std::vector<int> knn_predict_native_hnsw_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  const KNNOptions& options,
  HNSWParameters* used_params
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  k = std::min(k, static_cast<int>(train.size()));
  const HNSWParameters parameters = tune_hnsw_parameters(
    static_cast<int>(train.size()),
    static_cast<int>(x.cols),
    k,
    metric,
    options
  );
  if (used_params != nullptr) *used_params = parameters;

  const std::vector<float> train_x = detail::prepare_native_matrix(x, train, metric);
  const std::vector<float> validation_x = detail::prepare_native_matrix(x, validation, metric);
  const detail::NativeKNNResult search = detail::native_hnsw_search(
    train_x,
    static_cast<int>(train.size()),
    validation_x,
    static_cast<int>(validation.size()),
    static_cast<int>(x.cols),
    k,
    metric,
    detail::NativeHNSWParameters{parameters.m, parameters.ef_construction, parameters.ef_search},
    options.n_threads
  );

  std::vector<int> predicted(validation.size(), labels[static_cast<std::size_t>(train.front())]);
  for (std::size_t query = 0; query < validation.size(); ++query) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(search.neighbors));
    for (int rank = 0; rank < search.neighbors; ++rank) {
      const std::size_t pos = query * static_cast<std::size_t>(search.neighbors) + static_cast<std::size_t>(rank);
      const int local_id = search.indices[pos];
      if (local_id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(detail::native_knn_score(search.distances[pos], metric)),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(local_id)])]
      });
    }
    if (!best.empty()) predicted[query] = majority_vote(best);
  }
  return predicted;
}

#if defined(KODAMA_ENABLE_METAL)
std::vector<int> knn_predict_metal(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  const KNNOptions& options,
  detail::MetalIVFStats* ivf_stats
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<float> train_x = detail::prepare_native_matrix(x, train, metric);
  const std::vector<float> validation_x = detail::prepare_native_matrix(x, validation, metric);
  const detail::NativeKNNResult search = options.index_type == KNNIndexType::MetalIVFFlat ?
    detail::metal_ivf_knn_search(
      train_x,
      static_cast<int>(train.size()),
      validation_x,
      static_cast<int>(validation.size()),
      static_cast<int>(x.cols),
      k,
      metric,
      options.ivf_nlist,
      options.ivf_nprobe,
      {},
      ivf_stats
    ) :
    detail::metal_exact_knn_search(
      train_x,
      static_cast<int>(train.size()),
      validation_x,
      static_cast<int>(validation.size()),
      static_cast<int>(x.cols),
      k,
      metric
    );
  std::vector<int> predicted(validation.size(), labels[static_cast<std::size_t>(train.front())]);
  for (std::size_t query = 0; query < validation.size(); ++query) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(search.neighbors));
    for (int rank = 0; rank < search.neighbors; ++rank) {
      const std::size_t pos = query * static_cast<std::size_t>(search.neighbors) + static_cast<std::size_t>(rank);
      const int local_id = search.indices[pos];
      if (local_id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(detail::native_knn_score(search.distances[pos], metric)),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(local_id)])]
      });
    }
    if (!best.empty()) predicted[query] = majority_vote(best);
  }
  return predicted;
}
#endif

#if defined(KODAMA_ENABLE_CUDA)
std::vector<int> knn_predict_native_cuda(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  int gpu_device,
  int nlist,
  int nprobe,
  bool exact,
  detail::NativeCudaIVFStats* ivf_stats
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<float> train_x = make_search_matrix(x, train, metric);
  const std::vector<float> val_x = make_search_matrix(x, validation, metric);
  const int d = static_cast<int>(x.cols);
  const detail::NativeKNNResult search = exact ?
    detail::native_cuda_exact_knn_search(
      train_x,
      static_cast<int>(train.size()),
      val_x,
      static_cast<int>(validation.size()),
      d,
      k,
      metric,
      gpu_device
    ) :
    detail::native_cuda_ivf_knn_search(
      train_x,
      static_cast<int>(train.size()),
      val_x,
      static_cast<int>(validation.size()),
      d,
      k,
      metric,
      nlist,
      nprobe,
      0.99,
      gpu_device,
      {},
      ivf_stats
    );

  std::vector<int> pred(validation.size(), 0);
  for (std::size_t qi = 0; qi < validation.size(); ++qi) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(search.neighbors));
    for (int j = 0; j < search.neighbors; ++j) {
      const std::size_t offset = qi * static_cast<std::size_t>(search.neighbors) + static_cast<std::size_t>(j);
      const int id = search.indices[offset];
      if (id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(detail::native_knn_score(search.distances[offset], metric)),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(id)])]
      });
    }
    pred[qi] = majority_vote(best);
  }
  return pred;
}
#endif

KNNParametersUsed resolve_knn_parameters(const KNNOptions& options) {
  KNNParametersUsed used;
  used.metric = options.metric;
  used.k = options.k;
  used.ivf_nlist = options.ivf_nlist;
  used.ivf_nprobe = options.ivf_nprobe;
  used.hnsw_m = options.hnsw_m;
  used.hnsw_ef_construction = options.hnsw_ef_construction;
  used.hnsw_ef_search = options.hnsw_ef_search;
  used.hnsw_tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  used.hnsw_target_recall = hnsw_target_recall(options.hnsw_target_recall);
  used.gpu_device = options.gpu_device;
  used.n_threads = options.n_threads;

  if (options.backend == Backend::Metal || options.index_type == KNNIndexType::MetalExact ||
      options.index_type == KNNIndexType::MetalIVFFlat) {
#if defined(KODAMA_ENABLE_METAL)
    used.backend = Backend::Metal;
    used.index_type = options.index_type == KNNIndexType::MetalIVFFlat ?
      KNNIndexType::MetalIVFFlat : KNNIndexType::MetalExact;
#else
    throw std::runtime_error("Metal KNNCV backend was requested but this build was not compiled with KODAMA_ENABLE_METAL.");
#endif
  } else if (options.backend == Backend::CUDA || options.index_type == KNNIndexType::CudaExact ||
             options.index_type == KNNIndexType::CudaIVFFlat) {
#if defined(KODAMA_ENABLE_CUDA)
    used.backend = Backend::CUDA;
    used.index_type = options.index_type == KNNIndexType::CudaExact ?
      KNNIndexType::CudaExact : KNNIndexType::CudaIVFFlat;
#else
    throw std::runtime_error("CUDA KNNCV backend was requested but this build was not compiled with KODAMA_ENABLE_CUDA.");
#endif
  } else {
    used.backend = Backend::CPU;
    used.index_type = options.index_type;
    if (used.index_type == KNNIndexType::CudaExact || used.index_type == KNNIndexType::CudaIVFFlat ||
        used.index_type == KNNIndexType::MetalExact || used.index_type == KNNIndexType::MetalIVFFlat) {
      used.index_type = KNNIndexType::NativeHNSW;
    }
  }
  return used;
}

}  // namespace

KNNCVResult KNNCV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.backend == Backend::CUDA) return KNNCV_CUDA(x, labels, constrain, options);
  if (options.backend == Backend::Metal) return KNNCV_METAL(x, labels, constrain, options);
  return KNNCV_CPU(x, labels, constrain, options);
}

KNNCVResult KNNCV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  detail::validate_inputs(x, labels, constrain);
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  detail::Timer timer;
  KNNCVResult result;
  result.true_labels = labels;
  result.predicted_labels.assign(labels.size(), 0);
  result.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  result.parameters = resolve_knn_parameters(options);

  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<int> fold_pred;
    HNSWParameters hnsw_params;
    fold_pred = knn_predict_native_hnsw_cpu(
      x,
      labels,
      train,
      validation,
      result.parameters.k,
      result.parameters.metric,
      options,
      &hnsw_params
    );
    result.parameters.index_type = KNNIndexType::NativeHNSW;
    result.parameters.hnsw_m = hnsw_params.m;
    result.parameters.hnsw_ef_construction = hnsw_params.ef_construction;
    result.parameters.hnsw_ef_search = hnsw_params.ef_search;
    result.parameters.hnsw_tune_k = hnsw_params.tune_k;
    result.parameters.hnsw_target_recall = hnsw_params.target_recall;
    for (std::size_t i = 0; i < validation.size(); ++i) {
      result.predicted_labels[static_cast<std::size_t>(validation[i])] = fold_pred[i];
    }
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }

  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  return result;
}

KNNCVResult KNNCV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KNNOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  detail::validate_inputs(x, labels, constrain);
  if (cuda_options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  detail::Timer timer;
  KNNCVResult result;
  result.true_labels = labels;
  result.predicted_labels.assign(labels.size(), 0);
  result.fold_assignments = detail::make_folds(labels, constrain, cuda_options.cv);
  result.parameters = resolve_knn_parameters(cuda_options);
  result.parameters.index_type = cuda_options.index_type == KNNIndexType::CudaExact ?
    KNNIndexType::CudaExact : KNNIndexType::CudaIVFFlat;

  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    detail::NativeCudaIVFStats ivf_stats;
    std::vector<int> fold_pred = knn_predict_native_cuda(
      x,
      labels,
      train,
      validation,
      result.parameters.k,
      result.parameters.metric,
      result.parameters.gpu_device,
      cuda_options.ivf_nlist,
      cuda_options.ivf_nprobe,
      result.parameters.index_type == KNNIndexType::CudaExact,
      &ivf_stats
    );
    if (result.parameters.index_type == KNNIndexType::CudaIVFFlat) {
      result.parameters.ivf_nlist = ivf_stats.nlist;
      result.parameters.ivf_nprobe = std::max(result.parameters.ivf_nprobe, ivf_stats.nprobe);
      result.parameters.ivf_pilot_recall = result.parameters.ivf_pilot_recall == 0.0 ?
        ivf_stats.pilot_recall : std::min(result.parameters.ivf_pilot_recall, ivf_stats.pilot_recall);
    }
    for (std::size_t i = 0; i < validation.size(); ++i) {
      result.predicted_labels[static_cast<std::size_t>(validation[i])] = fold_pred[i];
    }
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }

  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  return result;
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("KNNCV_CUDA requires a CUDA build.");
#endif
}

KNNCVResult KNNCV_METAL(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KNNOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  if (metal_options.index_type != KNNIndexType::MetalIVFFlat) {
    metal_options.index_type = KNNIndexType::MetalExact;
  }
  detail::validate_inputs(x, labels, constrain);
  if (metal_options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  detail::Timer timer;
  KNNCVResult result;
  result.true_labels = labels;
  result.predicted_labels.assign(labels.size(), 0);
  result.fold_assignments = detail::make_folds(labels, constrain, metal_options.cv);
  result.parameters = resolve_knn_parameters(metal_options);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    detail::MetalIVFStats ivf_stats;
    const std::vector<int> fold_pred = knn_predict_metal(
      x,
      labels,
      train,
      validation,
      result.parameters.k,
      result.parameters.metric,
      metal_options,
      &ivf_stats
    );
    if (result.parameters.index_type == KNNIndexType::MetalIVFFlat) {
      result.parameters.ivf_nlist = ivf_stats.nlist;
      result.parameters.ivf_nprobe = std::max(result.parameters.ivf_nprobe, ivf_stats.nprobe);
      result.parameters.ivf_pilot_recall = std::min(
        result.parameters.ivf_pilot_recall == 0.0 ? 1.0 : result.parameters.ivf_pilot_recall,
        ivf_stats.pilot_recall
      );
    }
    for (std::size_t i = 0; i < validation.size(); ++i) {
      result.predicted_labels[static_cast<std::size_t>(validation[i])] = fold_pred[i];
    }
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }
  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  return result;
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("KNNCV_METAL requires an Apple Metal build.");
#endif
}

bool MetalAvailable() {
  return detail::metal_backend_available();
}

}  // namespace kodama
