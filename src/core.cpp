// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#if defined(KODAMA_DISABLE_NATIVE_PROGRESS)
#include <fstream>
#else
#include <iostream>
#endif
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace kodama {

EvolutionPolicy EvolutionPolicy::standard() {
  return EvolutionPolicy{};
}

EvolutionPolicy EvolutionPolicy::from_name(const std::string& name) {
  EvolutionPolicy policy = standard();
  if (name == "full") return policy;
  if (name == "no_prediction_guidance") {
    policy.prediction_guidance = false;
  } else if (name == "fixed_proposal_budget") {
    policy.adaptive_proposal_size = false;
  } else if (name == "no_transition_proposal") {
    policy.transition_proposal = false;
  } else if (name == "greedy_acceptance") {
    policy.stochastic_acceptance = false;
  } else if (name == "raw_cv_score") {
    policy.diversity_multiplier = false;
  } else if (name == "no_pls_transition_coarsening") {
    policy.pls_transition_coarsening = false;
  } else if (name == "no_pls_fragmentation_penalty") {
    policy.pls_fragmentation_penalty = false;
  } else {
    throw std::invalid_argument(
      "Unknown evolution policy '" + name +
      "'. Expected full, no_prediction_guidance, fixed_proposal_budget, "
      "no_transition_proposal, greedy_acceptance, raw_cv_score, "
      "no_pls_transition_coarsening, or no_pls_fragmentation_penalty."
    );
  }
  return policy;
}

namespace {

std::mutex& core_progress_mutex() {
  static std::mutex mutex;
  return mutex;
}

void emit_core_progress(const std::string& message) {
#if defined(KODAMA_DISABLE_NATIVE_PROGRESS)
  const char* path = std::getenv("KODAMA_PROGRESS_FILE");
  if (path == nullptr || path[0] == '\0') return;
  std::ofstream output(path, std::ios::out | std::ios::app);
  if (output) output << message << '\n';
#else
  std::cerr << message << std::endl;
#endif
}

void report_core_progress(
  const CoreOptions& options,
  int cycle,
  double accuracy,
  double score,
  int active_classes,
  bool accepted,
  double elapsed_seconds
) {
  if (!options.progress) return;
  const int interval = std::max(1, options.cycles / 20);
  if (cycle != 0 && cycle != options.cycles && cycle % interval != 0) return;
  std::lock_guard<std::mutex> lock(core_progress_mutex());
  std::string message = "[kodama]";
  if (options.progress_run > 0) {
    message += " M " + std::to_string(options.progress_run);
    if (options.progress_runs > 0) message += "/" + std::to_string(options.progress_runs);
  }
  message += " Tcycle " + std::to_string(cycle) + "/" + std::to_string(options.cycles);
  message += " acc=" + std::to_string(accuracy);
  message += " score=" + std::to_string(score);
  message += " classes=" + std::to_string(active_classes);
  message += std::string(" accepted=") + (accepted ? "yes" : "no");
  message += " elapsed=" + std::to_string(elapsed_seconds) + "s";
  emit_core_progress(message);
}

struct CVPrediction {
  std::vector<int> predicted;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
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

HNSWParameters tune_hnsw_parameters(int n, int p, int k, const KNNOptions& options) {
  HNSWParameters out;
  out.tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  out.target_recall = hnsw_target_recall(options.hnsw_target_recall);
  const int tune_k = std::max(1, out.tune_k);
  const bool high_dim = p >= 256;
  const bool large_n = n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = tune_k <= 15;
  const bool large_k = tune_k >= 100;

  if (small_k) {
    out.m = 32;
    out.ef_construction = 160;
    out.ef_search = std::max(120, 4 * tune_k);
  } else if ((very_large_high_dim && large_k) || (large_k || high_dim)) {
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

std::vector<float> make_search_matrix(MatrixView x, DistanceMetric metric) {
  std::vector<float> out(x.rows * x.cols, 0.0f);
  if (metric != DistanceMetric::Cosine) {
    if (x.value_type == MatrixValueType::Float32) {
      const float* data = static_cast<const float*>(x.data);
      std::copy_n(data, x.rows * x.cols, out.data());
    } else {
      for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t j = 0; j < x.cols; ++j) {
          out[r * x.cols + j] = x.value_float(r, j);
        }
      }
    }
    return out;
  }

  for (std::size_t r = 0; r < x.rows; ++r) {
    long double ss = 0.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float v = x.value_float(r, j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    const double scale = metric == DistanceMetric::Cosine && n > 0.0 && std::isfinite(n) ? 1.0 / n : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(static_cast<double>(x.value_float(r, j)) * scale);
    }
  }
  return out;
}

void gather_search_rows(
  const std::vector<float>& all_x,
  std::size_t cols,
  const std::vector<int>& rows,
  std::vector<float>& out
) {
  out.resize(rows.size() * cols);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const int row = rows[r];
    std::copy_n(all_x.data() + static_cast<std::size_t>(row) * cols, cols, out.data() + r * cols);
  }
}

std::vector<int> normalized_constrain(const std::vector<int>& constrain, std::size_t n) {
  if (!constrain.empty()) return constrain;
  std::vector<int> out(n);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

std::vector<int> normalized_fixed(const std::vector<int>& fixed, std::size_t n) {
  if (fixed.empty()) return std::vector<int>(n, 0);
  if (fixed.size() != n) throw std::invalid_argument("fixed size must be zero or match number of rows.");
  return fixed;
}

struct PrecomputedKNNFold {
  int fold = 0;
  std::vector<int> validation;
  std::vector<int> neighbor_rows;
  std::vector<float> scores;
  std::vector<int> empty_validation_rows;
  int k = 0;
};

struct PrecomputedKNN {
  std::vector<int> folds;
  std::vector<PrecomputedKNNFold> fold_data;
  KNNParametersUsed parameters;
#if defined(KODAMA_ENABLE_CUDA)
  std::shared_ptr<detail::NativeCudaKNNVoteGraph> cuda_vote_graph;
#endif
#if defined(KODAMA_ENABLE_METAL)
  std::shared_ptr<detail::NativeMetalKNNVoteGraph> metal_vote_graph;
#endif
};

struct KNNPredictionScratch {
  std::unordered_map<int, int> label_to_code;
  std::vector<int> dense_label_to_code;
  std::vector<int> label_values;
  std::vector<int> label_codes;
  std::vector<int> accelerator_prediction_codes;
  std::vector<int> vote_counts;
  std::vector<int> fallback_counts;
  std::vector<double> vote_scores;
  std::vector<int> touched;
  int dense_label_offset = 0;
  bool label_map_initialized = false;
};

void record_empty_validation_rows(PrecomputedKNNFold& fold) {
  fold.empty_validation_rows.clear();
  const std::size_t width = static_cast<std::size_t>(fold.k);
  for (std::size_t query = 0; query < fold.validation.size(); ++query) {
    const std::size_t offset = query * width;
    bool has_neighbor = false;
    for (std::size_t j = 0; j < width; ++j) {
      if (fold.neighbor_rows[offset + j] >= 0) {
        has_neighbor = true;
        break;
      }
    }
    if (!has_neighbor) fold.empty_validation_rows.push_back(fold.validation[query]);
  }
}

void apply_training_fold_fallbacks(
  const PrecomputedKNN& precomputed,
  KNNPredictionScratch& scratch,
  std::vector<int>& predicted
) {
  if (scratch.fallback_counts.size() < scratch.label_values.size()) {
    scratch.fallback_counts.resize(scratch.label_values.size(), 0);
  }
  std::fill(scratch.fallback_counts.begin(), scratch.fallback_counts.end(), 0);
  for (int code : scratch.label_codes) {
    ++scratch.fallback_counts[static_cast<std::size_t>(code)];
  }

  for (const PrecomputedKNNFold& fold : precomputed.fold_data) {
    for (int row : fold.validation) {
      --scratch.fallback_counts[static_cast<std::size_t>(scratch.label_codes[static_cast<std::size_t>(row)])];
    }
    int fallback_label = scratch.label_values.empty() ? 0 : scratch.label_values.front();
    int fallback_count = -1;
    for (std::size_t code = 0; code < scratch.label_values.size(); ++code) {
      const int label = scratch.label_values[code];
      const int count = scratch.fallback_counts[code];
      if (count > fallback_count || (count == fallback_count && label < fallback_label)) {
        fallback_label = label;
        fallback_count = count;
      }
    }

    for (int row : fold.empty_validation_rows) {
      predicted[static_cast<std::size_t>(row)] = fallback_label;
    }
    for (int row : fold.validation) {
      ++scratch.fallback_counts[static_cast<std::size_t>(scratch.label_codes[static_cast<std::size_t>(row)])];
    }
  }
}

void flatten_precomputed_knn(
  const PrecomputedKNN& precomputed,
  std::vector<int>& neighbor_rows,
  std::vector<float>& scores
) {
  const int samples = static_cast<int>(precomputed.folds.size());
  const int width = std::max(1, precomputed.parameters.k);
  const std::size_t items =
    static_cast<std::size_t>(samples) * static_cast<std::size_t>(width);
  neighbor_rows.assign(items, -1);
  scores.assign(items, -std::numeric_limits<float>::infinity());
  for (const PrecomputedKNNFold& fold : precomputed.fold_data) {
    for (std::size_t query = 0; query < fold.validation.size(); ++query) {
      const int output_row = fold.validation[query];
      if (output_row < 0 || output_row >= samples) continue;
      const std::size_t source =
        query * static_cast<std::size_t>(fold.k);
      const std::size_t destination =
        static_cast<std::size_t>(output_row) * static_cast<std::size_t>(width);
      const int count = std::min(width, fold.k);
      std::copy_n(
        fold.neighbor_rows.data() + source,
        count,
        neighbor_rows.data() + destination
      );
      std::copy_n(
        fold.scores.data() + source,
        count,
        scores.data() + destination
      );
    }
  }
}

void attach_resident_knn_vote_graph(PrecomputedKNN& precomputed) {
  std::vector<int> neighbor_rows;
  std::vector<float> scores;
  flatten_precomputed_knn(precomputed, neighbor_rows, scores);
  const int samples = static_cast<int>(precomputed.folds.size());
  const int neighbors = std::max(1, precomputed.parameters.k);
#if defined(KODAMA_ENABLE_CUDA)
  if (precomputed.parameters.backend == Backend::CUDA) {
    precomputed.cuda_vote_graph =
      std::make_shared<detail::NativeCudaKNNVoteGraph>(
        detail::native_cuda_build_knn_vote_graph(
          neighbor_rows,
          scores,
          samples,
          neighbors,
          precomputed.parameters.gpu_device
        )
      );
    for (PrecomputedKNNFold& fold : precomputed.fold_data) {
      std::vector<int>().swap(fold.neighbor_rows);
      std::vector<float>().swap(fold.scores);
    }
    return;
  }
#endif
#if defined(KODAMA_ENABLE_METAL)
  if (precomputed.parameters.backend == Backend::Metal) {
    precomputed.metal_vote_graph =
      std::make_shared<detail::NativeMetalKNNVoteGraph>(
        detail::metal_build_knn_vote_graph(
          neighbor_rows,
          scores,
          samples,
          neighbors
        )
      );
    for (PrecomputedKNNFold& fold : precomputed.fold_data) {
      std::vector<int>().swap(fold.neighbor_rows);
      std::vector<float>().swap(fold.scores);
    }
  }
#else
  (void)samples;
  (void)neighbors;
#endif
}

void initialize_knn_label_map(KNNPredictionScratch& scratch, const std::vector<int>& labels) {
  scratch.label_to_code.clear();
  scratch.label_values = labels;
  std::sort(scratch.label_values.begin(), scratch.label_values.end());
  scratch.label_values.erase(
    std::unique(scratch.label_values.begin(), scratch.label_values.end()),
    scratch.label_values.end()
  );
  scratch.label_to_code.reserve(labels.size());
  for (std::size_t code = 0; code < scratch.label_values.size(); ++code) {
    const int label = scratch.label_values[code];
    scratch.label_to_code.emplace(label, code);
  }
  scratch.dense_label_to_code.clear();
  if (!scratch.label_values.empty()) {
    const std::int64_t minimum = scratch.label_values.front();
    const std::int64_t maximum = scratch.label_values.back();
    const std::uint64_t range = static_cast<std::uint64_t>(maximum - minimum) + 1u;
    if (range <= labels.size()) {
      scratch.dense_label_offset = static_cast<int>(minimum);
      scratch.dense_label_to_code.assign(static_cast<std::size_t>(range), -1);
      for (std::size_t code = 0; code < scratch.label_values.size(); ++code) {
        scratch.dense_label_to_code[
          static_cast<std::size_t>(scratch.label_values[code] - scratch.dense_label_offset)
        ] = static_cast<int>(code);
      }
    }
  }
  scratch.label_map_initialized = true;
}

int knn_label_code(const KNNPredictionScratch& scratch, int label) {
  if (!scratch.dense_label_to_code.empty()) {
    const std::int64_t offset =
      static_cast<std::int64_t>(label) - scratch.dense_label_offset;
    if (offset >= 0 &&
        static_cast<std::size_t>(offset) < scratch.dense_label_to_code.size()) {
      return scratch.dense_label_to_code[static_cast<std::size_t>(offset)];
    }
    return -1;
  }
  const auto it = scratch.label_to_code.find(label);
  return it == scratch.label_to_code.end() ? -1 : it->second;
}

KNNParametersUsed resolve_core_knn_parameters(const KNNOptions& options) {
  KNNParametersUsed used;
  used.backend = options.backend == Backend::Auto ? Backend::CPU : options.backend;
  used.index_type = options.backend == Backend::Metal ?
    (options.index_type == KNNIndexType::MetalIVFFlat ? KNNIndexType::MetalIVFFlat : KNNIndexType::MetalExact) :
    options.index_type;
  if (used.backend == Backend::CPU &&
      (used.index_type == KNNIndexType::CudaExact || used.index_type == KNNIndexType::CudaIVFFlat ||
       used.index_type == KNNIndexType::MetalExact ||
       used.index_type == KNNIndexType::MetalIVFFlat)) {
    used.index_type = KNNIndexType::NativeHNSW;
  }
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
  return used;
}

PrecomputedKNN precompute_knn_cv_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  if (options.backend == Backend::CUDA) {
    throw std::invalid_argument("CoreKNN CPU/Metal precomputation cannot be used with the CUDA backend.");
  }

  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const std::vector<float> all_x = make_search_matrix(x, precomputed.parameters.metric);
  std::vector<float> train_x;
  std::vector<float> val_x;
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    const std::vector<int> train = detail::indices_where_fold(precomputed.folds, fold, false);
    if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");

    const int k = std::min(precomputed.parameters.k, static_cast<int>(train.size()));
    gather_search_rows(all_x, x.cols, train, train_x);
    gather_search_rows(all_x, x.cols, validation, val_x);
    const int d = static_cast<int>(x.cols);
    std::vector<int> idx(validation.size() * static_cast<std::size_t>(k), -1);
    std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
    if (precomputed.parameters.index_type == KNNIndexType::MetalExact ||
        precomputed.parameters.index_type == KNNIndexType::MetalIVFFlat) {
#if defined(KODAMA_ENABLE_METAL)
      detail::MetalIVFStats ivf_stats;
      const detail::NativeKNNResult metal =
        precomputed.parameters.index_type == KNNIndexType::MetalIVFFlat ?
          detail::metal_ivf_knn_search(
            train_x,
            static_cast<int>(train.size()),
            val_x,
            static_cast<int>(validation.size()),
            d,
            k,
            precomputed.parameters.metric,
            options.ivf_nlist,
            options.ivf_nprobe,
            {},
            &ivf_stats
          ) :
          detail::metal_exact_knn_search(
            train_x,
            static_cast<int>(train.size()),
            val_x,
            static_cast<int>(validation.size()),
            d,
            k,
            precomputed.parameters.metric
          );
      if (precomputed.parameters.index_type == KNNIndexType::MetalIVFFlat) {
        precomputed.parameters.ivf_nlist = ivf_stats.nlist;
        precomputed.parameters.ivf_nprobe = std::max(precomputed.parameters.ivf_nprobe, ivf_stats.nprobe);
        precomputed.parameters.ivf_pilot_recall = std::min(
          precomputed.parameters.ivf_pilot_recall == 0.0 ? 1.0 : precomputed.parameters.ivf_pilot_recall,
          ivf_stats.pilot_recall
        );
      }
      idx = metal.indices;
      scores.resize(metal.distances.size());
      for (std::size_t i = 0; i < scores.size(); ++i) {
        scores[i] = detail::native_knn_score(metal.distances[i], precomputed.parameters.metric);
      }
#else
      throw std::runtime_error("CoreKNN Metal backend requires KODAMA_ENABLE_METAL.");
#endif
    } else {
      const HNSWParameters hnsw = tune_hnsw_parameters(static_cast<int>(train.size()), d, k, options);
      precomputed.parameters.index_type = KNNIndexType::NativeHNSW;
      precomputed.parameters.hnsw_m = hnsw.m;
      precomputed.parameters.hnsw_ef_construction = hnsw.ef_construction;
      precomputed.parameters.hnsw_ef_search = hnsw.ef_search;
      precomputed.parameters.hnsw_tune_k = hnsw.tune_k;
      precomputed.parameters.hnsw_target_recall = hnsw.target_recall;
      const detail::NativeKNNResult native = detail::native_hnsw_search(
        train_x,
        static_cast<int>(train.size()),
        val_x,
        static_cast<int>(validation.size()),
        d,
        k,
        precomputed.parameters.metric,
        detail::NativeHNSWParameters{hnsw.m, hnsw.ef_construction, hnsw.ef_search},
        precomputed.parameters.n_threads
      );
      idx = native.indices;
      scores.resize(native.distances.size());
      for (std::size_t i = 0; i < scores.size(); ++i) {
        scores[i] = detail::native_knn_score(native.distances[i], precomputed.parameters.metric);
      }
    }

    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.scores = std::move(scores);
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      for (int j = 0; j < k; ++j) {
        const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
        if (id >= 0) {
          fold_data.neighbor_rows[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] =
            train[static_cast<std::size_t>(id)];
        }
      }
    }
    record_empty_validation_rows(fold_data);
    precomputed.fold_data.push_back(std::move(fold_data));
  }
  if (precomputed.parameters.backend == Backend::Metal) {
    attach_resident_knn_vote_graph(precomputed);
  }
  return precomputed;
}

NeighborGraph normalize_graph_indices(const NeighborGraph& graph, int samples) {
  if (samples < 2) throw std::invalid_argument("NeighborGraph samples must be at least 2.");
  if (graph.neighbors <= 0) throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  const std::size_t expected = static_cast<std::size_t>(samples) * static_cast<std::size_t>(graph.neighbors);
  if (graph.indices.size() != expected || graph.distances.size() != expected) {
    throw std::invalid_argument("NeighborGraph indices/distances size must equal samples * neighbors.");
  }

  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int value : graph.indices) {
    if (value < 0) continue;
    min_index = std::min(min_index, value);
    max_index = std::max(max_index, value);
  }
  const bool one_based = graph.index_base == GraphIndexBase::One ||
    (graph.index_base == GraphIndexBase::Auto && min_index >= 1 && max_index <= samples);

  NeighborGraph out;
  out.neighbors = graph.neighbors;
  out.index_base = GraphIndexBase::Zero;
  out.indices.resize(graph.indices.size(), -1);
  out.distances.resize(graph.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < graph.indices.size(); ++i) {
    int id = graph.indices[i];
    if (one_based && id > 0) --id;
    if (id < 0 || id >= samples) {
      out.indices[i] = -1;
      out.distances[i] = std::numeric_limits<float>::infinity();
      continue;
    }
    out.indices[i] = id;
    const float d = graph.distances[i];
    out.distances[i] = std::isfinite(d) && d >= 0.0f ? d : std::numeric_limits<float>::infinity();
  }
  return out;
}

PrecomputedKNN precompute_knn_cv_graph(
  const NeighborGraph& input_graph,
  int samples,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  if (labels.size() != static_cast<std::size_t>(samples)) {
    throw std::invalid_argument("labels size must match NeighborGraph samples.");
  }
  if (!constrain.empty() && constrain.size() != labels.size()) {
    throw std::invalid_argument("constrain size must be zero or match NeighborGraph samples.");
  }

  detail::Timer timer;
  const NeighborGraph graph = normalize_graph_indices(input_graph, samples);
  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.parameters.index_type = KNNIndexType::PrecomputedGraph;
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const int k = std::max(1, std::min(options.k, graph.neighbors));

  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    if (validation.empty()) continue;
    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    fold_data.scores.assign(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());

    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      const int row = validation[qi];
      int out_col = 0;
      const std::size_t in_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(graph.neighbors);
      const std::size_t out_base = qi * static_cast<std::size_t>(k);
      for (int j = 0; j < graph.neighbors && out_col < k; ++j) {
        const std::size_t in_offset = in_base + static_cast<std::size_t>(j);
        const int nb = graph.indices[in_offset];
        if (nb < 0 || nb >= samples || nb == row) continue;
        if (precomputed.folds[static_cast<std::size_t>(nb)] == fold) continue;
        const float distance = graph.distances[in_offset];
        if (!std::isfinite(distance)) continue;
        const std::size_t out_offset = out_base + static_cast<std::size_t>(out_col);
        fold_data.neighbor_rows[out_offset] = nb;
        const float d = std::max(0.0f, distance);
        if (precomputed.parameters.metric == DistanceMetric::Euclidean) {
          fold_data.scores[out_offset] = -(d * d);
        } else {
          fold_data.scores[out_offset] = 1.0f - d;
        }
        ++out_col;
      }
    }
    record_empty_validation_rows(fold_data);
    precomputed.fold_data.push_back(std::move(fold_data));
  }

  precomputed.parameters.k = k;
  if (precomputed.parameters.backend == Backend::CUDA ||
      precomputed.parameters.backend == Backend::Metal) {
    attach_resident_knn_vote_graph(precomputed);
  }
  (void)timer;
  return precomputed;
}

#if defined(KODAMA_ENABLE_CUDA)
PrecomputedKNN precompute_knn_cv_cuda(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");

  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.parameters.backend = Backend::CUDA;
  precomputed.parameters.index_type = options.index_type == KNNIndexType::CudaExact ?
    KNNIndexType::CudaExact : KNNIndexType::CudaIVFFlat;
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const std::vector<float> all_x = make_search_matrix(x, precomputed.parameters.metric);
  std::vector<float> train_x;
  std::vector<float> val_x;
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    const std::vector<int> train = detail::indices_where_fold(precomputed.folds, fold, false);
    if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");

    const int k = std::min(precomputed.parameters.k, static_cast<int>(train.size()));
    gather_search_rows(all_x, x.cols, train, train_x);
    gather_search_rows(all_x, x.cols, validation, val_x);
    const int d = static_cast<int>(x.cols);
    detail::NativeCudaIVFStats ivf_stats;
    const detail::NativeKNNResult search = precomputed.parameters.index_type == KNNIndexType::CudaExact ?
      detail::native_cuda_exact_knn_search(
        train_x,
        static_cast<int>(train.size()),
        val_x,
        static_cast<int>(validation.size()),
        d,
        k,
        precomputed.parameters.metric,
        precomputed.parameters.gpu_device
      ) :
      detail::native_cuda_ivf_knn_search(
        train_x,
        static_cast<int>(train.size()),
        val_x,
        static_cast<int>(validation.size()),
        d,
        k,
        precomputed.parameters.metric,
        options.ivf_nlist,
        options.ivf_nprobe,
        precomputed.parameters.hnsw_target_recall,
        precomputed.parameters.gpu_device,
        {},
        &ivf_stats
      );
    if (precomputed.parameters.index_type == KNNIndexType::CudaIVFFlat) {
      precomputed.parameters.ivf_nlist = ivf_stats.nlist;
      precomputed.parameters.ivf_nprobe = std::max(precomputed.parameters.ivf_nprobe, ivf_stats.nprobe);
      precomputed.parameters.ivf_pilot_recall = precomputed.parameters.ivf_pilot_recall == 0.0 ?
        ivf_stats.pilot_recall : std::min(precomputed.parameters.ivf_pilot_recall, ivf_stats.pilot_recall);
    }
    const std::vector<int>& idx = search.indices;
    std::vector<float> scores(search.distances.size(), -std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < scores.size(); ++i) {
      scores[i] = detail::native_knn_score(search.distances[i], precomputed.parameters.metric);
    }

    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.scores = std::move(scores);
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      for (int j = 0; j < k; ++j) {
        const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
        if (id >= 0) {
          fold_data.neighbor_rows[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] =
            train[static_cast<std::size_t>(id)];
        }
      }
    }
    record_empty_validation_rows(fold_data);
    precomputed.fold_data.push_back(std::move(fold_data));
  }
  attach_resident_knn_vote_graph(precomputed);
  return precomputed;
}
#endif

CVPrediction predict_precomputed_knn(
  const PrecomputedKNN& precomputed,
  const std::vector<int>& labels,
  KNNPredictionScratch& scratch
) {
  CVPrediction out;
  out.predicted.assign(labels.size(), 0);
  if (!scratch.label_map_initialized) {
    initialize_knn_label_map(scratch, labels);
  }
  scratch.label_codes.resize(labels.size());
  for (;;) {
    bool rebuilt = false;
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const int label = labels[i];
      const int code = knn_label_code(scratch, label);
      if (code < 0) {
        initialize_knn_label_map(scratch, labels);
        rebuilt = true;
        break;
      }
      scratch.label_codes[i] = code;
    }
    if (!rebuilt) break;
  }

  if (scratch.vote_counts.size() < scratch.label_values.size()) {
    scratch.vote_counts.resize(scratch.label_values.size(), 0);
  }
  if (scratch.fallback_counts.size() < scratch.label_values.size()) {
    scratch.fallback_counts.resize(scratch.label_values.size(), 0);
  }
  if (scratch.vote_scores.size() < scratch.label_values.size()) {
    scratch.vote_scores.resize(scratch.label_values.size(), 0.0);
  }
  std::fill(scratch.fallback_counts.begin(), scratch.fallback_counts.end(), 0);
  for (int code : scratch.label_codes) {
    scratch.fallback_counts[static_cast<std::size_t>(code)]++;
  }
  int fallback_label = scratch.label_values.empty() ? 0 : scratch.label_values.front();
  int fallback_count = -1;
  for (std::size_t code = 0; code < scratch.label_values.size(); ++code) {
    const int label = scratch.label_values[code];
    const int count = scratch.fallback_counts[code];
    if (count > fallback_count || (count == fallback_count && label < fallback_label)) {
      fallback_label = label;
      fallback_count = count;
    }
    scratch.fallback_counts[code] = 0;
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (precomputed.cuda_vote_graph && precomputed.cuda_vote_graph->valid()) {
    const int fallback_code = knn_label_code(scratch, fallback_label);
    detail::native_cuda_knn_vote_predict_into(
      *precomputed.cuda_vote_graph,
      scratch.label_codes,
      fallback_code,
      scratch.accelerator_prediction_codes
    );
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const int code = scratch.accelerator_prediction_codes[i];
      out.predicted[i] = code >= 0 && static_cast<std::size_t>(code) < scratch.label_values.size() ?
        scratch.label_values[static_cast<std::size_t>(code)] : fallback_label;
    }
    apply_training_fold_fallbacks(precomputed, scratch, out.predicted);
    return out;
  }
#endif
#if defined(KODAMA_ENABLE_METAL)
  if (precomputed.metal_vote_graph && precomputed.metal_vote_graph->valid()) {
    const int fallback_code = knn_label_code(scratch, fallback_label);
    detail::metal_knn_vote_predict_into(
      *precomputed.metal_vote_graph,
      scratch.label_codes,
      fallback_code,
      scratch.accelerator_prediction_codes
    );
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const int code = scratch.accelerator_prediction_codes[i];
      out.predicted[i] = code >= 0 && static_cast<std::size_t>(code) < scratch.label_values.size() ?
        scratch.label_values[static_cast<std::size_t>(code)] : fallback_label;
    }
    apply_training_fold_fallbacks(precomputed, scratch, out.predicted);
    return out;
  }
#endif
  scratch.touched.reserve(static_cast<std::size_t>(precomputed.parameters.k));
  for (const PrecomputedKNNFold& fold : precomputed.fold_data) {
    const std::size_t k = static_cast<std::size_t>(fold.k);
    for (std::size_t qi = 0; qi < fold.validation.size(); ++qi) {
      scratch.touched.clear();
      const std::size_t row_offset = qi * k;
      const int* neighbor_rows = fold.neighbor_rows.data() + row_offset;
      const float* neighbor_scores = fold.scores.data() + row_offset;
      for (std::size_t j = 0; j < k; ++j) {
        const int row = neighbor_rows[j];
        if (row < 0) continue;
        const int code = scratch.label_codes[static_cast<std::size_t>(row)];
        if (scratch.vote_counts[static_cast<std::size_t>(code)] == 0) {
          scratch.touched.push_back(code);
        }
        scratch.vote_counts[static_cast<std::size_t>(code)]++;
        scratch.vote_scores[static_cast<std::size_t>(code)] += static_cast<double>(neighbor_scores[j]);
      }

      int best_label = 0;
      int best_count = -1;
      double best_score = -std::numeric_limits<double>::infinity();
      if (scratch.touched.empty()) {
        out.predicted[static_cast<std::size_t>(fold.validation[qi])] = fallback_label;
        continue;
      }
      for (int code : scratch.touched) {
        const std::size_t idx = static_cast<std::size_t>(code);
        const int label = scratch.label_values[idx];
        if (scratch.vote_counts[idx] > best_count ||
            (scratch.vote_counts[idx] == best_count && scratch.vote_scores[idx] > best_score) ||
            (scratch.vote_counts[idx] == best_count && scratch.vote_scores[idx] == best_score && label < best_label)) {
          best_label = label;
          best_count = scratch.vote_counts[idx];
          best_score = scratch.vote_scores[idx];
        }
        scratch.vote_counts[idx] = 0;
        scratch.vote_scores[idx] = 0.0;
      }
      out.predicted[static_cast<std::size_t>(fold.validation[qi])] = best_label;
    }
  }
  apply_training_fold_fallbacks(precomputed, scratch, out.predicted);
  return out;
}

double core_objective_score(
  const std::vector<int>& labels,
  double accuracy,
  const CoreOptions& options
) {
  const bool use_diversity = options.guarded_diversity &&
    options.evolution.diversity_multiplier;
  const bool use_fragmentation = options.auto_class_coarsening &&
    options.classifier == CoreClassifier::PLS_LDA &&
    options.evolution.pls_fragmentation_penalty;
  const bool reject_single_class = options.guarded_diversity &&
    options.evolution.reject_single_class;
  if (!use_diversity && !use_fragmentation && !reject_single_class) {
    return accuracy;
  }

  if (labels.empty()) return accuracy;
  std::unordered_map<int, int> counts;
  counts.reserve(labels.size());
  for (int label : labels) counts[label]++;
  const int n_classes = static_cast<int>(counts.size());
  if (n_classes < 1) return accuracy;
  if (reject_single_class && n_classes == 1) {
    return -std::numeric_limits<double>::infinity();
  }

  const double n = static_cast<double>(labels.size());
  double score = accuracy;
  if (use_diversity) {
    double same_label_probability = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      same_label_probability += p * p;
    }
    const double different_label_probability = std::max(0.0, 1.0 - same_label_probability);
    score *= std::sqrt(different_label_probability);
  }

  if (use_fragmentation && n_classes > 1) {
    double entropy = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      if (p > 0.0) entropy -= p * std::log(p);
    }
    const double max_sample_entropy = n > 1.0 ? std::log(n) : 0.0;
    const double label_code_cost = max_sample_entropy > 0.0
      ? std::clamp(entropy / max_sample_entropy, 0.0, 1.0)
      : 0.0;
    const double keff = std::exp(entropy);
    const double fragmentation = std::max(0.0, std::log(static_cast<double>(n_classes) / std::max(1.0, keff)));
    const double fragmentation_cost = fragmentation / (1.0 + fragmentation);
    const double parsimony_cost = std::max(label_code_cost, fragmentation_cost);
    score -= (1.0 - accuracy) * parsimony_cost;
  }

  return score;
}

struct ClassTransitionStats {
  std::map<int, int> class_sizes;
  std::map<int, int> movable_sizes;
  std::map<int, std::map<int, int>> transitions;
};

ClassTransitionStats build_class_transition_stats(
  const std::vector<int>& labels,
  const std::vector<int>& previous_predictions,
  const std::vector<int>& fixed_flags
) {
  ClassTransitionStats stats;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int label = labels[i];
    ++stats.class_sizes[label];
    ++stats.transitions[label][previous_predictions[i]];
    if (fixed_flags.empty() || fixed_flags[i] != 1) ++stats.movable_sizes[label];
  }
  return stats;
}

bool propose_auto_class_coarsening(
  std::vector<int>& labels,
  const std::vector<int>& previous_predictions,
  const std::vector<int>& fixed_flags,
  const ClassTransitionStats& stats,
  const CoreOptions& options,
  std::mt19937_64& rng
) {
  if (!options.auto_class_coarsening) return false;
  if (labels.size() != previous_predictions.size()) return false;
  if (!fixed_flags.empty() && fixed_flags.size() != labels.size()) {
    throw std::invalid_argument("fixed size must be zero or match number of rows.");
  }

  const auto& class_sizes = stats.class_sizes;
  const auto& movable_sizes = stats.movable_sizes;
  const auto& transitions = stats.transitions;

  const int n_classes = static_cast<int>(class_sizes.size());
  if (n_classes <= 2) return false;

  double class_entropy = 0.0;
  const double n = static_cast<double>(labels.size());
  for (const auto& kv : class_sizes) {
    const double p = static_cast<double>(kv.second) / n;
    if (p > 0.0) class_entropy -= p * std::log(p);
  }
  const double keff = std::exp(class_entropy);
  const double fragmentation = std::max(0.0, std::log(static_cast<double>(n_classes) / std::max(1.0, keff)));

  double weighted_transition_entropy = 0.0;
  struct SourceCandidate {
    int source = 0;
    int destination = 0;
    double score = 0.0;
  };
  std::vector<SourceCandidate> candidates;

  for (const auto& kv : class_sizes) {
    const int source = kv.first;
    const int source_size = kv.second;
    const auto movable_it = movable_sizes.find(source);
    const int movable = movable_it == movable_sizes.end() ? 0 : movable_it->second;
    if (movable <= 0) continue;

    int destination = source;
    int destination_size = source_size;
    int best_count = 0;
    double transition_entropy = 0.0;
    const auto& row = transitions.at(source);
    for (const auto& dst : row) {
      const double p = static_cast<double>(dst.second) / static_cast<double>(source_size);
      if (p > 0.0) transition_entropy -= p * std::log(p);
      const auto class_it = class_sizes.find(dst.first);
      if (dst.first != source && class_it != class_sizes.end() &&
          (dst.second > best_count ||
           (dst.second == best_count && class_it->second > destination_size) ||
           (dst.second == best_count && class_it->second == destination_size && dst.first < destination))) {
        destination = dst.first;
        destination_size = class_it->second;
        best_count = dst.second;
      }
    }

    weighted_transition_entropy += (static_cast<double>(source_size) / n) * transition_entropy;
    if (destination == source || best_count == 0) continue;

    const double stay = static_cast<double>(row.count(source) ? row.at(source) : 0) / static_cast<double>(source_size);
    const double smallness = 1.0 / std::sqrt(static_cast<double>(movable));
    const double instability = 1.0 - stay;
    candidates.push_back(SourceCandidate{source, destination, smallness * instability});
  }
  if (candidates.empty()) return false;

  const double temperature = fragmentation / (fragmentation + weighted_transition_entropy + 1.0);
  const double redundant = std::max(0.0, static_cast<double>(n_classes) - keff);
  int merge_budget = static_cast<int>(std::ceil(temperature * redundant));
  if (merge_budget < 1 && fragmentation > 0.0) merge_budget = 1;
  merge_budget = std::min(merge_budget, n_classes - 2);
  if (merge_budget <= 0) return false;

  std::shuffle(candidates.begin(), candidates.end(), rng);
  std::stable_sort(candidates.begin(), candidates.end(), [](const SourceCandidate& a, const SourceCandidate& b) {
    return a.score > b.score;
  });

  std::unordered_map<int, int> remap;
  remap.reserve(class_sizes.size());
  for (const auto& kv : class_sizes) remap[kv.first] = kv.first;

  int merged = 0;
  for (const SourceCandidate& candidate : candidates) {
    if (merged >= merge_budget) break;
    if (remap[candidate.source] != candidate.source) continue;
    int destination = candidate.destination;
    while (remap[destination] != destination) destination = remap[destination];
    if (destination == candidate.source) continue;
    remap[candidate.source] = destination;
    merged++;
  }
  if (merged == 0) return false;

  bool changed = false;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!fixed_flags.empty() && fixed_flags[i] == 1) continue;
    int destination = labels[i];
    while (remap[destination] != destination) destination = remap[destination];
    if (destination != labels[i]) {
      labels[i] = destination;
      changed = true;
    }
  }
  return changed;
}

bool propose_many_to_one_absorption(
  std::vector<int>& labels,
  const std::vector<int>& previous_predictions,
  const std::vector<int>& fixed_flags,
  const ClassTransitionStats& stats,
  std::mt19937_64& rng
) {
  if (labels.size() != previous_predictions.size()) return false;
  if (!fixed_flags.empty() && fixed_flags.size() != labels.size()) {
    throw std::invalid_argument("fixed size must be zero or match number of rows.");
  }

  const auto& class_sizes = stats.class_sizes;
  const auto& movable_sizes = stats.movable_sizes;
  const auto& transitions = stats.transitions;

  const int n_classes = static_cast<int>(class_sizes.size());
  if (n_classes <= 2) return false;
  const double n = static_cast<double>(labels.size());

  struct SourceMove {
    int source = 0;
    double surplus = 0.0;
  };
  struct TargetMove {
    int target = 0;
    double weight = 0.0;
    std::vector<SourceMove> sources;
  };

  std::map<int, TargetMove> by_target;
  for (const auto& kv : class_sizes) {
    const int source = kv.first;
    const int source_size = kv.second;
    const auto movable_it = movable_sizes.find(source);
    if (movable_it == movable_sizes.end() || movable_it->second <= 0) continue;

    const auto& row = transitions.at(source);
    const int stay = row.count(source) ? row.at(source) : 0;
    int best_target = source;
    int best_count = 0;
    int best_target_size = source_size;
    double best_surplus = 0.0;
    for (const auto& dst : row) {
      if (dst.first == source) continue;
      const auto dst_size_it = class_sizes.find(dst.first);
      if (dst_size_it == class_sizes.end()) continue;
      const double expected = static_cast<double>(source_size) *
        static_cast<double>(dst_size_it->second) / std::max(1.0, n);
      const double surplus = static_cast<double>(dst.second) - expected;
      if (surplus > best_surplus ||
          (surplus == best_surplus && dst.second > best_count) ||
          (surplus == best_surplus && dst.second == best_count && dst_size_it->second > best_target_size) ||
          (surplus == best_surplus && dst.second == best_count && dst_size_it->second == best_target_size && dst.first < best_target)) {
        best_target = dst.first;
        best_target_size = dst_size_it->second;
        best_count = dst.second;
        best_surplus = surplus;
      }
    }
    if (best_target == source || best_surplus <= 0.0 || best_count <= stay) continue;

    TargetMove& move = by_target[best_target];
    move.target = best_target;
    move.weight += best_surplus;
    move.sources.push_back(SourceMove{source, best_surplus});
  }
  if (by_target.empty()) return false;

  std::vector<TargetMove> multi_source_moves;
  std::vector<TargetMove> fallback_moves;
  for (auto& kv : by_target) {
    std::sort(kv.second.sources.begin(), kv.second.sources.end(), [](const SourceMove& a, const SourceMove& b) {
      if (a.surplus != b.surplus) return a.surplus > b.surplus;
      return a.source < b.source;
    });
    if (kv.second.sources.size() > 1) {
      multi_source_moves.push_back(std::move(kv.second));
    } else {
      fallback_moves.push_back(std::move(kv.second));
    }
  }
  std::vector<TargetMove>& moves = multi_source_moves.empty() ? fallback_moves : multi_source_moves;
  if (moves.empty()) return false;

  std::vector<double> weights;
  weights.reserve(moves.size());
  for (const TargetMove& move : moves) weights.push_back(std::max(move.weight, std::numeric_limits<double>::min()));
  std::discrete_distribution<int> target_dist(weights.begin(), weights.end());
  const TargetMove& selected = moves[static_cast<std::size_t>(target_dist(rng))];

  std::unordered_map<int, int> remap;
  remap.reserve(selected.sources.size());
  for (const SourceMove& source : selected.sources) {
    if (source.source != selected.target) remap[source.source] = selected.target;
  }
  if (remap.empty()) return false;

  std::vector<int> proposed = labels;
  bool changed = false;
  for (std::size_t i = 0; i < proposed.size(); ++i) {
    if (!fixed_flags.empty() && fixed_flags[i] == 1) continue;
    const auto it = remap.find(proposed[i]);
    if (it == remap.end()) continue;
    proposed[i] = it->second;
    changed = true;
  }
  if (!changed) return false;
  std::unordered_map<int, int> proposed_counts;
  proposed_counts.reserve(proposed.size());
  for (int label : proposed) proposed_counts[label]++;
  const int after_classes = static_cast<int>(proposed_counts.size());
  if (after_classes <= 1 || after_classes >= n_classes) return false;

  labels.swap(proposed);
  return true;
}

int sample_proposal_group_count(
  int group_count,
  int cycle,
  int cycles,
  bool adaptive,
  std::mt19937_64& rng
) {
  if (group_count <= 1) return std::max(0, group_count);
  if (!adaptive || cycles <= 1) {
    std::uniform_int_distribution<int> n_dist(1, group_count);
    return n_dist(rng);
  }

  const double progress = std::clamp(
    static_cast<double>(cycle + 1) / static_cast<double>(cycles + 1),
    0.0,
    1.0
  );
  const double smooth = progress * progress * (3.0 - 2.0 * progress);
  const double temperature = std::clamp(1.0 - smooth, 0.0, 1.0);
  const int max_count = std::max(
    1,
    1 + static_cast<int>(std::floor(static_cast<double>(group_count - 1) * temperature))
  );
  std::uniform_int_distribution<int> n_dist(1, max_count);
  return n_dist(rng);
}

void apply_group_relabel(
  std::vector<int>& labels,
  const std::vector<int>& eligible,
  int replacement
) {
  for (int idx : eligible) {
    int& label = labels[static_cast<std::size_t>(idx)];
    if (label == replacement) continue;
    label = replacement;
  }
}

template <class Predictor>
CoreResult maximize_core(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options,
  Predictor&& predictor
) {
  detail::validate_inputs(x, initial_clbest, constrain);
  if (options.cycles < 0) throw std::invalid_argument("Core cycles must be non-negative.");

  detail::Timer timer;
  CoreResult result;
  result.clbest = initial_clbest;
  result.clbest_dirty = initial_clbest;
  result.vect_acc.assign(static_cast<std::size_t>(options.cycles), -1.0);
  result.vect_score.assign(static_cast<std::size_t>(options.cycles), -1.0);
  result.proposal_size.assign(static_cast<std::size_t>(options.cycles), 0);
  result.active_classes.assign(static_cast<std::size_t>(options.cycles), 0);
  result.accepted.assign(static_cast<std::size_t>(options.cycles), 0);
  result.improving_acceptance.assign(static_cast<std::size_t>(options.cycles), 0);
  result.temperature_acceptance.assign(static_cast<std::size_t>(options.cycles), 0);

  const std::vector<int> group_id = normalized_constrain(constrain, x.rows);
  const std::vector<int> fixed_flags = normalized_fixed(fixed, x.rows);

  std::vector<int> groups;
  std::vector<const std::vector<int>*> group_member_refs;
  std::map<int, std::vector<int>> group_members;
  const bool singleton_groups = constrain.empty();
  if (constrain.empty()) {
    groups.reserve(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i) {
      groups.push_back(static_cast<int>(i));
    }
  } else {
    for (std::size_t i = 0; i < group_id.size(); ++i) {
      group_members[group_id[i]].push_back(static_cast<int>(i));
    }
    groups.reserve(group_members.size());
    group_member_refs.reserve(group_members.size());
    for (const auto& kv : group_members) {
      groups.push_back(static_cast<int>(groups.size()));
      group_member_refs.push_back(&kv.second);
    }
  }
  std::mt19937_64 rng(options.seed);
  CVPrediction best_cv = predictor(result.clbest);
  result.cv_evaluations = 1;
  result.cvpredbest = best_cv.predicted;
  result.accbest = options.shake ? 0.0 : detail::accuracy(result.clbest, result.cvpredbest);
  result.scorebest = options.shake
    ? -std::numeric_limits<double>::infinity()
    : core_objective_score(result.clbest, result.accbest, options);
  result.peak_memory_mb = std::max(result.peak_memory_mb, best_cv.peak_memory_mb);

  {
    std::unordered_map<int, int> active;
    active.reserve(result.clbest.size());
    for (int label : result.clbest) ++active[label];
    report_core_progress(
      options, 0, result.accbest, result.scorebest,
      static_cast<int>(active.size()), true, timer.seconds()
    );
  }

  std::vector<int> current_cl = result.clbest;
  std::vector<int> current_cvpred = result.cvpredbest;
  double current_acc = result.accbest;
  double current_score = result.scorebest;

  std::vector<int> sampled_groups;
  std::vector<int> eligible;
  std::vector<int> candidate_labels;
  std::vector<int> candidate_counts;
  sampled_groups.reserve(groups.size());
  eligible.reserve(x.rows);
  candidate_labels.reserve(x.rows);
  candidate_counts.reserve(x.rows);

  for (int cycle = 0; cycle < options.cycles; ++cycle) {
    std::vector<int> cl = options.evolutionary_search ? current_cl : result.clbest;
    std::vector<int> cl_dirty = cl;
    const std::vector<int>& proposal_predictions = options.evolutionary_search ? current_cvpred : result.cvpredbest;

    if (!groups.empty()) {
      const int n_to_sample = sample_proposal_group_count(
        static_cast<int>(groups.size()),
        cycle,
        options.cycles,
        options.adaptive_proposal_size && options.evolution.adaptive_proposal_size,
        rng
      );
      result.proposal_size[static_cast<std::size_t>(cycle)] = n_to_sample;
      sampled_groups = groups;
      std::shuffle(sampled_groups.begin(), sampled_groups.end(), rng);
      sampled_groups.resize(static_cast<std::size_t>(n_to_sample));

      std::vector<int> empirical_labels;
      std::vector<int> empirical_counts;
      if (!options.evolution.prediction_guidance) {
        empirical_labels = cl;
        std::sort(empirical_labels.begin(), empirical_labels.end());
        for (std::size_t i = 0; i < empirical_labels.size();) {
          std::size_t j = i + 1;
          while (j < empirical_labels.size() && empirical_labels[j] == empirical_labels[i]) ++j;
          empirical_counts.push_back(static_cast<int>(j - i));
          i = j;
        }
        empirical_labels.erase(
          std::unique(empirical_labels.begin(), empirical_labels.end()),
          empirical_labels.end()
        );
      }

      for (int group : sampled_groups) {
        eligible.clear();
        if (singleton_groups) {
          if (fixed_flags[static_cast<std::size_t>(group)] != 1) eligible.push_back(group);
        } else {
          const std::vector<int>& members = *group_member_refs[static_cast<std::size_t>(group)];
          for (int idx : members) {
            if (fixed_flags[static_cast<std::size_t>(idx)] != 1) eligible.push_back(idx);
          }
        }
        if (eligible.empty()) continue;

        candidate_labels.clear();
        candidate_counts.clear();
        if (options.evolution.prediction_guidance) {
          for (int idx : eligible) {
            candidate_labels.push_back(proposal_predictions[static_cast<std::size_t>(idx)]);
          }
          std::sort(candidate_labels.begin(), candidate_labels.end());
          for (std::size_t i = 0; i < candidate_labels.size();) {
            std::size_t j = i + 1;
            while (j < candidate_labels.size() && candidate_labels[j] == candidate_labels[i]) ++j;
            candidate_counts.push_back(static_cast<int>(j - i));
            i = j;
          }
          candidate_labels.erase(std::unique(candidate_labels.begin(), candidate_labels.end()), candidate_labels.end());
        } else {
          candidate_labels = empirical_labels;
          candidate_counts = empirical_counts;
        }
        if (candidate_labels.empty()) continue;

        std::discrete_distribution<int> label_dist(candidate_counts.begin(), candidate_counts.end());
        const int replacement = candidate_labels[static_cast<std::size_t>(label_dist(rng))];
        apply_group_relabel(cl, eligible, replacement);
      }
    }

    ClassTransitionStats transition_stats;
    bool auto_changed = false;
    bool absorption_changed = false;
    const bool allow_transition = options.evolution.transition_proposal;
    const bool attempt_pls_coarsening = allow_transition &&
      options.auto_class_coarsening && options.evolution.pls_transition_coarsening;
    const bool attempt_absorption = allow_transition && options.many_to_one_absorption;
    if (attempt_pls_coarsening || attempt_absorption) {
      ++result.transition_attempted;
      transition_stats = build_class_transition_stats(cl, proposal_predictions, fixed_flags);
    }
    if (attempt_pls_coarsening) {
      ++result.pls_coarsening_attempted;
      auto_changed = propose_auto_class_coarsening(
        cl,
        proposal_predictions,
        fixed_flags,
        transition_stats,
        options,
        rng
      );
      if (auto_changed) ++result.coarsening_moves;
    }
    if (attempt_absorption) {
      ++result.many_to_one_attempted;
      if (auto_changed) {
        const ClassTransitionStats absorption_stats = build_class_transition_stats(
          cl,
          proposal_predictions,
          fixed_flags
        );
        absorption_changed = propose_many_to_one_absorption(
          cl, proposal_predictions, fixed_flags, absorption_stats, rng
        );
      } else {
        absorption_changed = propose_many_to_one_absorption(
          cl, proposal_predictions, fixed_flags, transition_stats, rng
        );
      }
      if (absorption_changed) ++result.absorption_moves;
    }

    CVPrediction cv = predictor(cl);
    ++result.cv_evaluations;
    ++result.proposals_evaluated;
    const double acc = detail::accuracy(cl, cv.predicted);
    const double score = core_objective_score(cl, acc, options);
    result.peak_memory_mb = std::max(result.peak_memory_mb, cv.peak_memory_mb);

    const bool improves_best = score > result.scorebest;
    if (improves_best) {
      result.cvpredbest = cv.predicted;
      result.clbest = cl;
      result.clbest_dirty = cl_dirty;
      result.accbest = acc;
      result.scorebest = score;
      ++result.best_state_updates;
    }

    bool accepted_current = improves_best;
    bool improving_accept = improves_best;
    bool stochastic_accept = false;
    if (options.evolutionary_search) {
      const double cooling = 1.0 - static_cast<double>(cycle + 1) /
        static_cast<double>(std::max(1, options.cycles));
      const double temperature = std::max(1.0e-9, 0.10 * std::max(0.0, 1.0 - current_acc) * cooling);
      bool accept_current = score >= current_score;
      improving_accept = accept_current;
      if (!accept_current && options.evolution.stochastic_acceptance &&
          temperature > 1.0e-9) {
        ++result.stochastic_state_attempts;
        std::uniform_real_distribution<double> accept_dist(0.0, 1.0);
        accept_current = accept_dist(rng) < std::exp((score - current_score) / temperature);
        stochastic_accept = accept_current;
      }
      if (accept_current) {
        current_cl = cl;
        current_cvpred = cv.predicted;
        current_acc = acc;
        current_score = score;
        ++result.current_state_accepts;
        if (stochastic_accept) ++result.stochastic_state_accepts;
      } else {
        ++result.current_state_rejections;
      }
      accepted_current = accept_current;
    }

    result.accepted[static_cast<std::size_t>(cycle)] = accepted_current ? 1 : 0;
    result.improving_acceptance[static_cast<std::size_t>(cycle)] =
      accepted_current && improving_accept ? 1 : 0;
    result.temperature_acceptance[static_cast<std::size_t>(cycle)] =
      stochastic_accept ? 1 : 0;
    if (accepted_current && (auto_changed || absorption_changed)) {
      ++result.transition_accepted;
      if (auto_changed) ++result.pls_coarsening_accepted;
      if (absorption_changed) ++result.many_to_one_accepted;
    }

    {
      const std::vector<int>& diagnostic_labels = options.evolutionary_search
        ? current_cl : result.clbest;
      std::unordered_map<int, int> active;
      active.reserve(diagnostic_labels.size());
      for (int label : diagnostic_labels) ++active[label];
      result.active_classes[static_cast<std::size_t>(cycle)] =
        static_cast<int>(active.size());
    }

    result.vect_acc[static_cast<std::size_t>(cycle)] = result.accbest;
    result.vect_score[static_cast<std::size_t>(cycle)] = result.scorebest;
    result.cycles_completed = cycle + 1;
    report_core_progress(
      options,
      cycle + 1,
      result.accbest,
      result.scorebest,
      result.active_classes[static_cast<std::size_t>(cycle)],
      accepted_current,
      timer.seconds()
    );
    if (acc == 1.0 && std::isfinite(score)) result.success = true;
  }

  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = std::max(result.peak_memory_mb, detail::peak_memory_mb());
  return result;
}

}  // namespace

namespace {

PLSOptions to_plslda_cv_options(const CorePLSLDAOptions& options, Backend backend) {
  PLSOptions out;
  out.cv = options.cv;
  out.max_components = options.max_components;
  out.fixed_components = options.fixed_components;
  out.center = options.center;
  out.scale = options.scale;
  out.backend = backend;
  out.gpu_device = options.gpu_device;
  out.n_threads = options.n_threads;
  out.data_epoch = options.data_epoch;
  return out;
}

std::uint64_t next_pls_data_epoch() {
  static std::atomic<std::uint64_t> epoch{1};
  return epoch.fetch_add(1, std::memory_order_relaxed);
}

CoreResult core_knn_graph_backend(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options,
  Backend backend
) {
  detail::Timer total_timer;
  if (samples < 2) throw std::invalid_argument("CoreKNNGraph requires at least two samples.");
  if (initial_clbest.size() != static_cast<std::size_t>(samples)) {
    throw std::invalid_argument("initial labels size must match NeighborGraph samples.");
  }
  if (!constrain.empty() && constrain.size() != initial_clbest.size()) {
    throw std::invalid_argument("constrain size must be zero or match NeighborGraph samples.");
  }
  if (!fixed.empty() && fixed.size() != initial_clbest.size()) {
    throw std::invalid_argument("fixed size must be zero or match NeighborGraph samples.");
  }

  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = backend;
  const PrecomputedKNN precomputed = precompute_knn_cv_graph(
    graph,
    samples,
    initial_clbest,
    constrain,
    knn_options.knn
  );
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  std::vector<float> dummy(static_cast<std::size_t>(samples), 0.0f);
  MatrixView dummy_view{dummy.data(), static_cast<std::size_t>(samples), 1u};
  CoreResult result = maximize_core(
    dummy_view,
    initial_clbest,
    constrain,
    fixed,
    knn_options,
    [&](const std::vector<int>& labels) {
      return predict_precomputed_knn(precomputed, labels, scratch);
    }
  );
  result.fold_assignments = precomputed.folds;
  result.runtime_seconds = total_timer.seconds();
  return result;
}

}  // namespace

CoreResult CorePLSLDA_CPU(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  detail::Timer total_timer;
  CoreOptions pls_options = options;
  pls_options.classifier = CoreClassifier::PLS_LDA;
  pls_options.pls.backend = Backend::CPU;
  pls_options.pls.data_epoch = next_pls_data_epoch();
  const PLSOptions cv_options = to_plslda_cv_options(pls_options.pls, Backend::CPU);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CPU(x, labels, constrain, cv_options);
    return CVPrediction{cv.predicted_labels, cv.runtime_seconds, cv.peak_memory_mb};
  });
  result.fold_assignments = detail::make_folds(initial_clbest, constrain, cv_options.cv);
  result.runtime_seconds = total_timer.seconds();
  return result;
}

CoreResult CoreKNN_CPU(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  detail::Timer total_timer;
  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = Backend::CPU;
  detail::validate_inputs(x, initial_clbest, constrain);
  const PrecomputedKNN precomputed = precompute_knn_cv_cpu(x, initial_clbest, constrain, knn_options.knn);
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
  result.fold_assignments = precomputed.folds;
  result.runtime_seconds = total_timer.seconds();
  return result;
}

CoreResult CoreKNNGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  return core_knn_graph_backend(
    graph, samples, initial_clbest, constrain, fixed, options, Backend::CPU
  );
}

CoreResult CoreKNNGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  return core_knn_graph_backend(
    graph, samples, initial_clbest, constrain, fixed, options, Backend::CUDA
  );
#else
  (void)graph; (void)samples; (void)initial_clbest; (void)constrain; (void)fixed; (void)options;
  throw std::runtime_error("CoreKNNGraph_CUDA requires a CUDA build.");
#endif
}

CoreResult CoreKNNGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  return core_knn_graph_backend(
    graph, samples, initial_clbest, constrain, fixed, options, Backend::Metal
  );
#else
  (void)graph; (void)samples; (void)initial_clbest; (void)constrain; (void)fixed; (void)options;
  throw std::runtime_error("CoreKNNGraph_METAL requires an Apple Metal build.");
#endif
}

CoreResult CorePLSLDA_CUDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  detail::Timer total_timer;
  CoreOptions pls_options = options;
  pls_options.classifier = CoreClassifier::PLS_LDA;
  pls_options.pls.backend = Backend::CUDA;
  pls_options.pls.data_epoch = next_pls_data_epoch();
  const PLSOptions cv_options = to_plslda_cv_options(pls_options.pls, Backend::CUDA);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CUDA(x, labels, constrain, cv_options);
    return CVPrediction{cv.predicted_labels, cv.runtime_seconds, cv.peak_memory_mb};
  });
  result.fold_assignments = detail::make_folds(initial_clbest, constrain, cv_options.cv);
  result.runtime_seconds = total_timer.seconds();
  return result;
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CorePLSLDA_CUDA requires a CUDA build.");
#endif
}

CoreResult CorePLSLDA_METAL(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  detail::Timer total_timer;
  CoreOptions metal_options = options;
  metal_options.classifier = CoreClassifier::PLS_LDA;
  metal_options.pls.backend = Backend::Metal;
  metal_options.pls.data_epoch = next_pls_data_epoch();
  const PLSOptions cv_options = to_plslda_cv_options(metal_options.pls, Backend::Metal);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, metal_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_METAL(x, labels, constrain, cv_options);
    return CVPrediction{cv.predicted_labels, cv.runtime_seconds, cv.peak_memory_mb};
  });
  result.fold_assignments = detail::make_folds(initial_clbest, constrain, cv_options.cv);
  result.runtime_seconds = total_timer.seconds();
  return result;
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CorePLSLDA_METAL requires an Apple Metal build.");
#endif
}

CoreResult CoreKNN_CUDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  detail::Timer total_timer;
  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = Backend::CUDA;
  detail::validate_inputs(x, initial_clbest, constrain);
  const PrecomputedKNN precomputed = precompute_knn_cv_cuda(x, initial_clbest, constrain, knn_options.knn);
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
  result.fold_assignments = precomputed.folds;
  result.runtime_seconds = total_timer.seconds();
  return result;
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CoreKNN_CUDA requires a CUDA build.");
#endif
}

CoreResult CoreKNN_METAL(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  detail::Timer total_timer;
  CoreOptions metal_options = options;
  metal_options.classifier = CoreClassifier::KNN;
  metal_options.knn.backend = Backend::Metal;
  if (metal_options.knn.index_type != KNNIndexType::MetalIVFFlat) {
    metal_options.knn.index_type = KNNIndexType::MetalExact;
  }
  detail::validate_inputs(x, initial_clbest, constrain);
  const PrecomputedKNN precomputed = precompute_knn_cv_cpu(
    x,
    initial_clbest,
    constrain,
    metal_options.knn
  );
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  CoreResult result = maximize_core(x, initial_clbest, constrain, fixed, metal_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
  result.fold_assignments = precomputed.folds;
  result.runtime_seconds = total_timer.seconds();
  return result;
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CoreKNN_METAL requires an Apple Metal build.");
#endif
}

CoreResult CorePLSLDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.pls.backend == Backend::CUDA) return CorePLSLDA_CUDA(x, initial_clbest, constrain, fixed, options);
  if (options.pls.backend == Backend::Metal) return CorePLSLDA_METAL(x, initial_clbest, constrain, fixed, options);
  return CorePLSLDA_CPU(x, initial_clbest, constrain, fixed, options);
}

CoreResult CoreKNN(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.knn.backend == Backend::CUDA) return CoreKNN_CUDA(x, initial_clbest, constrain, fixed, options);
  if (options.knn.backend == Backend::Metal) return CoreKNN_METAL(x, initial_clbest, constrain, fixed, options);
  return CoreKNN_CPU(x, initial_clbest, constrain, fixed, options);
}

CoreResult core_cpp(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.classifier == CoreClassifier::KNN) return CoreKNN(x, initial_clbest, constrain, fixed, options);
  return CorePLSLDA(x, initial_clbest, constrain, fixed, options);
}

CoreResult Core(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  return core_cpp(x, clbest, constrain, fixed, options);
}

}  // namespace kodama
