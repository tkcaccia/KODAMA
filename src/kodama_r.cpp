// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <Rcpp.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <utility>

#include "kodama/kodama.hpp"

namespace {

constexpr int kTransposeBlock = 64;

struct RGraphHandle {
  kodama::KODAMAGraphResult prepared;
};

Rcpp::XPtr<RGraphHandle> graph_handle_from_sexp(SEXP value) {
  if (TYPEOF(value) != EXTPTRSXP || !Rf_inherits(value, "kodama_graph_handle")) {
    Rcpp::stop("graph handle must be an external pointer created by KODAMA.graph().");
  }
  Rcpp::XPtr<RGraphHandle> handle(value);
  if (handle.get() == nullptr) Rcpp::stop("graph handle is no longer valid.");
  return handle;
}

SEXP graph_handle_to_sexp(kodama::NeighborGraph&& graph, const int samples) {
  graph.index_base = kodama::GraphIndexBase::One;
  auto* value = new RGraphHandle;
  value->prepared.knn = std::move(graph);
  value->prepared.samples = samples;
  Rcpp::XPtr<RGraphHandle> handle(value, true);
  handle.attr("class") = Rcpp::CharacterVector::create("kodama_graph_handle");
  return handle;
}

SEXP graph_handle_to_sexp(kodama::KODAMAGraphResult&& result) {
  result.knn.index_base = kodama::GraphIndexBase::One;
  result.spatial_knn.index_base = kodama::GraphIndexBase::One;
  auto* value = new RGraphHandle;
  value->prepared = std::move(result);
  Rcpp::XPtr<RGraphHandle> handle(value, true);
  handle.attr("class") = Rcpp::CharacterVector::create("kodama_graph_handle");
  return handle;
}

template <typename Source, typename Destination, typename Convert>
void blocked_row_major_to_column_major(
  const Source* source,
  Destination* destination,
  const int rows,
  const int columns,
  Convert convert
) {
  for (int column0 = 0; column0 < columns; column0 += kTransposeBlock) {
    const int column1 = std::min(columns, column0 + kTransposeBlock);
    for (int row0 = 0; row0 < rows; row0 += kTransposeBlock) {
      const int row1 = std::min(rows, row0 + kTransposeBlock);
      for (int column = column0; column < column1; ++column) {
        Destination* output = destination + static_cast<std::size_t>(column) * rows + row0;
        const Source* input = source + static_cast<std::size_t>(row0) * columns + column;
        for (int row = row0; row < row1; ++row) {
          *output++ = convert(*input);
          input += columns;
        }
      }
    }
  }
}

template <typename Source, typename Destination, typename Convert>
void blocked_column_major_to_row_major(
  const Source* source,
  Destination* destination,
  const int rows,
  const int columns,
  Convert convert
) {
  for (int column0 = 0; column0 < columns; column0 += kTransposeBlock) {
    const int column1 = std::min(columns, column0 + kTransposeBlock);
    for (int row0 = 0; row0 < rows; row0 += kTransposeBlock) {
      const int row1 = std::min(rows, row0 + kTransposeBlock);
      for (int column = column0; column < column1; ++column) {
        const Source* input = source + static_cast<std::size_t>(column) * rows + row0;
        Destination* output = destination + static_cast<std::size_t>(row0) * columns + column;
        for (int row = row0; row < row1; ++row) {
          *output = convert(*input++);
          output += columns;
        }
      }
    }
  }
}

kodama::DistanceMetric parse_metric(const std::string& metric) {
  if (metric == "euclidean" || metric == "l2") return kodama::DistanceMetric::Euclidean;
  if (metric == "cosine") return kodama::DistanceMetric::Cosine;
  if (metric == "inner_product" || metric == "ip") return kodama::DistanceMetric::InnerProduct;
  Rcpp::stop("Unsupported metric: " + metric);
}

kodama::Backend parse_backend(const std::string& backend) {
  if (backend == "auto") return kodama::Backend::Auto;
  if (backend == "cpu") return kodama::Backend::CPU;
  if (backend == "cuda") return kodama::Backend::CUDA;
  if (backend == "metal") return kodama::Backend::Metal;
  Rcpp::stop("Unsupported backend: " + backend);
}

kodama::NormalizationMethod parse_normalization_method(const std::string& method) {
  if (method == "pqn") return kodama::NormalizationMethod::PQN;
  if (method == "sum") return kodama::NormalizationMethod::Sum;
  if (method == "median") return kodama::NormalizationMethod::Median;
  if (method == "sqrt") return kodama::NormalizationMethod::Sqrt;
  if (method == "none") return kodama::NormalizationMethod::None;
  Rcpp::stop("Unsupported normalization method: " + method);
}

kodama::ScalingMethod parse_scaling_method(const std::string& method) {
  if (method == "none") return kodama::ScalingMethod::None;
  if (method == "centering") return kodama::ScalingMethod::Centering;
  if (method == "autoscaling") return kodama::ScalingMethod::Autoscaling;
  if (method == "rangescaling") return kodama::ScalingMethod::RangeScaling;
  if (method == "paretoscaling") return kodama::ScalingMethod::ParetoScaling;
  Rcpp::stop("Unsupported scaling method: " + method);
}

kodama::UMAPGraphMode parse_umap_graph_mode(const std::string& mode) {
  if (mode == "binary") return kodama::UMAPGraphMode::Binary;
  if (mode == "fuzzy") return kodama::UMAPGraphMode::Fuzzy;
  Rcpp::stop("Unsupported UMAP graph mode: " + mode);
}

kodama::CoreClassifier parse_classifier(const std::string& classifier) {
  if (classifier == "knn") return kodama::CoreClassifier::KNN;
  if (classifier == "pls_lda") return kodama::CoreClassifier::PLS_LDA;
  Rcpp::stop("Unsupported classifier: " + classifier);
}

kodama::GraphWeightType parse_graph_weight_type(const std::string& weight) {
  if (weight == "snn") return kodama::GraphWeightType::SNN;
  if (weight == "distance") return kodama::GraphWeightType::Distance;
  if (weight == "adaptive") return kodama::GraphWeightType::Adaptive;
  if (weight == "binary") return kodama::GraphWeightType::Binary;
  Rcpp::stop("Unsupported graph weight type: " + weight);
}

kodama::GraphFeatureMode parse_graph_feature_mode(const std::string& mode) {
  if (mode == "laplacian_self_tuning") return kodama::GraphFeatureMode::LaplacianSelfTuning;
  Rcpp::stop("Unsupported graph feature mode: " + mode);
}

std::vector<int> optional_int_vector(Rcpp::Nullable<Rcpp::IntegerVector> value) {
  if (value.isNull()) return {};
  Rcpp::IntegerVector v(value);
  std::vector<int> out(static_cast<std::size_t>(v.size()));
  for (R_xlen_t i = 0; i < v.size(); ++i) out[static_cast<std::size_t>(i)] = v[i];
  return out;
}

std::vector<int> integer_vector_to_std(Rcpp::IntegerVector value) {
  std::vector<int> out(static_cast<std::size_t>(value.size()));
  for (R_xlen_t i = 0; i < value.size(); ++i) out[static_cast<std::size_t>(i)] = value[i];
  return out;
}

std::vector<float> matrix_to_float(Rcpp::NumericMatrix data) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(p), 0.0f);
  blocked_column_major_to_row_major(
    REAL(data), out.data(), n, p, [](const double value) { return static_cast<float>(value); }
  );
  return out;
}

kodama::NeighborGraph graph_from_r(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions.");
  }
  kodama::NeighborGraph graph;
  graph.neighbors = indices.ncol();
  graph.index_base = kodama::GraphIndexBase::One;
  graph.indices.assign(static_cast<std::size_t>(indices.nrow()) * static_cast<std::size_t>(indices.ncol()), 0);
  graph.distances.assign(static_cast<std::size_t>(distances.nrow()) * static_cast<std::size_t>(distances.ncol()), 0.0f);
  blocked_column_major_to_row_major(
    INTEGER(indices), graph.indices.data(), indices.nrow(), indices.ncol(),
    [](const int value) { return value; }
  );
  blocked_column_major_to_row_major(
    REAL(distances), graph.distances.data(), distances.nrow(), distances.ncol(),
    [](const double value) { return static_cast<float>(value); }
  );
  return graph;
}

Rcpp::List graph_to_r(const kodama::NeighborGraph& graph, int samples) {
  Rcpp::IntegerMatrix indices(samples, graph.neighbors);
  Rcpp::NumericMatrix distances(samples, graph.neighbors);
  blocked_row_major_to_column_major(
    graph.indices.data(), INTEGER(indices), samples, graph.neighbors,
    [](const int value) { return value; }
  );
  blocked_row_major_to_column_major(
    graph.distances.data(), REAL(distances), samples, graph.neighbors,
    [](const float value) { return static_cast<double>(value); }
  );
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("neighbors") = graph.neighbors
  );
}

Rcpp::List graph_handle_metadata(
  kodama::NeighborGraph&& graph,
  const int samples
) {
  const int neighbors = graph.neighbors;
  return Rcpp::List::create(
    Rcpp::Named("handle") = graph_handle_to_sexp(std::move(graph), samples),
    Rcpp::Named("samples") = samples,
    Rcpp::Named("neighbors") = neighbors,
    Rcpp::Named("storage") = "handle"
  );
}

Rcpp::List graph_handle_metadata(kodama::KODAMAGraphResult&& result) {
  const int neighbors = result.neighbors > 0 ? result.neighbors : result.knn.neighbors;
  const int samples = result.samples;
  const int spatial_neighbors = result.spatial_knn.neighbors;
  const int spatial_dimensions = result.spatial_dimensions;
  return Rcpp::List::create(
    Rcpp::Named("handle") = graph_handle_to_sexp(std::move(result)),
    Rcpp::Named("samples") = samples,
    Rcpp::Named("neighbors") = neighbors,
    Rcpp::Named("spatial_neighbors") = spatial_neighbors,
    Rcpp::Named("spatial_dimensions") = spatial_dimensions,
    Rcpp::Named("storage") = "handle"
  );
}

Rcpp::List folds_to_r(const std::vector<kodama::FoldResult>& folds) {
  Rcpp::IntegerVector fold(folds.size());
  Rcpp::IntegerVector n_train(folds.size());
  Rcpp::IntegerVector n_validation(folds.size());
  Rcpp::NumericVector accuracy(folds.size());
  for (std::size_t i = 0; i < folds.size(); ++i) {
    fold[static_cast<R_xlen_t>(i)] = folds[i].fold;
    n_train[static_cast<R_xlen_t>(i)] = folds[i].n_train;
    n_validation[static_cast<R_xlen_t>(i)] = folds[i].n_validation;
    accuracy[static_cast<R_xlen_t>(i)] = folds[i].accuracy;
  }
  return Rcpp::List::create(
    Rcpp::Named("fold") = fold,
    Rcpp::Named("n_train") = n_train,
    Rcpp::Named("n_validation") = n_validation,
    Rcpp::Named("accuracy") = accuracy
  );
}

Rcpp::DataFrame stage_timings_to_r(
  const std::vector<kodama::KODAMAStageTiming>& timings
) {
  Rcpp::CharacterVector step(timings.size());
  Rcpp::NumericVector wall(timings.size());
  Rcpp::NumericVector accumulated(timings.size());
  for (std::size_t i = 0; i < timings.size(); ++i) {
    step[static_cast<R_xlen_t>(i)] = timings[i].step;
    wall[static_cast<R_xlen_t>(i)] = timings[i].wall_seconds;
    accumulated[static_cast<R_xlen_t>(i)] = timings[i].accumulated_seconds;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("step") = step,
    Rcpp::Named("wall_seconds") = wall,
    Rcpp::Named("accumulated_seconds") = accumulated,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List confusion_to_r(const kodama::ConfusionMatrix& confusion) {
  Rcpp::IntegerMatrix counts(confusion.n_labels, confusion.n_labels);
  for (std::size_t i = 0; i < confusion.n_labels; ++i) {
    for (std::size_t j = 0; j < confusion.n_labels; ++j) {
      counts(static_cast<int>(i), static_cast<int>(j)) = confusion.counts[i * confusion.n_labels + j];
    }
  }
  Rcpp::IntegerVector labels(confusion.labels.begin(), confusion.labels.end());
  return Rcpp::List::create(
    Rcpp::Named("labels") = labels,
    Rcpp::Named("counts") = counts
  );
}

Rcpp::List knncv_to_r(const kodama::KNNCVResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("predicted") = Rcpp::IntegerVector(result.predicted_labels.begin(), result.predicted_labels.end()),
    Rcpp::Named("truth") = Rcpp::IntegerVector(result.true_labels.begin(), result.true_labels.end()),
    Rcpp::Named("folds") = Rcpp::IntegerVector(result.fold_assignments.begin(), result.fold_assignments.end()),
    Rcpp::Named("fold_accuracy") = folds_to_r(result.folds),
    Rcpp::Named("accuracy") = result.global_accuracy,
    Rcpp::Named("confusion") = confusion_to_r(result.confusion),
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb,
    Rcpp::Named("backend") = kodama::to_string(result.parameters.backend),
    Rcpp::Named("metric") = kodama::to_string(result.parameters.metric),
    Rcpp::Named("index_type") = kodama::to_string(result.parameters.index_type),
    Rcpp::Named("k") = result.parameters.k
  );
}

Rcpp::List plscv_to_r(const kodama::PLSCVResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("predicted") = Rcpp::IntegerVector(result.predicted_labels.begin(), result.predicted_labels.end()),
    Rcpp::Named("truth") = Rcpp::IntegerVector(result.true_labels.begin(), result.true_labels.end()),
    Rcpp::Named("folds") = Rcpp::IntegerVector(result.fold_assignments.begin(), result.fold_assignments.end()),
    Rcpp::Named("fold_accuracy") = folds_to_r(result.folds),
    Rcpp::Named("accuracy_by_components") = Rcpp::NumericVector(result.accuracy_by_components.begin(), result.accuracy_by_components.end()),
    Rcpp::Named("selected_components") = result.selected_components,
    Rcpp::Named("accuracy") = result.global_accuracy,
    Rcpp::Named("confusion") = confusion_to_r(result.confusion),
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb,
    Rcpp::Named("backend") = kodama::to_string(result.parameters.backend),
    Rcpp::Named("mode") = kodama::to_string(result.parameters.mode),
    Rcpp::Named("max_components") = result.parameters.max_components,
    Rcpp::Named("fixed_components") = result.parameters.fixed_components
  );
}

Rcpp::List core_to_r(const kodama::CoreResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("clbest") = Rcpp::IntegerVector(result.clbest.begin(), result.clbest.end()),
    Rcpp::Named("clbest_dirty") = Rcpp::IntegerVector(result.clbest_dirty.begin(), result.clbest_dirty.end()),
    Rcpp::Named("cvpredbest") = Rcpp::IntegerVector(result.cvpredbest.begin(), result.cvpredbest.end()),
    Rcpp::Named("accbest") = result.accbest,
    Rcpp::Named("scorebest") = result.scorebest,
    Rcpp::Named("vect_acc") = Rcpp::NumericVector(result.vect_acc.begin(), result.vect_acc.end()),
    Rcpp::Named("vect_score") = Rcpp::NumericVector(result.vect_score.begin(), result.vect_score.end()),
    Rcpp::Named("proposal_size") = Rcpp::IntegerVector(result.proposal_size.begin(), result.proposal_size.end()),
    Rcpp::Named("active_classes") = Rcpp::IntegerVector(result.active_classes.begin(), result.active_classes.end()),
    Rcpp::Named("accepted") = Rcpp::LogicalVector(result.accepted.begin(), result.accepted.end()),
    Rcpp::Named("improving_acceptance") = Rcpp::LogicalVector(result.improving_acceptance.begin(), result.improving_acceptance.end()),
    Rcpp::Named("temperature_acceptance") = Rcpp::LogicalVector(result.temperature_acceptance.begin(), result.temperature_acceptance.end()),
    Rcpp::Named("cycles_completed") = result.cycles_completed,
    Rcpp::Named("proposals_evaluated") = result.proposals_evaluated,
    Rcpp::Named("best_state_updates") = result.best_state_updates,
    Rcpp::Named("current_state_accepts") = result.current_state_accepts,
    Rcpp::Named("stochastic_state_attempts") = result.stochastic_state_attempts,
    Rcpp::Named("stochastic_state_accepts") = result.stochastic_state_accepts,
    Rcpp::Named("current_state_rejections") = result.current_state_rejections,
    Rcpp::Named("coarsening_moves") = result.coarsening_moves,
    Rcpp::Named("absorption_moves") = result.absorption_moves,
    Rcpp::Named("transition_attempted") = result.transition_attempted,
    Rcpp::Named("transition_accepted") = result.transition_accepted,
    Rcpp::Named("many_to_one_attempted") = result.many_to_one_attempted,
    Rcpp::Named("many_to_one_accepted") = result.many_to_one_accepted,
    Rcpp::Named("pls_coarsening_attempted") = result.pls_coarsening_attempted,
    Rcpp::Named("pls_coarsening_accepted") = result.pls_coarsening_accepted,
    Rcpp::Named("cv_evaluations") = result.cv_evaluations,
    Rcpp::Named("success") = result.success,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb
  );
}

Rcpp::List graph_cluster_result_to_r(const kodama::GraphClusterResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("membership") = Rcpp::IntegerVector(result.membership.begin(), result.membership.end()),
    Rcpp::Named("modularity") = result.modularity,
    Rcpp::Named("n_communities") = result.n_communities,
    Rcpp::Named("n_vertices") = result.n_vertices,
    Rcpp::Named("n_edges") = result.n_edges,
    Rcpp::Named("target_clusters") = result.target_clusters,
    Rcpp::Named("target_gap") = result.target_gap,
    Rcpp::Named("target_exact") = result.target_exact,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("backend") = kodama::to_string(result.backend)
  );
}

Rcpp::NumericMatrix embedding_to_r(const kodama::EmbeddingResult& result) {
  Rcpp::NumericMatrix out(result.samples, result.components);
  for (int i = 0; i < result.samples; ++i) {
    for (int j = 0; j < result.components; ++j) {
      out(i, j) = result.embedding[static_cast<std::size_t>(i) * static_cast<std::size_t>(result.components) + static_cast<std::size_t>(j)];
    }
  }
  out.attr("runtime_seconds") = result.runtime_seconds;
  out.attr("backend") = kodama::to_string(result.backend);
  out.attr("initialization") = result.initialization;
  out.attr("initialization_backend") = kodama::to_string(result.initialization_backend);
  out.attr("optimizer") = result.optimizer;
  out.attr("graph_edges") = static_cast<double>(result.graph_edges);
  out.attr("graph_max_weight") = result.graph_max_weight;
  return out;
}

Rcpp::NumericMatrix float_matrix_to_r(
  const std::vector<float>& values,
  const int rows,
  const int columns
) {
  Rcpp::NumericMatrix out(rows, columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      out(row, column) = values[
        static_cast<std::size_t>(row) * columns + column
      ];
    }
  }
  return out;
}

Rcpp::List visualization_init_to_r(
  const kodama::VisualizationInitResult& result,
  const int seed
) {
  return Rcpp::List::create(
    Rcpp::Named("opentsne") = float_matrix_to_r(
      result.opentsne, result.samples, result.components
    ),
    Rcpp::Named("umap") = float_matrix_to_r(
      result.umap, result.samples, result.components
    ),
    Rcpp::Named("method") = std::string("kodama_cpp_") +
      kodama::to_string(result.backend) + "_rsvd",
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("seed") = seed,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("precision") = "float32"
  );
}

Rcpp::List pca_to_r(const kodama::PCAResult& result) {
  Rcpp::NumericMatrix scores(result.samples, result.components);
  Rcpp::NumericMatrix loadings(result.variables, result.components);
  for (int row = 0; row < result.samples; ++row) {
    for (int component = 0; component < result.components; ++component) {
      scores(row, component) = result.scores[
        static_cast<std::size_t>(row) * result.components + component
      ];
    }
  }
  for (int variable = 0; variable < result.variables; ++variable) {
    for (int component = 0; component < result.components; ++component) {
      loadings(variable, component) = result.loadings[
        static_cast<std::size_t>(variable) * result.components + component
      ];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("scores") = scores,
    Rcpp::Named("loadings") = loadings,
    Rcpp::Named("singular_values") = Rcpp::NumericVector(
      result.singular_values.begin(), result.singular_values.end()
    ),
    Rcpp::Named("sdev") = Rcpp::NumericVector(result.sdev.begin(), result.sdev.end()),
    Rcpp::Named("variance") = Rcpp::NumericVector(result.variance.begin(), result.variance.end()),
    Rcpp::Named("variance_explained") = Rcpp::NumericVector(
      result.variance_explained.begin(), result.variance_explained.end()
    ),
    Rcpp::Named("cumulative_variance_explained") = Rcpp::NumericVector(
      result.cumulative_variance_explained.begin(), result.cumulative_variance_explained.end()
    ),
    Rcpp::Named("total_variance") = result.total_variance,
    Rcpp::Named("center") = Rcpp::NumericVector(result.center.begin(), result.center.end()),
    Rcpp::Named("scale") = Rcpp::NumericVector(result.scale.begin(), result.scale.end()),
    Rcpp::Named("ncomp") = result.components,
    Rcpp::Named("oversample") = result.oversample,
    Rcpp::Named("power") = result.power_iterations,
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("runtime_seconds") = result.runtime_seconds
  );
}

Rcpp::List kodama_matrix_result_to_r(
  kodama::KODAMAMatrixResult& result,
  const kodama::KODAMAMatrixOptions& options,
  const int graph_output
) {
  double landmark_sum_seconds = 0.0;
  for (const double seconds : result.landmark_seconds) {
    landmark_sum_seconds += seconds;
  }
  const double landmark_mean_seconds = result.landmark_seconds.empty()
    ? 0.0
    : landmark_sum_seconds / static_cast<double>(result.landmark_seconds.size());
  std::vector<double> sorted_landmark_seconds = result.landmark_seconds;
  std::sort(sorted_landmark_seconds.begin(), sorted_landmark_seconds.end());
  double landmark_median_seconds = 0.0;
  if (!sorted_landmark_seconds.empty()) {
    const std::size_t middle = sorted_landmark_seconds.size() / 2;
    landmark_median_seconds = sorted_landmark_seconds[middle];
    if (sorted_landmark_seconds.size() % 2 == 0) {
      landmark_median_seconds = 0.5 * (
        sorted_landmark_seconds[middle - 1] + sorted_landmark_seconds[middle]
      );
    }
  }

  Rcpp::NumericVector acc(result.acc.begin(), result.acc.end());
  Rcpp::NumericMatrix v(result.runs, result.cycles);
  for (int i = 0; i < result.runs; ++i) {
    for (int j = 0; j < result.cycles; ++j) {
      v(i, j) = result.v[static_cast<std::size_t>(i) * result.cycles + j];
    }
  }

  Rcpp::IntegerMatrix res(result.runs, result.samples);
  Rcpp::IntegerMatrix res_constrain(result.runs, result.samples);
  int* res_output = INTEGER(res);
  int* constrain_output = INTEGER(res_constrain);
  constexpr int block = 32;
  for (int sample_begin = 0; sample_begin < result.samples; sample_begin += block) {
    const int sample_end = std::min(result.samples, sample_begin + block);
    for (int run_begin = 0; run_begin < result.runs; run_begin += block) {
      const int run_end = std::min(result.runs, run_begin + block);
      for (int j = sample_begin; j < sample_end; ++j) {
        for (int i = run_begin; i < run_end; ++i) {
          const std::size_t input = static_cast<std::size_t>(i) * result.samples + j;
          const std::size_t output = static_cast<std::size_t>(j) * result.runs + i;
          res_output[output] = result.res[input];
          const std::size_t constrain_input =
            static_cast<std::size_t>(result.res_constrain_rows > 1 ? i : 0) * result.samples + j;
          constrain_output[output] = result.res_constrain[constrain_input];
        }
      }
    }
  }

  Rcpp::RObject visual_init = R_NilValue;
  if (result.has_visual_init) {
    visual_init = visualization_init_to_r(
      result.visual_init,
      static_cast<int>(options.seed)
    );
  }
  Rcpp::RObject graph = R_NilValue;
  const auto graph_conversion_start = std::chrono::steady_clock::now();
  if (graph_output == 1) {
    graph = graph_to_r(result.knn, result.samples);
  } else if (graph_output == 2) {
    graph = graph_handle_metadata(std::move(result.knn), result.samples);
  }
  const double graph_conversion_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - graph_conversion_start
  ).count();

  Rcpp::IntegerVector diagnostic_run(result.run_diagnostics.size());
  Rcpp::IntegerVector diagnostic_cycles(result.run_diagnostics.size());
  Rcpp::IntegerVector transition_attempted(result.run_diagnostics.size());
  Rcpp::IntegerVector transition_accepted(result.run_diagnostics.size());
  Rcpp::IntegerVector many_to_one_attempted(result.run_diagnostics.size());
  Rcpp::IntegerVector many_to_one_accepted(result.run_diagnostics.size());
  Rcpp::IntegerVector pls_coarsening_attempted(result.run_diagnostics.size());
  Rcpp::IntegerVector pls_coarsening_accepted(result.run_diagnostics.size());
  Rcpp::IntegerVector cv_evaluations(result.run_diagnostics.size());
  Rcpp::CharacterVector landmark_rows_hash(result.run_diagnostics.size());
  Rcpp::CharacterVector initial_labels_hash(result.run_diagnostics.size());
  Rcpp::CharacterVector fold_assignments_hash(result.run_diagnostics.size());
  for (std::size_t i = 0; i < result.run_diagnostics.size(); ++i) {
    const auto& diagnostic = result.run_diagnostics[i];
    diagnostic_run[i] = diagnostic.run;
    diagnostic_cycles[i] = diagnostic.cycles_completed;
    transition_attempted[i] = diagnostic.transition_attempted;
    transition_accepted[i] = diagnostic.transition_accepted;
    many_to_one_attempted[i] = diagnostic.many_to_one_attempted;
    many_to_one_accepted[i] = diagnostic.many_to_one_accepted;
    pls_coarsening_attempted[i] = diagnostic.pls_coarsening_attempted;
    pls_coarsening_accepted[i] = diagnostic.pls_coarsening_accepted;
    cv_evaluations[i] = diagnostic.cv_evaluations;
    std::ostringstream landmark_stream;
    std::ostringstream label_stream;
    std::ostringstream fold_stream;
    landmark_stream << std::hex << diagnostic.landmark_rows_hash;
    label_stream << std::hex << diagnostic.initial_labels_hash;
    fold_stream << std::hex << diagnostic.fold_assignments_hash;
    landmark_rows_hash[i] = landmark_stream.str();
    initial_labels_hash[i] = label_stream.str();
    fold_assignments_hash[i] = fold_stream.str();
  }
  const Rcpp::DataFrame run_diagnostics = Rcpp::DataFrame::create(
    Rcpp::Named("run") = diagnostic_run,
    Rcpp::Named("cycles_completed") = diagnostic_cycles,
    Rcpp::Named("transition_attempted") = transition_attempted,
    Rcpp::Named("transition_accepted") = transition_accepted,
    Rcpp::Named("many_to_one_attempted") = many_to_one_attempted,
    Rcpp::Named("many_to_one_accepted") = many_to_one_accepted,
    Rcpp::Named("pls_coarsening_attempted") = pls_coarsening_attempted,
    Rcpp::Named("pls_coarsening_accepted") = pls_coarsening_accepted,
    Rcpp::Named("cv_evaluations") = cv_evaluations,
    Rcpp::Named("landmark_rows_hash") = landmark_rows_hash,
    Rcpp::Named("initial_labels_hash") = initial_labels_hash,
    Rcpp::Named("fold_assignments_hash") = fold_assignments_hash
  );
  Rcpp::IntegerVector cycle_run(result.cycle_diagnostics.size());
  Rcpp::IntegerVector cycle_number(result.cycle_diagnostics.size());
  Rcpp::IntegerVector proposal_size(result.cycle_diagnostics.size());
  Rcpp::IntegerVector active_classes(result.cycle_diagnostics.size());
  Rcpp::LogicalVector accepted(result.cycle_diagnostics.size());
  Rcpp::LogicalVector improving_acceptance(result.cycle_diagnostics.size());
  Rcpp::LogicalVector temperature_acceptance(result.cycle_diagnostics.size());
  for (std::size_t i = 0; i < result.cycle_diagnostics.size(); ++i) {
    const auto& diagnostic = result.cycle_diagnostics[i];
    cycle_run[i] = diagnostic.run;
    cycle_number[i] = diagnostic.cycle;
    proposal_size[i] = diagnostic.proposal_size;
    active_classes[i] = diagnostic.active_classes;
    accepted[i] = diagnostic.accepted != 0;
    improving_acceptance[i] = diagnostic.improving_acceptance != 0;
    temperature_acceptance[i] = diagnostic.temperature_acceptance != 0;
  }
  const Rcpp::DataFrame cycle_diagnostics = Rcpp::DataFrame::create(
    Rcpp::Named("run") = cycle_run,
    Rcpp::Named("cycle") = cycle_number,
    Rcpp::Named("proposal_size") = proposal_size,
    Rcpp::Named("active_classes") = active_classes,
    Rcpp::Named("accepted") = accepted,
    Rcpp::Named("improving_acceptance") = improving_acceptance,
    Rcpp::Named("temperature_acceptance") = temperature_acceptance
  );

  return Rcpp::List::create(
    Rcpp::Named("acc") = acc,
    Rcpp::Named("v") = v,
    Rcpp::Named("res") = res,
    Rcpp::Named("knn") = graph,
    Rcpp::Named("knn_is_kodama_corrected") = result.knn_is_kodama_corrected,
    Rcpp::Named("graph_storage_bytes") =
      static_cast<double>(result.graph_storage_bytes),
    Rcpp::Named("visual_init") = visual_init,
    Rcpp::Named("res_constrain") = res_constrain,
    Rcpp::Named("run_diagnostics") = run_diagnostics,
    Rcpp::Named("cycle_diagnostics") = cycle_diagnostics,
    Rcpp::Named("graph_builds") = result.graph_builds,
    Rcpp::Named("spatial_graph_builds") = result.spatial_graph_builds,
    Rcpp::Named("shared_landmark_partition_used") =
      result.shared_landmark_partition_used,
    Rcpp::Named("shared_landmark_partition_strata") =
      result.shared_landmark_partition_strata,
    Rcpp::Named("graph_index_type") = kodama::to_string(result.graph_index_type),
    Rcpp::Named("graph_ivf_nlist") = result.graph_ivf_nlist,
    Rcpp::Named("graph_ivf_nprobe") = result.graph_ivf_nprobe,
    Rcpp::Named("graph_ivf_pilot_recall") = result.graph_ivf_pilot_recall,
    Rcpp::Named("landmark_seconds") = Rcpp::NumericVector(
      result.landmark_seconds.begin(), result.landmark_seconds.end()
    ),
    Rcpp::Named("coarse_partition_seconds") = Rcpp::NumericVector(
      result.coarse_partition_seconds.begin(), result.coarse_partition_seconds.end()
    ),
    Rcpp::Named("landmark_sampling_seconds") = Rcpp::NumericVector(
      result.landmark_sampling_seconds.begin(), result.landmark_sampling_seconds.end()
    ),
    Rcpp::Named("constraint_seconds") = Rcpp::NumericVector(
      result.constraint_seconds.begin(), result.constraint_seconds.end()
    ),
    Rcpp::Named("landmark_prepare_seconds") = Rcpp::NumericVector(
      result.landmark_prepare_seconds.begin(), result.landmark_prepare_seconds.end()
    ),
    Rcpp::Named("landmark_initialization_seconds") = Rcpp::NumericVector(
      result.landmark_initialization_seconds.begin(), result.landmark_initialization_seconds.end()
    ),
    Rcpp::Named("landmark_graph_seconds") = Rcpp::NumericVector(
      result.landmark_graph_seconds.begin(), result.landmark_graph_seconds.end()
    ),
    Rcpp::Named("core_evolution_seconds") = Rcpp::NumericVector(
      result.core_evolution_seconds.begin(), result.core_evolution_seconds.end()
    ),
    Rcpp::Named("projection_seconds") = Rcpp::NumericVector(
      result.projection_seconds.begin(), result.projection_seconds.end()
    ),
    Rcpp::Named("landmark_occupied_strata") = Rcpp::IntegerVector(
      result.landmark_occupied_strata.begin(), result.landmark_occupied_strata.end()
    ),
    Rcpp::Named("landmark_represented_strata") = Rcpp::IntegerVector(
      result.landmark_represented_strata.begin(), result.landmark_represented_strata.end()
    ),
    Rcpp::Named("landmark_grid_bins") = Rcpp::IntegerVector(
      result.landmark_grid_bins.begin(), result.landmark_grid_bins.end()
    ),
    Rcpp::Named("n.cores") = result.n_threads,
    Rcpp::Named("gpu_auto_workers") = result.gpu_auto_workers,
    Rcpp::Named("gpu_scheduler_enabled") = result.gpu_scheduler_enabled,
    Rcpp::Named("gpu_scheduler_lanes") = result.gpu_scheduler_lanes,
    Rcpp::Named("kmeans_input_uploads") =
      static_cast<double>(result.kmeans_input_uploads),
    Rcpp::Named("projection_sparse_uploads") = static_cast<double>(result.projection_sparse_uploads),
    Rcpp::Named("projection_full_downloads") = static_cast<double>(result.projection_full_downloads),
    Rcpp::Named("result_row_uploads") = static_cast<double>(result.result_row_uploads),
    Rcpp::Named("result_matrix_downloads") = static_cast<double>(result.result_matrix_downloads),
    Rcpp::Named("gpu_worker_memory_estimate_mb") = result.gpu_worker_memory_estimate_mb,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("analysis_storage") = "float32",
    Rcpp::Named("classifier") = kodama::to_string(options.classifier),
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("graph_backend") = kodama::to_string(result.graph_backend),
    Rcpp::Named("optimization_backend") = kodama::to_string(result.optimization_backend),
    Rcpp::Named("dissimilarity_backend") = kodama::to_string(result.dissimilarity_backend),
    Rcpp::Named("graph_feature_mode") = kodama::to_string(options.graph_feature_mode),
    Rcpp::Named("timing") = Rcpp::List::create(
      Rcpp::Named("input_copy_seconds") = result.input_copy_seconds,
      Rcpp::Named("visual_init_seconds") = result.visual_init_seconds,
      Rcpp::Named("graph_feature_seconds") = result.graph_feature_seconds,
      Rcpp::Named("spatial_precompute_seconds") = result.spatial_precompute_seconds,
      Rcpp::Named("graph_seconds") = result.graph_seconds,
      Rcpp::Named("shared_landmark_partition_seconds") =
        result.shared_landmark_partition_seconds,
      Rcpp::Named("spatial_graph_seconds") = result.spatial_graph_seconds,
      Rcpp::Named("landmark_sum_seconds") = landmark_sum_seconds,
      Rcpp::Named("landmark_mean_seconds") = landmark_mean_seconds,
      Rcpp::Named("landmark_median_seconds") = landmark_median_seconds,
      Rcpp::Named("optimization_wall_seconds") = result.optimization_wall_seconds,
      Rcpp::Named("optimization_sum_seconds") = result.optimization_sum_seconds,
      Rcpp::Named("dissimilarity_seconds") = result.dissimilarity_seconds,
      Rcpp::Named("r_graph_conversion_seconds") = graph_conversion_seconds,
      Rcpp::Named("runtime_seconds") = result.runtime_seconds
    ),
    Rcpp::Named("timings") = stage_timings_to_r(result.timings),
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb
  );
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List kodama_matrix_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> W = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int M = 100,
  int Tcycle = 20,
  int ncomp = 50,
  int landmarks = 10000,
  int splitting = 0,
  int n_cores = 4,
  int graph_neighbors = 100,
  int knn_k = 30,
  double spatial_resolution = 0.4,
  bool spatial_graph_mix = false,
  int spatial_constraint_mode = 0,
  int spatial_coordinate_mode = 0,
  std::string metric = "euclidean",
  std::string classifier = "knn",
  std::string backend = "cpu",
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true,
  bool compute_visual_init = false,
  int graph_output = 0,
  int folds = 5,
  std::string evolution_policy = "full"
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x(static_cast<std::size_t>(n) * static_cast<std::size_t>(p), 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      x[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)] =
        static_cast<float>(data(i, j));
    }
  }

  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.folds = folds;
  options.evolution = kodama::EvolutionPolicy::from_name(evolution_policy);
  options.n_threads = n_cores;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.spatial_coordinate_mode = spatial_coordinate_mode == 1 ?
    kodama::SpatialCoordinateMode::Population :
    kodama::SpatialCoordinateMode::Standard;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.compute_visual_init = compute_visual_init;
  options.materialize_graph = graph_output != 0;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;
  options.samples = optional_int_vector(samples);

  if (!spatial.isNull()) {
    Rcpp::NumericMatrix s(spatial);
    if (s.nrow() != n) Rcpp::stop("spatial must have the same number of rows as data.");
    options.spatial_cols = s.ncol();
    options.spatial.assign(static_cast<std::size_t>(s.nrow()) * static_cast<std::size_t>(s.ncol()), 0.0f);
    for (int i = 0; i < s.nrow(); ++i) {
      for (int j = 0; j < s.ncol(); ++j) {
        options.spatial[static_cast<std::size_t>(i) * static_cast<std::size_t>(s.ncol()) + static_cast<std::size_t>(j)] =
          static_cast<float>(s(i, j));
      }
    }
  }

  kodama::MatrixView view{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  kodama::KODAMAMatrixResult result = kodama::KODAMAMatrix(
    view,
    optional_int_vector(W),
    optional_int_vector(constrain),
    optional_int_vector(fix),
    options
  );

  return kodama_matrix_result_to_r(result, options, graph_output);
}

// [[Rcpp::export]]
Rcpp::List kodama_matrix_graph_cpp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::IntegerMatrix> spatial_indices = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial_distances = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericVector> spatial_jitter = R_NilValue,
  int prepared_spatial_dimensions = 0,
  Rcpp::Nullable<Rcpp::NumericMatrix> data = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> W = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int M = 100,
  int Tcycle = 20,
  int ncomp = 50,
  int landmarks = 10000,
  int splitting = 0,
  int n_cores = 4,
  int graph_neighbors = 100,
  int knn_k = 30,
  double spatial_resolution = 0.3,
  bool spatial_graph_mix = false,
  int spatial_constraint_mode = 0,
  int spatial_coordinate_mode = 0,
  std::string classifier = "knn",
  std::string backend = "cpu",
  std::string graph_feature_mode = "laplacian_self_tuning",
  int graph_feature_components = 0,
  int graph_feature_steps = 3,
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true,
  int graph_output = 0,
  int folds = 5,
  std::string evolution_policy = "full"
) {
  const int n = indices.nrow();
  kodama::NeighborGraph graph = graph_from_r(indices, distances);
  kodama::KODAMAGraphResult prepared;
  prepared.knn = graph;
  prepared.samples = n;
  if (spatial_indices.isNotNull() || spatial_distances.isNotNull()) {
    if (spatial_indices.isNull() || spatial_distances.isNull()) {
      Rcpp::stop("prepared spatial indices and distances must be supplied together.");
    }
    prepared.spatial_knn = graph_from_r(
      Rcpp::IntegerMatrix(spatial_indices),
      Rcpp::NumericMatrix(spatial_distances)
    );
    prepared.spatial_dimensions = prepared_spatial_dimensions;
    prepared.spatial_graph_builds = 1;
  }
  if (spatial_jitter.isNotNull()) {
    Rcpp::NumericVector jitter(spatial_jitter);
    prepared.spatial_jitter.assign(jitter.begin(), jitter.end());
  }
  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.folds = folds;
  options.evolution = kodama::EvolutionPolicy::from_name(evolution_policy);
  options.n_threads = n_cores;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.spatial_coordinate_mode = spatial_coordinate_mode == 1 ?
    kodama::SpatialCoordinateMode::Population :
    kodama::SpatialCoordinateMode::Standard;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = kodama::DistanceMetric::Euclidean;
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.materialize_graph = graph_output != 0;
  options.compute_visual_init = false;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;
  options.graph_feature_mode = parse_graph_feature_mode(graph_feature_mode);
  options.graph_feature_components = graph_feature_components;
  options.graph_feature_steps = graph_feature_steps;
  options.samples = optional_int_vector(samples);
  if (!spatial.isNull()) {
    Rcpp::NumericMatrix s(spatial);
    if (s.nrow() != n) Rcpp::stop("spatial rows must match graph rows.");
    options.spatial_cols = s.ncol();
    options.spatial.assign(static_cast<std::size_t>(s.nrow()) * static_cast<std::size_t>(s.ncol()), 0.0f);
    for (int i = 0; i < s.nrow(); ++i) {
      for (int j = 0; j < s.ncol(); ++j) {
        options.spatial[static_cast<std::size_t>(i) * static_cast<std::size_t>(s.ncol()) + static_cast<std::size_t>(j)] =
          static_cast<float>(s(i, j));
      }
    }
  }
  std::vector<int> labels = optional_int_vector(W);
  std::vector<int> constraints = optional_int_vector(constrain);
  std::vector<int> fixed = optional_int_vector(fix);
  kodama::KODAMAMatrixResult result;
  if (data.isNotNull()) {
    Rcpp::NumericMatrix data_matrix(data);
    if (data_matrix.nrow() != n) Rcpp::stop("data rows must match graph rows.");
    std::vector<float> x = matrix_to_float(data_matrix);
    kodama::MatrixView view{x.data(), static_cast<std::size_t>(data_matrix.nrow()), static_cast<std::size_t>(data_matrix.ncol())};
    prepared.dimensions = data_matrix.ncol();
    result = kodama::KODAMAMatrix(
      view, prepared, labels, constraints, fixed, options
    );
  } else {
    result = kodama::KODAMAMatrixFromGraph(graph, n, labels, constraints, fixed, options);
  }
  return kodama_matrix_result_to_r(result, options, graph_output);
}

// [[Rcpp::export]]
Rcpp::List kodama_matrix_graph_handle_cpp(
  SEXP graph_handle,
  Rcpp::Nullable<Rcpp::NumericMatrix> data = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> W = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int M = 100,
  int Tcycle = 20,
  int ncomp = 50,
  int landmarks = 10000,
  int splitting = 0,
  int n_cores = 4,
  int graph_neighbors = 100,
  int knn_k = 30,
  double spatial_resolution = 0.3,
  bool spatial_graph_mix = false,
  int spatial_constraint_mode = 0,
  int spatial_coordinate_mode = 0,
  std::string classifier = "knn",
  std::string backend = "cpu",
  std::string graph_feature_mode = "laplacian_self_tuning",
  int graph_feature_components = 0,
  int graph_feature_steps = 3,
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true,
  int graph_output = 0,
  int folds = 5,
  std::string evolution_policy = "full"
) {
  Rcpp::XPtr<RGraphHandle> handle = graph_handle_from_sexp(graph_handle);
  const int n = handle->prepared.samples;
  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.folds = folds;
  options.evolution = kodama::EvolutionPolicy::from_name(evolution_policy);
  options.n_threads = n_cores;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.spatial_coordinate_mode = spatial_coordinate_mode == 1 ?
    kodama::SpatialCoordinateMode::Population :
    kodama::SpatialCoordinateMode::Standard;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = kodama::DistanceMetric::Euclidean;
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.materialize_graph = graph_output != 0;
  options.compute_visual_init = false;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;
  options.graph_feature_mode = parse_graph_feature_mode(graph_feature_mode);
  options.graph_feature_components = graph_feature_components;
  options.graph_feature_steps = graph_feature_steps;
  options.samples = optional_int_vector(samples);
  if (!spatial.isNull()) {
    Rcpp::NumericMatrix s(spatial);
    if (s.nrow() != n) Rcpp::stop("spatial rows must match graph rows.");
    options.spatial_cols = s.ncol();
    options.spatial = matrix_to_float(s);
  }
  const std::vector<int> labels = optional_int_vector(W);
  const std::vector<int> constraints = optional_int_vector(constrain);
  const std::vector<int> fixed = optional_int_vector(fix);
  kodama::KODAMAMatrixResult result;
  if (data.isNotNull()) {
    Rcpp::NumericMatrix data_matrix(data);
    if (data_matrix.nrow() != n) Rcpp::stop("data rows must match graph rows.");
    std::vector<float> x = matrix_to_float(data_matrix);
    const kodama::MatrixView view{
      x.data(), static_cast<std::size_t>(data_matrix.nrow()),
      static_cast<std::size_t>(data_matrix.ncol())
    };
    result = kodama::KODAMAMatrix(
      view, handle->prepared, labels, constraints, fixed, options
    );
  } else {
    result = kodama::KODAMAMatrix(
      handle->prepared, labels, constraints, fixed, options);
  }
  return kodama_matrix_result_to_r(result, options, graph_output);
}

// [[Rcpp::export]]
Rcpp::List knncv_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::IntegerVector labels,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  int folds = 10,
  bool stratified = true,
  int seed = 1,
  int k = 10,
  std::string metric = "cosine",
  std::string backend = "cpu",
  int n_threads = 1,
  int gpu_device = 0
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::KNNOptions options;
  options.cv.folds = folds;
  options.cv.stratified = stratified;
  options.cv.seed = static_cast<std::uint64_t>(seed);
  options.k = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  kodama::KNNCVResult result = kodama::KNNCV(
    kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
    integer_vector_to_std(labels),
    optional_int_vector(constrain),
    options
  );
  return knncv_to_r(result);
}

// [[Rcpp::export]]
Rcpp::List plsldacv_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::IntegerVector labels,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  int folds = 10,
  bool stratified = true,
  int seed = 1,
  int ncomp = 50,
  bool center = true,
  bool scale = true,
  std::string backend = "cpu",
  int n_threads = 1,
  int gpu_device = 0
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::PLSOptions options;
  options.cv.folds = folds;
  options.cv.stratified = stratified;
  options.cv.seed = static_cast<std::uint64_t>(seed);
  options.max_components = ncomp;
  options.fixed_components = ncomp;
  options.center = center;
  options.scale = scale;
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  kodama::PLSCVResult result = kodama::PLSLDACV(
    kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
    integer_vector_to_std(labels),
    optional_int_vector(constrain),
    options
  );
  return plscv_to_r(result);
}

Rcpp::List core_impl_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::IntegerVector starting_labels,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain,
  Rcpp::Nullable<Rcpp::IntegerVector> fix,
  int cycles,
  int folds,
  bool stratified,
  int seed,
  int k,
  int ncomp,
  std::string metric,
  std::string backend,
  int n_threads,
  int gpu_device,
  kodama::CoreClassifier classifier
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::CoreOptions options;
  options.cycles = cycles;
  options.classifier = classifier;
  options.seed = static_cast<std::uint64_t>(seed);
  options.knn.cv.folds = folds;
  options.knn.cv.stratified = stratified;
  options.knn.cv.seed = static_cast<std::uint64_t>(seed);
  options.knn.k = k;
  options.knn.metric = parse_metric(metric);
  options.knn.backend = parse_backend(backend);
  options.knn.n_threads = n_threads;
  options.knn.gpu_device = gpu_device;
  options.knn.hnsw_tune_k = std::max(50, k);
  options.knn.hnsw_target_recall = 0.99;
  options.pls.cv.folds = folds;
  options.pls.cv.stratified = stratified;
  options.pls.cv.seed = static_cast<std::uint64_t>(seed);
  options.pls.max_components = ncomp;
  options.pls.fixed_components = ncomp;
  options.pls.backend = parse_backend(backend);
  options.pls.n_threads = n_threads;
  options.pls.gpu_device = gpu_device;
  const kodama::MatrixView view{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  const std::vector<int> labels = integer_vector_to_std(starting_labels);
  const std::vector<int> constraints = optional_int_vector(constrain);
  const std::vector<int> fixed = optional_int_vector(fix);
  kodama::CoreResult result = classifier == kodama::CoreClassifier::KNN ?
    kodama::CoreKNN(view, labels, constraints, fixed, options) :
    kodama::CorePLSLDA(view, labels, constraints, fixed, options);
  Rcpp::List out = core_to_r(result);
  out["backend"] = kodama::to_string(
    classifier == kodama::CoreClassifier::KNN ?
      options.knn.backend :
      options.pls.backend
  );
  return out;
}

// [[Rcpp::export]]
Rcpp::List core_knn_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::IntegerVector starting_labels,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int cycles = 100,
  int folds = 10,
  bool stratified = true,
  int seed = 1,
  int k = 30,
  std::string metric = "euclidean",
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0
) {
  return core_impl_cpp(data, starting_labels, constrain, fix, cycles, folds, stratified, seed, k, 1, metric, backend, n_threads, gpu_device, kodama::CoreClassifier::KNN);
}

// [[Rcpp::export]]
Rcpp::List core_plslda_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::IntegerVector starting_labels,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int cycles = 100,
  int folds = 10,
  bool stratified = true,
  int seed = 1,
  int ncomp = 50,
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0
) {
  return core_impl_cpp(data, starting_labels, constrain, fix, cycles, folds, stratified, seed, 30, ncomp, "euclidean", backend, n_threads, gpu_device, kodama::CoreClassifier::PLS_LDA);
}

// [[Rcpp::export]]
Rcpp::List kodama_knn_graph_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  int k = 30,
  std::string metric = "euclidean",
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0,
  int seed = 1234,
  std::string storage = "handle"
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::KODAMAGraphOptions options;
  options.neighbors = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  options.seed = static_cast<std::uint64_t>(seed);
  options.samples = optional_int_vector(samples);
  if (storage != "matrix" && storage != "handle") {
    Rcpp::stop("storage must be 'matrix' or 'handle'.");
  }
  options.materialize_graph = storage == "matrix";
  const kodama::MatrixView view{
    x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)
  };
  kodama::KODAMAGraphResult result;
  if (spatial.isNotNull()) {
    Rcpp::NumericMatrix spatial_matrix(spatial);
    if (spatial_matrix.nrow() != n || spatial_matrix.ncol() < 1) {
      Rcpp::stop("spatial must have one row per data sample.");
    }
    std::vector<float> spatial_data = matrix_to_float(spatial_matrix);
    const kodama::MatrixView spatial_view{
      spatial_data.data(), static_cast<std::size_t>(spatial_matrix.nrow()),
      static_cast<std::size_t>(spatial_matrix.ncol())
    };
    result = kodama::KODAMAGraph(view, spatial_view, options);
  } else {
    result = kodama::KODAMAGraph(view, options);
  }
  Rcpp::RObject visual_init = visualization_init_to_r(result.visual_init, seed);
  const Rcpp::DataFrame timings = stage_timings_to_r(result.timings);
  const auto conversion_start = std::chrono::steady_clock::now();
  Rcpp::List out;
  if (storage == "handle") {
    out = graph_handle_metadata(std::move(result));
  } else {
    out = graph_to_r(result.knn, n);
    if (result.spatial_graph_builds > 0) {
      out["spatial_knn"] = graph_to_r(result.spatial_knn, n);
      out["spatial_jitter"] = Rcpp::NumericVector(
        result.spatial_jitter.begin(), result.spatial_jitter.end()
      );
      out["spatial_dimensions"] = result.spatial_dimensions;
    }
  }
  const double conversion_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - conversion_start
  ).count();
  out["visual_init"] = visual_init;
  out["samples"] = result.samples;
  out["dimensions"] = result.dimensions;
  out["backend"] = kodama::to_string(result.backend);
  out["index_type"] = kodama::to_string(result.index_type);
  out["ivf_nlist"] = result.ivf_nlist;
  out["ivf_nprobe"] = result.ivf_nprobe;
  out["ivf_pilot_recall"] = result.ivf_pilot_recall;
  out["metric"] = metric;
  out["graph_builds"] = result.graph_builds;
  out["spatial_graph_builds"] = result.spatial_graph_builds;
  out["graph_storage_bytes"] = static_cast<double>(result.graph_storage_bytes);
  out["runtime_seconds"] = result.runtime_seconds;
  out["timing"] = Rcpp::List::create(
    Rcpp::Named("input_copy_seconds") = result.input_copy_seconds,
    Rcpp::Named("graph_seconds") = result.graph_seconds,
    Rcpp::Named("spatial_graph_seconds") = result.spatial_graph_seconds,
    Rcpp::Named("visual_init_seconds") = result.visual_init_seconds,
    Rcpp::Named("r_graph_conversion_seconds") = conversion_seconds,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds
  );
  out["timings"] = timings;
  return out;
}

// [[Rcpp::export]]
Rcpp::List kodama_graph_materialize_cpp(SEXP graph_handle) {
  Rcpp::XPtr<RGraphHandle> handle = graph_handle_from_sexp(graph_handle);
  const auto start = std::chrono::steady_clock::now();
  Rcpp::List out = graph_to_r(
    kodama::KODAMAGraphMaterialize(handle->prepared), handle->prepared.samples);
  if (handle->prepared.spatial_graph_builds > 0) {
    out["spatial_knn"] = graph_to_r(
      handle->prepared.spatial_knn, handle->prepared.samples
    );
    out["spatial_jitter"] = Rcpp::NumericVector(
      handle->prepared.spatial_jitter.begin(),
      handle->prepared.spatial_jitter.end()
    );
    out["spatial_dimensions"] = handle->prepared.spatial_dimensions;
    out["spatial_graph_builds"] = handle->prepared.spatial_graph_builds;
  }
  out["conversion_seconds"] = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start
  ).count();
  return out;
}

// [[Rcpp::export]]
Rcpp::List kodama_pca_cpp(
  Rcpp::NumericMatrix data,
  int ncomp = 2,
  bool center = true,
  bool scale = false,
  std::string backend = "cpu",
  int seed = 4,
  int n_threads = 1,
  int gpu_device = 0,
  int oversample = -1,
  int power = -1
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::PCAOptions options;
  options.n_components = ncomp;
  options.center = center;
  options.scale = scale;
  options.backend = parse_backend(backend);
  options.seed = static_cast<std::uint64_t>(seed);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  options.oversample = oversample;
  options.power_iterations = power;
  return pca_to_r(kodama::PCA(
    kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
    options
  ));
}

// [[Rcpp::export]]
Rcpp::List kodama_normalization_cpp(
  Rcpp::NumericMatrix train,
  Rcpp::Nullable<Rcpp::NumericMatrix> test = R_NilValue,
  std::string method = "pqn",
  Rcpp::Nullable<Rcpp::NumericVector> reference = R_NilValue,
  std::string backend = "cpu",
  int n_threads = 1,
  int gpu_device = 0
) {
  const int train_rows = train.nrow();
  const int variables = train.ncol();
  std::vector<float> train_data = matrix_to_float(train);
  std::vector<float> test_data;
  int test_rows = 0;
  if (!test.isNull()) {
    Rcpp::NumericMatrix test_matrix(test);
    if (test_matrix.ncol() != variables) Rcpp::stop("Xtrain and Xtest must have the same variables.");
    test_rows = test_matrix.nrow();
    test_data = matrix_to_float(test_matrix);
  }
  kodama::NormalizationOptions options;
  options.method = parse_normalization_method(method);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  if (!reference.isNull()) {
    Rcpp::NumericVector values(reference);
    options.reference.assign(values.begin(), values.end());
  }
  const kodama::NormalizationResult result = kodama::Normalization(
    kodama::MatrixView{train_data.data(), static_cast<std::size_t>(train_rows),
      static_cast<std::size_t>(variables)},
    test_rows == 0 ? kodama::MatrixView{} : kodama::MatrixView{test_data.data(),
      static_cast<std::size_t>(test_rows), static_cast<std::size_t>(variables)},
    options
  );
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("newXtrain") = float_matrix_to_r(result.train, train_rows, variables),
    Rcpp::Named("coeXtrain") = Rcpp::NumericVector(
      result.train_coefficients.begin(), result.train_coefficients.end()),
    Rcpp::Named("reference") = Rcpp::NumericVector(result.reference.begin(), result.reference.end()),
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("precision") = "float32"
  );
  if (test_rows > 0) {
    out["newXtest"] = float_matrix_to_r(result.test, test_rows, variables);
    out["coeXtest"] = Rcpp::NumericVector(
      result.test_coefficients.begin(), result.test_coefficients.end());
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List kodama_scaling_cpp(
  Rcpp::NumericMatrix train,
  Rcpp::Nullable<Rcpp::NumericMatrix> test = R_NilValue,
  std::string method = "autoscaling",
  std::string backend = "cpu",
  int n_threads = 1,
  int gpu_device = 0
) {
  const int train_rows = train.nrow();
  const int variables = train.ncol();
  std::vector<float> train_data = matrix_to_float(train);
  std::vector<float> test_data;
  int test_rows = 0;
  if (!test.isNull()) {
    Rcpp::NumericMatrix test_matrix(test);
    if (test_matrix.ncol() != variables) Rcpp::stop("Xtrain and Xtest must have the same variables.");
    test_rows = test_matrix.nrow();
    test_data = matrix_to_float(test_matrix);
  }
  kodama::ScalingOptions options;
  options.method = parse_scaling_method(method);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  const kodama::ScalingResult result = kodama::Scaling(
    kodama::MatrixView{train_data.data(), static_cast<std::size_t>(train_rows),
      static_cast<std::size_t>(variables)},
    test_rows == 0 ? kodama::MatrixView{} : kodama::MatrixView{test_data.data(),
      static_cast<std::size_t>(test_rows), static_cast<std::size_t>(variables)},
    options
  );
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("newXtrain") = float_matrix_to_r(result.train, train_rows, variables),
    Rcpp::Named("center") = Rcpp::NumericVector(result.center.begin(), result.center.end()),
    Rcpp::Named("scale") = Rcpp::NumericVector(result.scale.begin(), result.scale.end()),
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("precision") = "float32"
  );
  if (test_rows > 0) out["newXtest"] = float_matrix_to_r(result.test, test_rows, variables);
  return out;
}

// [[Rcpp::export]]
Rcpp::List kodama_passing_message_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::NumericMatrix spatial,
  int number_knn = 15,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0
) {
  if (data.nrow() != spatial.nrow()) {
    Rcpp::stop("data and spatial must have the same number of rows.");
  }
  const int rows = data.nrow();
  const int variables = data.ncol();
  const int dimensions = spatial.ncol();
  std::vector<float> expression = matrix_to_float(data);
  std::vector<float> coordinates = matrix_to_float(spatial);
  std::vector<int> sample_ids;
  if (!samples.isNull()) {
    const Rcpp::IntegerVector values(samples);
    sample_ids.assign(values.begin(), values.end());
  }
  kodama::PassingMessageOptions options;
  options.neighbors = number_knn;
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  const kodama::PassingMessageResult result = kodama::PassingMessage(
    kodama::MatrixView{expression.data(), static_cast<std::size_t>(rows),
      static_cast<std::size_t>(variables)},
    kodama::MatrixView{coordinates.data(), static_cast<std::size_t>(rows),
      static_cast<std::size_t>(dimensions)},
    sample_ids, options
  );
  return Rcpp::List::create(
    Rcpp::Named("data") = float_matrix_to_r(result.values, rows, variables),
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("sample_groups") = static_cast<double>(result.sample_groups),
    Rcpp::Named("sample_max_distances") = Rcpp::NumericVector(
      result.sample_max_distances.begin(), result.sample_max_distances.end()),
    Rcpp::Named("graph_seconds") = result.graph_seconds,
    Rcpp::Named("aggregation_seconds") = result.aggregation_seconds,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("precision") = "float32"
  );
}

// [[Rcpp::export]]
Rcpp::List kodama_spatial_features_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::NumericMatrix spatial,
  Rcpp::Nullable<Rcpp::IntegerVector> samples = R_NilValue,
  int n_threads = 4,
  bool require_nonzero_each_sample = true
) {
  if (data.nrow() != spatial.nrow()) {
    Rcpp::stop("data and spatial must have the same number of rows.");
  }
  const int rows = data.nrow();
  const int variables = data.ncol();
  std::vector<float> expression = matrix_to_float(data);
  std::vector<float> coordinates = matrix_to_float(spatial);
  std::vector<int> sample_ids;
  if (!samples.isNull()) {
    const Rcpp::IntegerVector values(samples);
    sample_ids.assign(values.begin(), values.end());
  }
  kodama::SpatialFeatureOptions options;
  options.n_threads = n_threads;
  options.require_nonzero_each_sample = require_nonzero_each_sample;
  const kodama::SpatialFeatureResult result = kodama::SpatialFeatureSelection_CPU(
    kodama::MatrixView{expression.data(), static_cast<std::size_t>(rows),
      static_cast<std::size_t>(variables)},
    kodama::MatrixView{coordinates.data(), static_cast<std::size_t>(rows),
      static_cast<std::size_t>(spatial.ncol())},
    sample_ids, options
  );
  Rcpp::IntegerVector ranking(result.ranking.begin(), result.ranking.end());
  for (int& value : ranking) ++value;
  Rcpp::IntegerVector basis_dimensions(
    result.basis_dimensions.begin(), result.basis_dimensions.end());
  Rcpp::NumericMatrix per_sample_p(
    static_cast<int>(result.sample_groups), variables);
  for (int sample = 0; sample < static_cast<int>(result.sample_groups); ++sample) {
    for (int variable = 0; variable < variables; ++variable) {
      per_sample_p(sample, variable) = result.per_sample_p_value[
        static_cast<std::size_t>(sample) * variables + variable];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("score") = Rcpp::NumericVector(result.score.begin(), result.score.end()),
    Rcpp::Named("p.value") = Rcpp::NumericVector(result.p_value.begin(), result.p_value.end()),
    Rcpp::Named("adjusted.p.value") = Rcpp::NumericVector(
      result.adjusted_p_value.begin(), result.adjusted_p_value.end()),
    Rcpp::Named("ranking") = ranking,
    Rcpp::Named("per.sample.score") = float_matrix_to_r(
      result.per_sample_score, static_cast<int>(result.sample_groups), variables),
    Rcpp::Named("per.sample.p.value") = per_sample_p,
    Rcpp::Named("sample.labels") = Rcpp::IntegerVector(
      result.sample_labels.begin(), result.sample_labels.end()),
    Rcpp::Named("basis.dimensions") = basis_dimensions,
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("basis.seconds") = result.basis_seconds,
    Rcpp::Named("statistic.seconds") = result.statistic_seconds,
    Rcpp::Named("runtime.seconds") = result.runtime_seconds,
    Rcpp::Named("precision") = "float32"
  );
}

// [[Rcpp::export]]
Rcpp::List kodama_visual_init_cpp(
  Rcpp::NumericMatrix data,
  std::string backend = "cpu",
  int seed = 4,
  int n_threads = 1,
  int gpu_device = 0
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::VisualizationInitOptions options;
  options.n_components = 2;
  options.backend = parse_backend(backend);
  options.seed = static_cast<std::uint64_t>(seed);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  return visualization_init_to_r(
    kodama::KODAMAVisualizationPCAInit(
      kodama::MatrixView{
        x.data(),
        static_cast<std::size_t>(n),
        static_cast<std::size_t>(p)
      },
      options
    ),
    seed
  );
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_umap_cpp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 30,
  int n_epochs = 200,
  double learning_rate = 1.0,
  double min_dist = 0.01,
  double repulsion_strength = 1.0,
  int negative_sample_rate = 5,
  int spectral_n_iter = 20,
  int n_threads = 1,
  int seed = 1234,
  std::string backend = "cpu",
  int gpu_device = 0,
  std::string graph_mode = "fuzzy",
  std::string init_source = "",
  std::string init_backend = "auto"
) {
  kodama::UMAPOptions options;
  options.n_neighbors = n_neighbors;
  options.n_epochs = n_epochs;
  options.learning_rate = learning_rate;
  options.min_dist = min_dist;
  options.repulsion_strength = repulsion_strength;
  options.negative_sample_rate = negative_sample_rate;
  options.spectral_n_iter = spectral_n_iter;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  options.graph_mode = parse_umap_graph_mode(graph_mode);
  options.init_source = init_source;
  options.init_backend = parse_backend(init_backend);
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != indices.nrow() || init_matrix.ncol() != 2) Rcpp::stop("init must have nrow(indices) rows and 2 columns.");
    options.init.assign(static_cast<std::size_t>(init_matrix.nrow()) * 2u, 0.0f);
    for (int i = 0; i < init_matrix.nrow(); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = static_cast<float>(init_matrix(i, 0));
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(init_matrix(i, 1));
    }
  }
  const kodama::NeighborGraph graph = graph_from_r(indices, distances);
  const kodama::Backend selected = parse_backend(backend);
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAUMAP_CUDA(graph, options) :
    selected == kodama::Backend::Metal ?
      kodama::KODAMAUMAP_METAL(graph, options) :
      kodama::KODAMAUMAP_CPU(graph, options);
  return embedding_to_r(result);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_umap_graph_handle_cpp(
  SEXP graph_handle,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 30,
  int n_epochs = 200,
  double learning_rate = 1.0,
  double min_dist = 0.01,
  double repulsion_strength = 1.0,
  int negative_sample_rate = 5,
  int spectral_n_iter = 20,
  int n_threads = 1,
  int seed = 1234,
  std::string backend = "cpu",
  int gpu_device = 0,
  std::string graph_mode = "fuzzy",
  std::string init_source = "",
  std::string init_backend = "auto"
) {
  Rcpp::XPtr<RGraphHandle> handle = graph_handle_from_sexp(graph_handle);
  kodama::UMAPOptions options;
  options.n_neighbors = n_neighbors;
  options.n_epochs = n_epochs;
  options.learning_rate = learning_rate;
  options.min_dist = min_dist;
  options.repulsion_strength = repulsion_strength;
  options.negative_sample_rate = negative_sample_rate;
  options.spectral_n_iter = spectral_n_iter;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  options.graph_mode = parse_umap_graph_mode(graph_mode);
  options.init_source = init_source;
  options.init_backend = parse_backend(init_backend);
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != handle->prepared.samples || init_matrix.ncol() != 2) {
      Rcpp::stop("init must have the graph sample count rows and 2 columns.");
    }
    options.init = matrix_to_float(init_matrix);
  }
  const kodama::NeighborGraph materialized =
    kodama::KODAMAGraphMaterialize(handle->prepared);
  const kodama::Backend selected = parse_backend(backend);
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAUMAP_CUDA(materialized, options) :
    selected == kodama::Backend::Metal ?
      kodama::KODAMAUMAP_METAL(materialized, options) :
      kodama::KODAMAUMAP_CPU(materialized, options);
  return embedding_to_r(result);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_opentsne_cpp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 0,
  double perplexity = 30.0,
  double theta = 0.5,
  int early_exaggeration_iter = 250,
  int n_iter = 500,
  double early_exaggeration = 12.0,
  double exaggeration = 1.0,
  double learning_rate = 0.0,
  bool learning_rate_auto = true,
  double initial_momentum = 0.8,
  double final_momentum = 0.8,
  double min_gain = 0.01,
  double max_step_norm = 5.0,
  int n_threads = 1,
  int seed = 4,
  std::string backend = "cpu",
  int gpu_device = 0,
  std::string init_source = "",
  std::string init_backend = "auto"
) {
  kodama::OpenTSNEOptions options;
  options.n_neighbors = n_neighbors;
  options.perplexity = perplexity;
  options.theta = theta;
  options.early_exaggeration_iter = early_exaggeration_iter;
  options.n_iter = n_iter;
  options.early_exaggeration = early_exaggeration;
  options.exaggeration = exaggeration;
  options.learning_rate = learning_rate;
  options.learning_rate_auto = learning_rate_auto;
  options.initial_momentum = initial_momentum;
  options.final_momentum = final_momentum;
  options.min_gain = min_gain;
  options.max_step_norm = max_step_norm;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  options.init_source = init_source;
  options.init_backend = parse_backend(init_backend);
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != indices.nrow() || init_matrix.ncol() != 2) Rcpp::stop("init must have nrow(indices) rows and 2 columns.");
    options.init.assign(static_cast<std::size_t>(init_matrix.nrow()) * 2u, 0.0f);
    for (int i = 0; i < init_matrix.nrow(); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = static_cast<float>(init_matrix(i, 0));
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(init_matrix(i, 1));
    }
  }
  const kodama::NeighborGraph graph = graph_from_r(indices, distances);
  const kodama::Backend selected = parse_backend(backend);
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAOpenTSNE_CUDA(graph, options) :
    selected == kodama::Backend::Metal ?
      kodama::KODAMAOpenTSNE_METAL(graph, options) :
      kodama::KODAMAOpenTSNE_CPU(graph, options);
  return embedding_to_r(result);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_opentsne_graph_handle_cpp(
  SEXP graph_handle,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 0,
  double perplexity = 30.0,
  double theta = 0.5,
  int early_exaggeration_iter = 250,
  int n_iter = 500,
  double early_exaggeration = 12.0,
  double exaggeration = 1.0,
  double learning_rate = 0.0,
  bool learning_rate_auto = true,
  double initial_momentum = 0.8,
  double final_momentum = 0.8,
  double min_gain = 0.01,
  double max_step_norm = 5.0,
  int n_threads = 1,
  int seed = 4,
  std::string backend = "cpu",
  int gpu_device = 0,
  std::string init_source = "",
  std::string init_backend = "auto"
) {
  Rcpp::XPtr<RGraphHandle> handle = graph_handle_from_sexp(graph_handle);
  kodama::OpenTSNEOptions options;
  options.n_neighbors = n_neighbors;
  options.perplexity = perplexity;
  options.theta = theta;
  options.early_exaggeration_iter = early_exaggeration_iter;
  options.n_iter = n_iter;
  options.early_exaggeration = early_exaggeration;
  options.exaggeration = exaggeration;
  options.learning_rate = learning_rate;
  options.learning_rate_auto = learning_rate_auto;
  options.initial_momentum = initial_momentum;
  options.final_momentum = final_momentum;
  options.min_gain = min_gain;
  options.max_step_norm = max_step_norm;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  options.init_source = init_source;
  options.init_backend = parse_backend(init_backend);
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != handle->prepared.samples || init_matrix.ncol() != 2) {
      Rcpp::stop("init must have the graph sample count rows and 2 columns.");
    }
    options.init = matrix_to_float(init_matrix);
  }
  const kodama::NeighborGraph materialized =
    kodama::KODAMAGraphMaterialize(handle->prepared);
  const kodama::Backend selected = parse_backend(backend);
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAOpenTSNE_CUDA(materialized, options) :
    selected == kodama::Backend::Metal ?
      kodama::KODAMAOpenTSNE_METAL(materialized, options) :
      kodama::KODAMAOpenTSNE_CPU(materialized, options);
  return embedding_to_r(result);
}

// [[Rcpp::export]]
Rcpp::List kodama_graph_cluster_cpp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  std::string weight = "distance",
  int n_threads = 4,
  int n_iterations = 10,
  int random_walk_steps = 4,
  int n_clusters = 0,
  double prune = 0.0,
  bool mutual = false
) {
  kodama::GraphClusterOptions options;
  options.backend = kodama::Backend::CPU;
  options.weight_type = parse_graph_weight_type(weight);
  options.n_threads = n_threads;
  options.n_iterations = n_iterations;
  options.random_walk_steps = random_walk_steps;
  options.target_clusters = n_clusters;
  options.prune = prune;
  options.mutual = mutual;
  const kodama::NeighborGraph graph = graph_from_r(indices, distances);
  return graph_cluster_result_to_r(kodama::KODAMAGraphCluster(graph, indices.nrow(), options));
}

// [[Rcpp::export]]
Rcpp::List kodama_graph_handle_cluster_cpp(
  SEXP graph_handle,
  std::string weight = "distance",
  int n_threads = 4,
  int n_iterations = 10,
  int random_walk_steps = 4,
  int n_clusters = 0,
  double prune = 0.0,
  bool mutual = false
) {
  Rcpp::XPtr<RGraphHandle> handle = graph_handle_from_sexp(graph_handle);
  kodama::GraphClusterOptions options;
  options.backend = kodama::Backend::CPU;
  options.weight_type = parse_graph_weight_type(weight);
  options.n_threads = n_threads;
  options.n_iterations = n_iterations;
  options.random_walk_steps = random_walk_steps;
  options.target_clusters = n_clusters;
  options.prune = prune;
  options.mutual = mutual;
  const kodama::NeighborGraph materialized =
    kodama::KODAMAGraphMaterialize(handle->prepared);
  return graph_cluster_result_to_r(
    kodama::KODAMAGraphCluster(
      materialized, handle->prepared.samples, options
    )
  );
}

// [[Rcpp::export]]
Rcpp::List kodama_embedding_cluster_cpp(
  Rcpp::NumericMatrix embedding,
  std::string graph_backend = "cpu",
  std::string weight = "distance",
  std::string metric = "euclidean",
  int k = 30,
  int n_threads = 4,
  int n_iterations = 10,
  int random_walk_steps = 4,
  int n_clusters = 0,
  double prune = 0.0,
  bool mutual = false,
  int gpu_device = 0
) {
  const int n = embedding.nrow();
  const int p = embedding.ncol();
  std::vector<float> x = matrix_to_float(embedding);
  kodama::GraphClusterOptions graph_options;
  graph_options.k = k;
  graph_options.metric = parse_metric(metric);
  graph_options.backend = parse_backend(graph_backend);
  graph_options.n_threads = n_threads;
  graph_options.gpu_device = gpu_device;
  const kodama::NeighborGraph graph = kodama::KODAMAKNNGraph(
    kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
    graph_options
  );
  kodama::GraphClusterOptions cluster_options = graph_options;
  cluster_options.backend = kodama::Backend::CPU;
  cluster_options.weight_type = parse_graph_weight_type(weight);
  cluster_options.n_iterations = n_iterations;
  cluster_options.random_walk_steps = random_walk_steps;
  cluster_options.target_clusters = n_clusters;
  cluster_options.prune = prune;
  cluster_options.mutual = mutual;
  cluster_options.gpu_device = gpu_device;
  Rcpp::List out = graph_cluster_result_to_r(
    kodama::KODAMAEmbeddingGraphCluster(
      kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
      graph,
      cluster_options
    )
  );
  out["graph"] = graph_to_r(graph, n);
  return out;
}
