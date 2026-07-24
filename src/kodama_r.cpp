// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <Rcpp.h>

#include "kodama/kodama.hpp"

namespace {

kodama::DistanceMetric parse_metric(const std::string& metric) {
  if (metric == "euclidean" || metric == "l2") return kodama::DistanceMetric::Euclidean;
  if (metric == "cosine") return kodama::DistanceMetric::Cosine;
  if (metric == "inner_product" || metric == "ip") return kodama::DistanceMetric::InnerProduct;
  Rcpp::stop("Unsupported metric: " + metric);
}

kodama::Backend parse_backend(const std::string& backend) {
  if (backend == "cpu") return kodama::Backend::CPU;
  if (backend == "cuda") return kodama::Backend::CUDA;
  if (backend == "metal") return kodama::Backend::Metal;
  Rcpp::stop("Unsupported backend: " + backend);
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
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      out[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)] =
        static_cast<float>(data(i, j));
    }
  }
  return out;
}

kodama::NeighborGraph graph_from_r(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions.");
  }
  kodama::NeighborGraph graph;
  graph.neighbors = indices.ncol();
  graph.indices.assign(static_cast<std::size_t>(indices.nrow()) * static_cast<std::size_t>(indices.ncol()), 0);
  graph.distances.assign(static_cast<std::size_t>(distances.nrow()) * static_cast<std::size_t>(distances.ncol()), 0.0f);
  for (int i = 0; i < indices.nrow(); ++i) {
    for (int j = 0; j < indices.ncol(); ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(indices.ncol()) + static_cast<std::size_t>(j);
      graph.indices[offset] = indices(i, j);
      graph.distances[offset] = static_cast<float>(distances(i, j));
    }
  }
  return graph;
}

Rcpp::List graph_to_r(const kodama::NeighborGraph& graph, int samples) {
  Rcpp::IntegerMatrix indices(samples, graph.neighbors);
  Rcpp::NumericMatrix distances(samples, graph.neighbors);
  for (int i = 0; i < samples; ++i) {
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * graph.neighbors + j;
      indices(i, j) = graph.indices[offset];
      distances(i, j) = graph.distances[offset];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("neighbors") = graph.neighbors
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
    Rcpp::Named("cycles_completed") = result.cycles_completed,
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
  return out;
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
  const kodama::KODAMAMatrixResult& result,
  const kodama::KODAMAMatrixOptions& options
) {
  Rcpp::NumericVector acc(result.acc.begin(), result.acc.end());
  Rcpp::NumericMatrix v(result.runs, result.cycles);
  for (int i = 0; i < result.runs; ++i) {
    for (int j = 0; j < result.cycles; ++j) {
      v(i, j) = result.v[static_cast<std::size_t>(i) * result.cycles + j];
    }
  }

  Rcpp::IntegerMatrix res(result.runs, result.samples);
  Rcpp::IntegerMatrix res_constrain(result.runs, result.samples);
  for (int i = 0; i < result.runs; ++i) {
    for (int j = 0; j < result.samples; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * result.samples + static_cast<std::size_t>(j);
      res(i, j) = result.res[offset];
      res_constrain(i, j) = result.res_constrain[offset];
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("acc") = acc,
    Rcpp::Named("v") = v,
    Rcpp::Named("res") = res,
    Rcpp::Named("knn") = graph_to_r(result.knn, result.samples),
    Rcpp::Named("base_knn") = graph_to_r(result.base_knn, result.samples),
    Rcpp::Named("res_constrain") = res_constrain,
    Rcpp::Named("n.cores") = result.n_threads,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("analysis_storage") = "float32",
    Rcpp::Named("classifier") = kodama::to_string(options.classifier),
    Rcpp::Named("backend") = kodama::to_string(options.backend),
    Rcpp::Named("graph_feature_mode") = kodama::to_string(options.graph_feature_mode),
    Rcpp::Named("timing") = Rcpp::List::create(
      Rcpp::Named("input_copy_seconds") = result.input_copy_seconds,
      Rcpp::Named("graph_feature_seconds") = result.graph_feature_seconds,
      Rcpp::Named("spatial_precompute_seconds") = result.spatial_precompute_seconds,
      Rcpp::Named("graph_seconds") = result.graph_seconds,
      Rcpp::Named("spatial_graph_seconds") = result.spatial_graph_seconds,
      Rcpp::Named("optimization_wall_seconds") = result.optimization_wall_seconds,
      Rcpp::Named("optimization_sum_seconds") = result.optimization_sum_seconds,
      Rcpp::Named("dissimilarity_seconds") = result.dissimilarity_seconds,
      Rcpp::Named("runtime_seconds") = result.runtime_seconds
    ),
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb
  );
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List kodama_matrix_cpp(
  Rcpp::NumericMatrix data,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
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
  std::string metric = "euclidean",
  std::string classifier = "knn",
  std::string backend = "cpu",
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true
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
  options.n_threads = n_cores;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;

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

  return kodama_matrix_result_to_r(result, options);
}

// [[Rcpp::export]]
Rcpp::List kodama_matrix_graph_cpp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::NumericMatrix> data = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
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
  std::string classifier = "knn",
  std::string backend = "cpu",
  std::string graph_feature_mode = "laplacian_self_tuning",
  int graph_feature_components = 0,
  int graph_feature_steps = 3,
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true
) {
  const int n = indices.nrow();
  kodama::NeighborGraph graph = graph_from_r(indices, distances);
  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.n_threads = n_cores;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = kodama::DistanceMetric::Euclidean;
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;
  options.graph_feature_mode = parse_graph_feature_mode(graph_feature_mode);
  options.graph_feature_components = graph_feature_components;
  options.graph_feature_steps = graph_feature_steps;
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
    result = kodama::KODAMAMatrixFromGraphData(view, graph, labels, constraints, fixed, options);
  } else {
    result = kodama::KODAMAMatrixFromGraph(graph, n, labels, constraints, fixed, options);
  }
  return kodama_matrix_result_to_r(result, options);
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
  int k = 30,
  std::string metric = "euclidean",
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x = matrix_to_float(data);
  kodama::GraphClusterOptions options;
  options.k = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  const kodama::NeighborGraph graph = kodama::KODAMAKNNGraph(
    kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)},
    options
  );
  Rcpp::List out = graph_to_r(graph, n);
  out["backend"] = kodama::to_string(options.backend);
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
  std::string graph_mode = "binary"
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
  if (selected == kodama::Backend::Metal) {
    Rcpp::stop("KODAMA UMAP currently supports CPU and CUDA backends, not Metal.");
  }
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAUMAP_CUDA(graph, options) :
    kodama::KODAMAUMAP_CPU(graph, options);
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
  int gpu_device = 0
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
  if (selected == kodama::Backend::Metal) {
    Rcpp::stop("KODAMA openTSNE currently supports CPU and CUDA backends, not Metal.");
  }
  const kodama::EmbeddingResult result = selected == kodama::Backend::CUDA ?
    kodama::KODAMAOpenTSNE_CUDA(graph, options) :
    kodama::KODAMAOpenTSNE_CPU(graph, options);
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
