// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include "kodama/version.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kodama {

namespace detail {
struct KODAMAGraphHandleAccess;
}

enum class Backend {
  Auto,
  CPU,
  CUDA,
  Metal
};

enum class DistanceMetric {
  Cosine,
  InnerProduct,
  Euclidean
};

enum class KNNIndexType {
  PrecomputedGraph,
  NativeHNSW,
  CudaExact,
  CudaIVFFlat,
  MetalExact,
  MetalIVFFlat
};

enum class PLSMode {
  PLS_DA,
  PLS_LDA
};

enum class CoreClassifier {
  PLS_LDA,
  KNN
};

struct EvolutionPolicy {
  bool prediction_guidance = true;
  bool adaptive_proposal_size = true;
  bool transition_proposal = true;
  bool stochastic_acceptance = true;
  bool diversity_multiplier = true;
  bool pls_transition_coarsening = true;
  bool pls_fragmentation_penalty = true;
  bool reject_single_class = true;

  static EvolutionPolicy standard();
  static EvolutionPolicy from_name(const std::string& name);
};

enum class GraphWeightType {
  SNN,
  Distance,
  Adaptive,
  Binary
};

enum class GraphFeatureMode {
  LaplacianSelfTuning
};

enum class SpatialCoordinateMode {
  Standard,
  Population
};

enum class UMAPGraphMode {
  Binary,
  Fuzzy
};

enum class MatrixValueType {
  Float64,
  Float32
};

enum class GraphIndexBase {
  Auto,
  Zero,
  One
};

enum class NormalizationMethod {
  PQN,
  Sum,
  Median,
  Sqrt,
  None
};

enum class ScalingMethod {
  None,
  Centering,
  Autoscaling,
  RangeScaling,
  ParetoScaling
};

struct MatrixView {
  const void* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  MatrixValueType value_type = MatrixValueType::Float64;

  MatrixView() = default;

  MatrixView(const double* data_ptr, std::size_t n_rows, std::size_t n_cols)
      : data(data_ptr), rows(n_rows), cols(n_cols), value_type(MatrixValueType::Float64) {}

  MatrixView(const float* data_ptr, std::size_t n_rows, std::size_t n_cols)
      : data(data_ptr), rows(n_rows), cols(n_cols), value_type(MatrixValueType::Float32) {}

  float value_float(std::size_t i, std::size_t j) const {
    const std::size_t offset = i * cols + j;
    if (value_type == MatrixValueType::Float32) {
      return static_cast<const float*>(data)[offset];
    }
    return static_cast<float>(static_cast<const double*>(data)[offset]);
  }

  double operator()(std::size_t i, std::size_t j) const {
    const std::size_t offset = i * cols + j;
    if (value_type == MatrixValueType::Float32) {
      return static_cast<double>(static_cast<const float*>(data)[offset]);
    }
    return static_cast<const double*>(data)[offset];
  }
};

struct FoldOptions {
  int folds = 10;
  bool stratified = true;
  std::uint64_t seed = 1;
};

struct KNNOptions {
  FoldOptions cv;
  int k = 10;
  DistanceMetric metric = DistanceMetric::Cosine;
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  int hnsw_m = 0;
  int hnsw_ef_construction = 0;
  int hnsw_ef_search = 0;
  int hnsw_tune_k = 50;
  double hnsw_target_recall = 0.99;
  int gpu_device = 0;
  int n_threads = 1;
};

struct PLSOptions {
  FoldOptions cv;
  int max_components = 10;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  Backend backend = Backend::CPU;
  int gpu_device = 0;
  int n_threads = 1;
  // Nonzero values scope reusable fold workspaces to one immutable data epoch.
  std::uint64_t data_epoch = 0;
};

struct CorePLSLDAOptions {
  FoldOptions cv;
  int max_components = 10;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  Backend backend = Backend::CPU;
  int gpu_device = 0;
  int n_threads = 1;
  std::uint64_t data_epoch = 0;

  CorePLSLDAOptions() = default;

  CorePLSLDAOptions& operator=(const PLSOptions& options) {
    cv = options.cv;
    max_components = options.max_components;
    fixed_components = options.fixed_components;
    center = options.center;
    scale = options.scale;
    backend = options.backend;
    gpu_device = options.gpu_device;
    n_threads = options.n_threads;
    data_epoch = options.data_epoch;
    return *this;
  }

  operator PLSOptions() const {
    PLSOptions out;
    out.cv = cv;
    out.max_components = max_components;
    out.fixed_components = fixed_components;
    out.center = center;
    out.scale = scale;
    out.backend = backend;
    out.gpu_device = gpu_device;
    out.n_threads = n_threads;
    out.data_epoch = data_epoch;
    return out;
  }
};

struct FoldResult {
  int fold = 0;
  int n_train = 0;
  int n_validation = 0;
  double accuracy = 0.0;
};

struct ConfusionMatrix {
  std::vector<int> labels;
  std::vector<int> counts;
  std::size_t n_labels = 0;

  int operator()(std::size_t truth, std::size_t predicted) const {
    return counts[truth * n_labels + predicted];
  }
};

struct KNNParametersUsed {
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  DistanceMetric metric = DistanceMetric::Cosine;
  int k = 10;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  double ivf_pilot_recall = 0.0;
  int hnsw_m = 0;
  int hnsw_ef_construction = 0;
  int hnsw_ef_search = 0;
  int hnsw_tune_k = 50;
  double hnsw_target_recall = 0.99;
  int gpu_device = 0;
  int n_threads = 1;
};

struct KNNCVResult {
  std::vector<int> predicted_labels;
  std::vector<int> true_labels;
  std::vector<int> fold_assignments;
  std::vector<FoldResult> folds;
  double global_accuracy = 0.0;
  ConfusionMatrix confusion;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
  KNNParametersUsed parameters;
};

struct PLSParametersUsed {
  Backend backend = Backend::CPU;
  PLSMode mode = PLSMode::PLS_DA;
  int max_components = 10;
  int selected_components = 1;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  int gpu_device = 0;
  int n_threads = 1;
};

struct PLSCVResult {
  std::vector<int> predicted_labels;
  std::vector<int> true_labels;
  std::vector<int> fold_assignments;
  std::vector<FoldResult> folds;
  std::vector<double> accuracy_by_components;
  int selected_components = 1;
  double global_accuracy = 0.0;
  ConfusionMatrix confusion;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
  PLSParametersUsed parameters;
};

struct CoreOptions {
  int cycles = 100;
  bool progress = false;
  int progress_run = 0;
  int progress_runs = 0;
  CoreClassifier classifier = CoreClassifier::PLS_LDA;
  bool shake = false;
  std::uint64_t seed = 1;
  bool auto_class_coarsening = false;
  bool many_to_one_absorption = false;
  bool evolutionary_search = false;
  bool guarded_diversity = false;
  bool adaptive_proposal_size = true;
  EvolutionPolicy evolution = EvolutionPolicy::standard();
  CorePLSLDAOptions pls;
  KNNOptions knn = [] {
    KNNOptions options;
    options.k = 30;
    return options;
  }();
};

struct CoreResult {
  std::vector<int> clbest;
  std::vector<int> clbest_dirty;
  std::vector<int> cvpredbest;
  double accbest = 0.0;
  double scorebest = 0.0;
  std::vector<double> vect_acc;
  std::vector<double> vect_score;
  std::vector<int> proposal_size;
  std::vector<int> active_classes;
  std::vector<unsigned char> accepted;
  std::vector<unsigned char> improving_acceptance;
  std::vector<unsigned char> temperature_acceptance;
  std::vector<int> fold_assignments;
  int cycles_completed = 0;
  int proposals_evaluated = 0;
  int best_state_updates = 0;
  int current_state_accepts = 0;
  int stochastic_state_attempts = 0;
  int stochastic_state_accepts = 0;
  int current_state_rejections = 0;
  int coarsening_moves = 0;
  int absorption_moves = 0;
  int transition_attempted = 0;
  int transition_accepted = 0;
  int many_to_one_attempted = 0;
  int many_to_one_accepted = 0;
  int pls_coarsening_attempted = 0;
  int pls_coarsening_accepted = 0;
  int cv_evaluations = 0;
  bool success = false;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
};

struct NeighborGraph {
  std::vector<int> indices;
  std::vector<float> distances;
  int neighbors = 0;
  GraphIndexBase index_base = GraphIndexBase::Auto;
};

struct ResidentIVFSearchStats {
  Backend backend = Backend::CPU;
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
  double search_seconds = 0.0;
};

/**
 * Move-only owner of an accelerator-resident IVF-Flat index.
 *
 * The training matrix, projected matrix, centroids, inverted-list offsets,
 * and inverted-list identifiers remain allocated on the selected CUDA or
 * Metal device until this object is destroyed.
 */
class ResidentIVFIndex {
 public:
  ResidentIVFIndex();
  ~ResidentIVFIndex();
  ResidentIVFIndex(ResidentIVFIndex&&) noexcept;
  ResidentIVFIndex& operator=(ResidentIVFIndex&&) noexcept;

  ResidentIVFIndex(const ResidentIVFIndex&) = delete;
  ResidentIVFIndex& operator=(const ResidentIVFIndex&) = delete;

  bool valid() const noexcept;
  Backend backend() const noexcept;
  DistanceMetric metric() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int nlist() const noexcept;
  double build_seconds() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit ResidentIVFIndex(std::unique_ptr<Impl> impl);

  friend ResidentIVFIndex BuildResidentIVFIndex(
    MatrixView,
    const KNNOptions&
  );
  friend NeighborGraph SearchResidentIVFIndex(
    const ResidentIVFIndex&,
    MatrixView,
    int,
    ResidentIVFSearchStats*
  );
  friend NeighborGraph SearchResidentIVFIndexSelf(
    const ResidentIVFIndex&,
    int,
    bool,
    ResidentIVFSearchStats*
  );
};

struct GraphClusterOptions {
  Backend backend = Backend::CPU;
  GraphWeightType weight_type = GraphWeightType::Distance;
  DistanceMetric metric = DistanceMetric::Euclidean;
  int k = 30;
  int n_threads = 1;
  int n_iterations = 10;
  int random_walk_steps = 4;
  int target_clusters = 0;
  int gpu_device = 0;
  double prune = 0.0;
  bool mutual = false;
};

struct GraphClusterResult {
  std::vector<int> membership;
  double modularity = 0.0;
  int n_communities = 0;
  int n_vertices = 0;
  int n_edges = 0;
  int target_clusters = 0;
  int target_gap = 0;
  bool target_exact = true;
  double runtime_seconds = 0.0;
  Backend backend = Backend::CPU;
};

struct VisualizationInitOptions {
  int n_components = 2;
  int n_threads = 1;
  std::uint64_t seed = 4;
  int gpu_device = 0;
  Backend backend = Backend::CPU;
};

struct VisualizationInitResult {
  std::vector<float> umap;
  std::vector<float> opentsne;
  int samples = 0;
  int components = 2;
  Backend backend = Backend::CPU;
  double runtime_seconds = 0.0;
};

struct KODAMAStageTiming {
  std::string step;
  double wall_seconds = 0.0;
  double accumulated_seconds = 0.0;
};

struct KODAMAGraphOptions {
  int neighbors = 100;
  int n_threads = 1;
  std::uint64_t seed = 1234;
  DistanceMetric metric = DistanceMetric::Euclidean;
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  int gpu_device = 0;
  bool materialize_graph = false;
  std::vector<int> samples;
};

struct KODAMAGraphResult;

class KODAMAGraphHandle {
 public:
  struct Impl;

  KODAMAGraphHandle();
  ~KODAMAGraphHandle();
  KODAMAGraphHandle(const KODAMAGraphHandle&) noexcept;
  KODAMAGraphHandle& operator=(const KODAMAGraphHandle&) noexcept;
  KODAMAGraphHandle(KODAMAGraphHandle&&) noexcept;
  KODAMAGraphHandle& operator=(KODAMAGraphHandle&&) noexcept;

  bool valid() const noexcept;
  Backend backend() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  bool host_materialized() const noexcept;

 private:
  explicit KODAMAGraphHandle(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend struct detail::KODAMAGraphHandleAccess;
  friend NeighborGraph KODAMAGraphMaterialize(const KODAMAGraphResult& graph);
};

struct KODAMAGraphResult {
  std::vector<KODAMAStageTiming> timings;
  std::shared_ptr<KODAMAGraphHandle> handle;
  NeighborGraph knn;
  NeighborGraph spatial_knn;
  std::vector<float> spatial_jitter;
  VisualizationInitResult visual_init;
  int samples = 0;
  int dimensions = 0;
  int neighbors = 0;
  int spatial_dimensions = 0;
  int graph_builds = 0;
  int spatial_graph_builds = 0;
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  double ivf_pilot_recall = 0.0;
  double input_copy_seconds = 0.0;
  double graph_seconds = 0.0;
  double spatial_graph_seconds = 0.0;
  double visual_init_seconds = 0.0;
  double runtime_seconds = 0.0;
  std::uint64_t graph_storage_bytes = 0;
};

struct KODAMAMatrixOptions {
  int runs = 100;
  int cycles = 20;
  int components = 50;
  int landmarks = 10000;
  int splitting = 0;
  int graph_neighbors = 100;
  int folds = 5;
  int n_threads = 1;
  int spatial_cols = 0;
  double spatial_resolution = 0.4;
  bool spatial_graph_mix = false;
  int spatial_constraint_mode = 0;
  SpatialCoordinateMode spatial_coordinate_mode = SpatialCoordinateMode::Standard;
  std::uint64_t seed = 1234;
  DistanceMetric metric = DistanceMetric::Euclidean;
  Backend backend = Backend::CPU;
  CoreClassifier classifier = CoreClassifier::KNN;
  EvolutionPolicy evolution = EvolutionPolicy::standard();
  bool progress = false;
  bool apply_kodama_dissimilarity = true;
  bool compute_visual_init = true;
  bool materialize_graph = false;
  GraphFeatureMode graph_feature_mode = GraphFeatureMode::LaplacianSelfTuning;
  int graph_feature_components = 0;
  int graph_feature_steps = 3;
  std::vector<float> spatial;
  std::vector<int> samples;
  KNNOptions knn = [] {
    KNNOptions options;
    options.k = 30;
    return options;
  }();
  CorePLSLDAOptions pls;
};

struct CoreCycleDiagnostic {
  int run = 0;
  int cycle = 0;
  int proposal_size = 0;
  int active_classes = 0;
  unsigned char accepted = 0;
  unsigned char improving_acceptance = 0;
  unsigned char temperature_acceptance = 0;
};

struct CoreRunDiagnostic {
  int run = 0;
  int cycles_completed = 0;
  int transition_attempted = 0;
  int transition_accepted = 0;
  int many_to_one_attempted = 0;
  int many_to_one_accepted = 0;
  int pls_coarsening_attempted = 0;
  int pls_coarsening_accepted = 0;
  int cv_evaluations = 0;
  std::uint64_t landmark_rows_hash = 0;
  std::uint64_t initial_labels_hash = 0;
  std::uint64_t fold_assignments_hash = 0;
};

struct KODAMAMatrixResult {
  std::vector<KODAMAStageTiming> timings;
  std::vector<double> acc;
  std::vector<double> v;
  std::vector<int> res;
  std::vector<int> res_constrain;
  std::vector<int> landmark_occupied_strata;
  std::vector<int> landmark_represented_strata;
  std::vector<int> landmark_grid_bins;
  std::vector<double> landmark_seconds;
  std::vector<double> coarse_partition_seconds;
  std::vector<double> landmark_sampling_seconds;
  std::vector<double> constraint_seconds;
  std::vector<double> landmark_prepare_seconds;
  std::vector<double> landmark_initialization_seconds;
  std::vector<double> landmark_graph_seconds;
  std::vector<double> core_evolution_seconds;
  std::vector<double> projection_seconds;
  std::vector<CoreRunDiagnostic> run_diagnostics;
  std::vector<CoreCycleDiagnostic> cycle_diagnostics;
  NeighborGraph knn;
  VisualizationInitResult visual_init;
  int runs = 0;
  int samples = 0;
  int cycles = 0;
  int res_constrain_rows = 0;
  int effective_landmarks = 0;
  int graph_builds = 0;
  int spatial_graph_builds = 0;
  int n_threads = 1;
  Backend backend = Backend::CPU;
  Backend graph_backend = Backend::CPU;
  Backend optimization_backend = Backend::CPU;
  Backend dissimilarity_backend = Backend::CPU;
  KNNIndexType graph_index_type = KNNIndexType::NativeHNSW;
  int graph_ivf_nlist = 0;
  int graph_ivf_nprobe = 0;
  int shared_landmark_partition_strata = 0;
  double graph_ivf_pilot_recall = 0.0;
  bool has_visual_init = false;
  bool knn_is_kodama_corrected = false;
  bool gpu_auto_workers = false;
  bool gpu_scheduler_enabled = false;
  bool shared_landmark_partition_used = false;
  int gpu_scheduler_lanes = 0;
  std::uint64_t kmeans_input_uploads = 0;
  std::uint64_t projection_sparse_uploads = 0;
  std::uint64_t projection_full_downloads = 0;
  std::uint64_t result_row_uploads = 0;
  std::uint64_t result_matrix_downloads = 0;
  int gpu_sm_count = 0;
  double gpu_free_memory_mb = 0.0;
  double gpu_total_memory_mb = 0.0;
  double gpu_worker_memory_estimate_mb = 0.0;
  double runtime_seconds = 0.0;
  double input_copy_seconds = 0.0;
  double visual_init_seconds = 0.0;
  double graph_feature_seconds = 0.0;
  double spatial_precompute_seconds = 0.0;
  double graph_seconds = 0.0;
  double shared_landmark_partition_seconds = 0.0;
  double spatial_graph_seconds = 0.0;
  double optimization_wall_seconds = 0.0;
  double optimization_sum_seconds = 0.0;
  double dissimilarity_seconds = 0.0;
  double peak_memory_mb = 0.0;
  std::uint64_t graph_storage_bytes = 0;
};

struct UMAPOptions {
  int n_components = 2;
  int n_epochs = 200;
  int n_neighbors = 30;
  int negative_sample_rate = 5;
  double learning_rate = 1.0;
  double min_dist = 0.01;
  double repulsion_strength = 1.0;
  int spectral_n_iter = 20;
  int n_threads = 1;
  int seed = 1234;
  int gpu_device = 0;
  UMAPGraphMode graph_mode = UMAPGraphMode::Fuzzy;
  std::vector<float> init;
  std::string init_source;
  Backend init_backend = Backend::Auto;
};

struct OpenTSNEOptions {
  int n_components = 2;
  int n_neighbors = 0;
  double perplexity = 30.0;
  double theta = 0.5;
  int early_exaggeration_iter = 250;
  int n_iter = 500;
  double early_exaggeration = 12.0;
  double exaggeration = 1.0;
  double learning_rate = 0.0;
  bool learning_rate_auto = true;
  double initial_momentum = 0.8;
  double final_momentum = 0.8;
  double min_gain = 0.01;
  double max_step_norm = 5.0;
  int n_threads = 1;
  int seed = 4;
  int gpu_device = 0;
  std::vector<float> init;
  std::string init_source;
  Backend init_backend = Backend::Auto;
};

struct EmbeddingResult {
  std::vector<float> embedding;
  int samples = 0;
  int components = 2;
  Backend backend = Backend::CPU;
  std::string initialization;
  Backend initialization_backend = Backend::Auto;
  std::string optimizer;
  std::size_t graph_edges = 0;
  float graph_max_weight = 0.0f;
  double runtime_seconds = 0.0;
};

struct PCAOptions {
  int n_components = 2;
  bool center = true;
  bool scale = false;
  int oversample = -1;
  int power_iterations = -1;
  int n_threads = 1;
  std::uint64_t seed = 4;
  int gpu_device = 0;
  Backend backend = Backend::CPU;
};

struct PCAResult {
  std::vector<float> scores;
  std::vector<float> loadings;
  std::vector<float> singular_values;
  std::vector<float> sdev;
  std::vector<float> variance;
  std::vector<float> variance_explained;
  std::vector<float> cumulative_variance_explained;
  std::vector<float> center;
  std::vector<float> scale;
  int samples = 0;
  int variables = 0;
  int components = 0;
  int oversample = 0;
  int power_iterations = 0;
  Backend backend = Backend::CPU;
  double total_variance = 0.0;
  double runtime_seconds = 0.0;
};

/** General float32 SIMPLS model for language bindings. */
struct PLSFitResult {
  std::vector<float> weights;
  std::vector<float> response_loadings;
  std::vector<float> scores;
  std::vector<float> coefficients;
  std::vector<float> fitted;
  std::vector<float> x_center;
  std::vector<float> x_scale;
  std::vector<float> y_center;
  int samples = 0;
  int predictors = 0;
  int responses = 0;
  int components = 0;
  Backend backend = Backend::CPU;
  double runtime_seconds = 0.0;
};

struct NormalizationOptions {
  NormalizationMethod method = NormalizationMethod::PQN;
  Backend backend = Backend::CPU;
  int n_threads = 1;
  int gpu_device = 0;
  std::vector<float> reference;
};

struct NormalizationResult {
  std::vector<float> train;
  std::vector<float> test;
  std::vector<float> train_coefficients;
  std::vector<float> test_coefficients;
  std::vector<float> reference;
  std::size_t train_rows = 0;
  std::size_t test_rows = 0;
  std::size_t variables = 0;
  NormalizationMethod method = NormalizationMethod::PQN;
  Backend backend = Backend::CPU;
  double runtime_seconds = 0.0;
};

struct ScalingOptions {
  ScalingMethod method = ScalingMethod::Autoscaling;
  Backend backend = Backend::CPU;
  int n_threads = 1;
  int gpu_device = 0;
};

struct ScalingResult {
  std::vector<float> train;
  std::vector<float> test;
  std::vector<float> center;
  std::vector<float> scale;
  std::size_t train_rows = 0;
  std::size_t test_rows = 0;
  std::size_t variables = 0;
  ScalingMethod method = ScalingMethod::Autoscaling;
  Backend backend = Backend::CPU;
  double runtime_seconds = 0.0;
};

struct PassingMessageOptions {
  int neighbors = 15;
  Backend backend = Backend::CPU;
  int n_threads = 1;
  int gpu_device = 0;
};

struct PassingMessageResult {
  std::vector<float> values;
  std::vector<float> sample_max_distances;
  std::size_t samples = 0;
  std::size_t variables = 0;
  std::size_t sample_groups = 0;
  int neighbors = 15;
  Backend backend = Backend::CPU;
  double graph_seconds = 0.0;
  double aggregation_seconds = 0.0;
  double runtime_seconds = 0.0;
};

/** Options for independent multi-slide spatial feature screening. */
struct SpatialFeatureOptions {
  int n_threads = 1;
  bool require_nonzero_each_sample = true;
};

/**
 * Multi-scale low-rank spatial projection statistics and combined significance.
 * Expression and basis arithmetic are float32; p-values use double precision
 * to retain useful tail resolution.
 */
struct SpatialFeatureResult {
  std::vector<float> score;
  std::vector<double> p_value;
  std::vector<double> adjusted_p_value;
  std::vector<float> per_sample_score;
  std::vector<double> per_sample_p_value;
  std::vector<int> ranking;
  std::vector<int> sample_labels;
  std::vector<int> basis_dimensions;
  std::size_t samples = 0;
  std::size_t variables = 0;
  std::size_t sample_groups = 0;
  Backend backend = Backend::CPU;
  double basis_seconds = 0.0;
  double statistic_seconds = 0.0;
  double runtime_seconds = 0.0;
};

KNNCVResult KNNCV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_METAL(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

PLSCVResult PLSDACV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSDACV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSDACV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_METAL(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_CPU(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_CUDA(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_METAL(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

CoreResult core_cpp(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_CPU(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_CPU(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_CUDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_CUDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_METAL(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_METAL(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNNGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNNGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNNGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const CoreOptions& options = CoreOptions()
);

CoreResult Core(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

KODAMAMatrixResult KODAMAMatrix_CPU(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix_CUDA(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix_METAL(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAGraphResult KODAMAGraph_CPU(
  MatrixView x,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph_CPU(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph_CUDA(
  MatrixView x,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph_CUDA(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph_METAL(
  MatrixView x,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph_METAL(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAGraphResult KODAMAGraph(
  MatrixView x,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

/** Apply the original KODAMA multi-sample separation to spatial coordinate 1. */
std::vector<float> KODAMASeparateSpatialSamples(
  const std::vector<float>& spatial,
  int rows,
  int columns,
  const std::vector<int>& samples
);

/** Materialize a handle-backed graph as one-based host arrays on demand. */
NeighborGraph KODAMAGraphMaterialize(const KODAMAGraphResult& graph);

KODAMAGraphResult KODAMAGraph(
  MatrixView x,
  MatrixView spatial,
  const KODAMAGraphOptions& options = KODAMAGraphOptions()
);

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const KODAMAGraphResult& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix(
  const KODAMAGraphResult& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

/**
 * Apply the KODAMA ensemble-agreement distance correction to a public,
 * one-based neighbor graph without allocating another graph.
 *
 * KODAMAMatrix(..., apply_kodama_dissimilarity=false) returns the retained
 * base graph. This function supports lazy correction of that graph when it is
 * actually needed for visualization.
 */
void KODAMADissimilarityInPlace(
  NeighborGraph& graph,
  const std::vector<int>& run_labels,
  int runs,
  int samples,
  Backend backend = Backend::CPU,
  int n_threads = 1,
  int gpu_device = 0
);

std::vector<float> KODAMAGraphFeatures_CPU(
  const NeighborGraph& graph,
  int samples,
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_CPU(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_CUDA(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_METAL(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_CPU(
  const NeighborGraph& graph,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_METAL(
  const NeighborGraph& graph,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_METAL(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_CPU(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_CPU(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_CPU(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_METAL(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_METAL(
  const NeighborGraph& graph,
  MatrixView raw_data,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

VisualizationInitResult KODAMAVisualizationPCAInit(
  MatrixView raw_data,
  const VisualizationInitOptions& options = VisualizationInitOptions()
);

PCAResult PCA(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_CPU(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_CUDA(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_METAL(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult RSVD(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PLSFitResult PLS(
  MatrixView x,
  MatrixView y,
  const PLSOptions& options = PLSOptions()
);

PLSFitResult PLS_CPU(
  MatrixView x,
  MatrixView y,
  const PLSOptions& options = PLSOptions()
);

PLSFitResult PLS_CUDA(
  MatrixView x,
  MatrixView y,
  const PLSOptions& options = PLSOptions()
);

PLSFitResult PLS_METAL(
  MatrixView x,
  MatrixView y,
  const PLSOptions& options = PLSOptions()
);

NormalizationResult Normalization(
  MatrixView train,
  MatrixView test = MatrixView(),
  const NormalizationOptions& options = NormalizationOptions()
);

NormalizationResult Normalization(
  MatrixView train,
  const NormalizationOptions& options
);

NormalizationResult Normalization_CPU(
  MatrixView train,
  MatrixView test = MatrixView(),
  const NormalizationOptions& options = NormalizationOptions()
);

NormalizationResult Normalization_CPU(
  MatrixView train,
  const NormalizationOptions& options
);

NormalizationResult Normalization_CUDA(
  MatrixView train,
  MatrixView test = MatrixView(),
  const NormalizationOptions& options = NormalizationOptions()
);

NormalizationResult Normalization_CUDA(
  MatrixView train,
  const NormalizationOptions& options
);

NormalizationResult Normalization_METAL(
  MatrixView train,
  MatrixView test = MatrixView(),
  const NormalizationOptions& options = NormalizationOptions()
);

NormalizationResult Normalization_METAL(
  MatrixView train,
  const NormalizationOptions& options
);

ScalingResult Scaling(
  MatrixView train,
  MatrixView test = MatrixView(),
  const ScalingOptions& options = ScalingOptions()
);

ScalingResult Scaling(
  MatrixView train,
  const ScalingOptions& options
);

ScalingResult Scaling_CPU(
  MatrixView train,
  MatrixView test = MatrixView(),
  const ScalingOptions& options = ScalingOptions()
);

ScalingResult Scaling_CPU(
  MatrixView train,
  const ScalingOptions& options
);

ScalingResult Scaling_CUDA(
  MatrixView train,
  MatrixView test = MatrixView(),
  const ScalingOptions& options = ScalingOptions()
);

ScalingResult Scaling_CUDA(
  MatrixView train,
  const ScalingOptions& options
);

ScalingResult Scaling_METAL(
  MatrixView train,
  MatrixView test = MatrixView(),
  const ScalingOptions& options = ScalingOptions()
);

ScalingResult Scaling_METAL(
  MatrixView train,
  const ScalingOptions& options
);

PassingMessageResult PassingMessage(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const PassingMessageOptions& options = PassingMessageOptions()
);

PassingMessageResult PassingMessage_CPU(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const PassingMessageOptions& options = PassingMessageOptions()
);

PassingMessageResult PassingMessage_CUDA(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const PassingMessageOptions& options = PassingMessageOptions()
);

PassingMessageResult PassingMessage_METAL(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const PassingMessageOptions& options = PassingMessageOptions()
);

SpatialFeatureResult SpatialFeatureSelection(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const SpatialFeatureOptions& options = SpatialFeatureOptions()
);

SpatialFeatureResult SpatialFeatureSelection_CPU(
  MatrixView data,
  MatrixView spatial,
  const std::vector<int>& samples = {},
  const SpatialFeatureOptions& options = SpatialFeatureOptions()
);

NeighborGraph KODAMAKNNGraph_CPU(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph_CUDA(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph_METAL(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

ResidentIVFIndex BuildResidentIVFIndex(
  MatrixView train,
  const KNNOptions& options = KNNOptions()
);

/**
 * Search an existing accelerator-resident IVF index.
 *
 * Query rows are uploaded for this call. Returned neighbor identifiers are
 * one-based, matching NeighborGraph throughout the public API.
 */
NeighborGraph SearchResidentIVFIndex(
  const ResidentIVFIndex& index,
  MatrixView query,
  int k,
  ResidentIVFSearchStats* stats = nullptr
);

/**
 * Search the resident training matrix against itself without uploading it
 * again. Self-neighbors are excluded when exclude_self is true.
 */
NeighborGraph SearchResidentIVFIndexSelf(
  const ResidentIVFIndex& index,
  int k,
  bool exclude_self = true,
  ResidentIVFSearchStats* stats = nullptr
);

GraphClusterResult KODAMAGraphCluster_CPU(
  const NeighborGraph& graph,
  int samples,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAGraphCluster(
  const NeighborGraph& graph,
  int samples,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAEmbeddingGraphCluster(
  MatrixView embedding,
  const NeighborGraph& graph,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAEmbeddingCluster(
  MatrixView embedding,
  const GraphClusterOptions& options = GraphClusterOptions()
);

const char* to_string(Backend backend);
const char* to_string(DistanceMetric metric);
const char* to_string(KNNIndexType index_type);
const char* to_string(PLSMode mode);
const char* to_string(CoreClassifier classifier);
const char* to_string(GraphWeightType weight_type);
const char* to_string(GraphFeatureMode mode);
bool MetalAvailable();

}  // namespace kodama
