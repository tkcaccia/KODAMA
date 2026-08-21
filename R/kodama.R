# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

as_kodama_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

kodama_make_visual_init <- function(data,
                                    seed = 4L,
                                    backend = "cpu",
                                    n.cores = 1L,
                                    gpu.device = 0L) {
  out <- kodama_visual_init_cpp(
    as_kodama_matrix(data),
    backend = backend,
    seed = as.integer(seed),
    n_threads = as.integer(n.cores),
    gpu_device = as.integer(gpu.device)
  )
  attr(out$opentsne, "visual_init") <- "raw_pca"
  attr(out$umap, "visual_init") <- "raw_pca"
  out
}

kodama_visual_init <- function(x,
                               method = c("opentsne", "umap"),
                               backend = NULL) {
  method <- match.arg(method)
  if (!is.list(x) || is.null(x$visual_init)) return(NULL)
  init <- x$visual_init
  if (!is.null(backend) && is.list(init) && !is.null(init$backend) &&
      !identical(as.character(init$backend), as.character(backend))) {
    return(NULL)
  }
  if (is.matrix(init)) return(init)
  if (is.list(init) && !is.null(init[[method]])) return(init[[method]])
  NULL
}

as_kodama_labels <- function(x) {
  as.integer(as.factor(x))
}

as_kodama_samples <- function(samples, n) {
  if (is.null(samples)) return(NULL)
  if (length(samples) != n) stop("samples must have one value per data row.")
  sample_names <- names(table(samples))
  encoded <- match(as.character(samples), sample_names)
  if (anyNA(encoded)) stop("samples must not contain missing values.")
  as.integer(encoded)
}

extract_kodama_graph <- function(x) {
  if (is.list(x) && !is.null(x$knn)) return(x$knn)
  if (is.list(x) && inherits(x$handle, "kodama_graph_handle")) return(x)
  if (is.list(x) && !is.null(x$indices) && !is.null(x$distances)) return(x)
  NULL
}

kodama_graph_is_handle <- function(x) {
  is.list(x) && inherits(x$handle, "kodama_graph_handle")
}

kodama_graph_samples <- function(x) {
  if (kodama_graph_is_handle(x)) return(as.integer(x$samples))
  nrow(x$indices)
}

kodama_graph_neighbors <- function(x) {
  if (kodama_graph_is_handle(x)) return(as.integer(x$neighbors))
  ncol(x$indices)
}

kodama_graph_output_mode <- function(x) {
  if (identical(x, "handle")) return(2L)
  if (isTRUE(x)) return(1L)
  if (identical(x, FALSE)) return(0L)
  stop("return.graph must be FALSE, TRUE, or 'handle'.")
}

kodama_class_counts <- function(res) {
  if (is.null(res)) return(integer())
  apply(res, 1L, function(z) length(unique(z)))
}

kodama_best_run <- function(acc) {
  if (length(acc) == 0L || all(is.na(acc))) return(NA_integer_)
  as.integer(which.max(acc))
}

kodama_begin_progress <- function(progress, progress.file = NULL) {
  if (!isTRUE(progress)) return(list(path = NULL, previous = NULL))
  path <- progress.file
  if (is.null(path)) {
    path <- tempfile("kodama-progress-", fileext = ".log")
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  path <- file.path(normalizePath(dirname(path), mustWork = TRUE), basename(path))
  previous <- Sys.getenv("KODAMA_PROGRESS_FILE", unset = NA_character_)
  Sys.setenv(KODAMA_PROGRESS_FILE = path)
  message("KODAMA progress file: ", path)
  list(path = path, previous = previous)
}

kodama_end_progress <- function(state) {
  if (is.null(state$path)) return(invisible(NULL))
  if (is.na(state$previous)) {
    Sys.unsetenv("KODAMA_PROGRESS_FILE")
  } else {
    Sys.setenv(KODAMA_PROGRESS_FILE = state$previous)
  }
  invisible(NULL)
}

as_kodama_matrix_result <- function(result, parameters, visual_init = NULL) {
  counts <- kodama_class_counts(result$res)
  best_run <- kodama_best_run(result$acc)
  result$parameters <- parameters
  if (!is.null(visual_init)) result$visual_init <- visual_init
  if (is.list(result$visual_init)) {
    if (!is.null(result$visual_init$opentsne)) {
      attr(result$visual_init$opentsne, "visual_init") <- "raw_pca"
    }
    if (!is.null(result$visual_init$umap)) {
      attr(result$visual_init$umap, "visual_init") <- "raw_pca"
    }
  }
  result$class_counts <- counts
  result$best_run <- best_run
  result$best_labels <- if (!is.na(best_run)) as.integer(result$res[best_run, ]) else integer()
  class(result) <- unique(c("kodama_matrix", class(result)))
  result
}

#' Cross-validated KNN classification
#'
#' @param data Numeric matrix with samples in rows and variables in columns.
#' @param labels Class labels used as the cross-validation truth.
#' @param constrain Optional integer vector assigning samples to indivisible
#'   fold groups.
#' @param folds Number of cross-validation folds.
#' @param stratified Whether to stratify folds by class labels.
#' @param seed Integer random seed for fold construction.
#' @param k Number of neighbors used by the KNN classifier.
#' @param metric Distance or similarity metric.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @return A cross-validation result containing predictions, truth, fold
#'   assignments, fold and global accuracy, confusion matrix, timing, and
#'   backend diagnostics.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris$Species)
#' fit <- KNNCV(x, y, folds = 3, k = 5, backend = "cpu")
#' fit$accuracy
#' @export
KNNCV <- function(data,
                  labels,
                  constrain = NULL,
                  folds = 10L,
                  stratified = TRUE,
                  seed = 1L,
                  k = 10L,
                  metric = c("cosine", "inner_product", "euclidean"),
                  backend = NULL,
                  n.cores = 1L,
                  gpu.device = 0L) {
  metric <- match.arg(metric)
  backend <- kodama_resolve_backend(backend)
  knncv_cpp(
    as_kodama_matrix(data),
    as_kodama_labels(labels),
    if (is.null(constrain)) NULL else as.integer(constrain),
    as.integer(folds),
    isTRUE(stratified),
    as.integer(seed),
    as.integer(k),
    metric,
    backend,
    as.integer(n.cores),
    as.integer(gpu.device)
  )
}

#' Cross-validated SIMPLS + PLS-LDA classification
#'
#' @param data Numeric matrix with samples in rows and variables in columns.
#' @param labels Class labels used as the cross-validation truth.
#' @param constrain Optional integer vector assigning samples to indivisible
#'   fold groups.
#' @param folds Number of cross-validation folds.
#' @param stratified Whether to stratify folds by class labels.
#' @param seed Integer random seed for fold construction.
#' @param ncomp Number of SIMPLS latent components requested.
#' @param center Whether the C++ core centers the analysis matrix.
#' @param scale Whether the C++ core scales the analysis matrix.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @return A cross-validation result containing predictions, truth, fold
#'   assignments, fold and global accuracy, confusion matrix, timing, and
#'   backend diagnostics.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris$Species)
#' fit <- PLSLDACV(x, y, folds = 3, ncomp = 2, backend = "cpu")
#' fit$accuracy
#' @export
PLSLDACV <- function(data,
                     labels,
                     constrain = NULL,
                     folds = 10L,
                     stratified = TRUE,
                     seed = 1L,
                     ncomp = min(50L, ncol(data)),
                     center = TRUE,
                     scale = TRUE,
                     backend = NULL,
                     n.cores = 1L,
                     gpu.device = 0L) {
  backend <- kodama_resolve_backend(backend)
  plsldacv_cpp(
    as_kodama_matrix(data),
    as_kodama_labels(labels),
    if (is.null(constrain)) NULL else as.integer(constrain),
    as.integer(folds),
    isTRUE(stratified),
    as.integer(seed),
    as.integer(ncomp),
    isTRUE(center),
    isTRUE(scale),
    backend,
    as.integer(n.cores),
    as.integer(gpu.device)
  )
}

#' KODAMA core optimization with KNN classifier
#'
#' @param data Numeric matrix with samples in rows and variables in columns.
#' @param labels Initial labels for the label-evolution process.
#' @param constrain Optional integer vector assigning samples to indivisible
#'   proposal and fold groups.
#' @param fix Optional integer vector marking samples whose labels are fixed.
#' @param cycles Number of proposal/evaluation cycles.
#' @param folds Number of cross-validation folds.
#' @param stratified Whether to stratify folds by class labels.
#' @param seed Integer random seed.
#' @param k Number of neighbors used by the KNN classifier.
#' @param metric Distance or similarity metric.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @return A list containing the best labels and predictions, accuracy and score
#'   traces, plus explicit proposal, acceptance, rejection, coarsening, and
#'   absorption counters for optimizer-state diagnostics.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' y <- rep(1:3, each = 50)
#' fit <- CoreKNN(x, y, cycles = 1, folds = 3, k = 5, n.cores = 1)
#' fit$accuracy
#' @export
CoreKNN <- function(data,
                    labels,
                    constrain = NULL,
                    fix = NULL,
                    cycles = 100L,
                    folds = 10L,
                    stratified = TRUE,
                    seed = 1L,
                    k = 30L,
                    metric = c("euclidean", "cosine", "inner_product"),
                    backend = NULL,
                    n.cores = 4L,
                    gpu.device = 0L) {
  metric <- match.arg(metric)
  backend <- kodama_resolve_backend(backend)
  core_knn_cpp(
    as_kodama_matrix(data),
    as_kodama_labels(labels),
    if (is.null(constrain)) NULL else as.integer(constrain),
    if (is.null(fix)) NULL else as.integer(fix),
    as.integer(cycles),
    as.integer(folds),
    isTRUE(stratified),
    as.integer(seed),
    as.integer(k),
    metric,
    backend,
    as.integer(n.cores),
    as.integer(gpu.device)
  )
}

#' KODAMA core optimization with PLS-LDA classifier
#'
#' @param data Numeric matrix with samples in rows and variables in columns.
#' @param labels Initial labels for the label-evolution process.
#' @param constrain Optional integer vector assigning samples to indivisible
#'   proposal and fold groups.
#' @param fix Optional integer vector marking samples whose labels are fixed.
#' @param cycles Number of proposal/evaluation cycles.
#' @param folds Number of cross-validation folds.
#' @param stratified Whether to stratify folds by class labels.
#' @param seed Integer random seed.
#' @param ncomp Number of SIMPLS latent components requested.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @return A list containing the best labels and predictions, accuracy and score
#'   traces, plus explicit proposal, acceptance, rejection, coarsening, and
#'   absorption counters for optimizer-state diagnostics.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' y <- rep(1:3, each = 50)
#' fit <- CorePLSLDA(x, y, cycles = 1, folds = 3, ncomp = 2, n.cores = 1)
#' fit$accuracy
#' @export
CorePLSLDA <- function(data,
                       labels,
                       constrain = NULL,
                       fix = NULL,
                       cycles = 100L,
                       folds = 10L,
                       stratified = TRUE,
                       seed = 1L,
                       ncomp = min(50L, ncol(data)),
                       backend = NULL,
                       n.cores = 4L,
                       gpu.device = 0L) {
  backend <- kodama_resolve_backend(backend)
  core_plslda_cpp(
    as_kodama_matrix(data),
    as_kodama_labels(labels),
    if (is.null(constrain)) NULL else as.integer(constrain),
    if (is.null(fix)) NULL else as.integer(fix),
    as.integer(cycles),
    as.integer(folds),
    isTRUE(stratified),
    as.integer(seed),
    as.integer(ncomp),
    backend,
    as.integer(n.cores),
    as.integer(gpu.device)
  )
}

#' Run KODAMA matrix optimization
#'
#' @param data Optional raw numeric matrix. It is required when `graph` is not
#'   supplied. The C++ core stores analysis values as float32.
#' @param graph Optional `KODAMA.graph` result or a bare list with `indices`
#'   and `distances`.
#' @param spatial Optional numeric matrix of spatial coordinates.
#' @param samples Optional sample or slide identifier. With more than one
#'   unique value, spatial coordinate 1 is separated exactly as in the original
#'   KODAMA implementation before spatial graph and constraint construction.
#' @param W Optional integer vector.
#' @param constrain Optional integer vector of sample constraints.
#' @param fix Optional integer vector marking fixed samples.
#' @param M Number of independent KODAMA runs.
#' @param Tcycle Number of optimization cycles per run.
#' @param ncomp Number of PLS components for the PLS-LDA classifier.
#' @param landmarks Maximum number of samples optimized directly in each run.
#' @param splitting Initial number of label classes used for each run.
#' @param n.cores CPU worker count. `NULL` uses `options(n.cores = ...)`, then
#'   `N_CORES`, and otherwise 4. For CUDA or Metal matrix optimization, an
#'   explicit `0` enables backend-specific automatic independent-run lanes.
#' @param graph.neighbors Number of neighbors retained in the returned graph.
#' @param knn.k Number of neighbors used by the KNN classifier.
#' @param spatial.resolution Resolution parameter for constrained grouping when
#'   optional coordinate constraints are supplied.
#' @param spatial.graph.mix Logical flag passed to the C++ constraint builder.
#' @param spatial.constraint.mode Constraint construction mode.
#' @param spatial.mode Coordinate treatment. `"population"` applies the
#'   classic repeated-coordinate regularization independently in every `M`
#'   run before spatial clustering. The default `"standard"` is unchanged.
#' @param metric Distance or similarity metric.
#' @param classifier Either `"knn"` or `"pls_lda"`.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param seed Integer random seed.
#' @param visual.init Whether the same native matrix call performs one float32
#'   PCA and stores both UMAP and openTSNE initializations for reuse by
#'   `KODAMA.visualization`.
#' @param progress Whether the C++ core prints run/cycle progress.
#' @param progress.file Optional progress-log path. When `progress=TRUE`, the
#'   native workers append M/Tcycle checkpoints here. A temporary path is
#'   created and announced when this is `NULL`.
#' @param folds Number of cross-validation folds used by the classifier.
#' @param ... Reserved internal controls for reproducibility experiments.
#' @param apply.kodama.dissimilarity Whether to return the KODAMA-corrected
#'   neighbor graph rather than only the base graph.
#' @param return.graph `FALSE` omits the result graph, `TRUE` materializes R
#'   index and distance matrices, and `"handle"` retains one float32 graph in
#'   an external C++ pointer for reuse without two large R matrices.
#' @return A list returned by the kodama-cpp core. The full-data graph is built
#'   once before all `M` runs. The result contains one `knn` graph;
#'   `knn_is_kodama_corrected`, `graph_storage_bytes`, `graph_builds`, and
#'   `timing` expose its lifecycle. Per-run landmark-selection times are
#'   retained in `landmark_seconds`; their sum, mean, and median are also
#'   reported in `timing` so landmark construction is not conflated with the
#'   classifier core.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' fit <- KODAMA.matrix(
#'   data = x, classifier = "knn", M = 1, Tcycle = 1,
#'   landmarks = 100, splitting = 10, knn.k = 5, n.cores = 1,
#'   return.graph = FALSE
#' )
#' fit$acc
#' @aliases KODAMA.matrix
#' @export
kodama_matrix <- function(data = NULL,
                          graph = NULL,
                          spatial = NULL,
                          samples = NULL,
                          W = NULL,
                          constrain = NULL,
                          fix = NULL,
                          M = 100L,
                          Tcycle = 20L,
                          ncomp = NULL,
                          landmarks = 10000L,
                          splitting = NULL,
                          n.cores = NULL,
                          graph.neighbors = NULL,
                          knn.k = 30L,
                          spatial.resolution = 0.4,
                          spatial.graph.mix = FALSE,
                          spatial.constraint.mode = c("kmeans", "graph", "auto"),
                          spatial.mode = c("standard", "population"),
                          metric = "euclidean",
                          classifier = c("knn", "pls_lda"),
                          backend = NULL,
                          seed = 1234L,
                          folds = 5L,
                          visual.init = TRUE,
                          progress = TRUE,
                          progress.file = NULL,
                          apply.kodama.dissimilarity = TRUE,
                          return.graph = FALSE,
                          ...) {
  experimental <- list(...)
  unknown_experimental <- setdiff(names(experimental), ".evolution.policy")
  if (length(unknown_experimental)) {
    stop("Unknown reserved KODAMA.matrix control: ", unknown_experimental[[1L]])
  }
  .evolution.policy <- if (is.null(experimental$.evolution.policy)) {
    "full"
  } else {
    experimental$.evolution.policy
  }
  classifier <- match.arg(classifier)
  backend <- kodama_resolve_backend(backend)
  n.cores <- kodama_resolve_n_cores(n.cores, default = 4L, allow.zero = TRUE)
  spatial.constraint.mode <- match.arg(spatial.constraint.mode)
  spatial.mode <- match.arg(spatial.mode)
  .evolution.policy <- match.arg(.evolution.policy, c(
    "full", "no_prediction_guidance", "fixed_proposal_budget",
    "no_transition_proposal", "greedy_acceptance", "raw_cv_score",
    "no_pls_transition_coarsening", "no_pls_fragmentation_penalty"
  ))
  graph_output <- kodama_graph_output_mode(return.graph)
  if (!is.null(graph)) {
    graph_input <- extract_kodama_graph(graph)
    if (is.null(graph_input)) {
      stop("graph must be a KODAMA.graph result or a list with indices and distances.")
    }
    if (!is.null(data) && !is.null(extract_kodama_graph(data))) {
      stop("data must be the raw numeric matrix; pass graph inputs through graph.")
    }
    raw_data <- if (is.null(data)) NULL else as_kodama_matrix(data)
    n_samples <- kodama_graph_samples(graph_input)
    if (!is.null(raw_data) && nrow(raw_data) != n_samples) {
      stop("data and graph must contain the same number of samples.")
    }
    if (is.null(ncomp)) {
      ncomp <- if (is.null(raw_data)) 50L else min(50L, ncol(raw_data))
    }
    if (is.null(splitting)) splitting <- ifelse(n_samples < 40000, 100L, 300L)
    if (is.null(graph.neighbors)) graph.neighbors <- kodama_graph_neighbors(graph_input)
    return(kodama_matrix_graph(
      indices = graph,
      data = raw_data,
      spatial = spatial,
      samples = samples,
      W = W,
      constrain = constrain,
      fix = fix,
      M = M,
      Tcycle = Tcycle,
      ncomp = ncomp,
      landmarks = landmarks,
      splitting = splitting,
      n.cores = n.cores,
      graph.neighbors = graph.neighbors,
      knn.k = knn.k,
      spatial.resolution = spatial.resolution,
      spatial.graph.mix = spatial.graph.mix,
      spatial.constraint.mode = spatial.constraint.mode,
      spatial.mode = spatial.mode,
      classifier = classifier,
      backend = backend,
      seed = seed,
      folds = folds,
      visual.init = visual.init,
      progress = progress,
      progress.file = progress.file,
      apply.kodama.dissimilarity = apply.kodama.dissimilarity,
      return.graph = return.graph,
      .evolution.policy = .evolution.policy
    ))
  }

  if (is.null(data)) {
    stop("data or graph is required.")
  }
  if (!is.null(extract_kodama_graph(data))) {
    stop("data must be the raw numeric matrix; pass graph inputs through graph.")
  }
  data_matrix <- as_kodama_matrix(data)
  sample_ids <- as_kodama_samples(samples, nrow(data_matrix))
  if (is.null(ncomp)) ncomp <- min(50L, ncol(data_matrix))
  if (is.null(splitting)) {
    splitting <- ifelse(nrow(data_matrix) < 40000, 100L, 300L)
  }
  if (is.null(graph.neighbors)) graph.neighbors <- 100L
  parameters <- list(
    M = as.integer(M),
    Tcycle = as.integer(Tcycle),
    ncomp = as.integer(ncomp),
    landmarks = as.integer(landmarks),
    splitting = as.integer(splitting),
    n.cores = as.integer(n.cores),
    graph.neighbors = as.integer(graph.neighbors),
    knn.k = as.integer(knn.k),
    spatial.resolution = as.numeric(spatial.resolution),
    samples = if (is.null(sample_ids)) 0L else length(unique(sample_ids)),
    spatial.graph.mix = isTRUE(spatial.graph.mix),
    spatial.constraint.mode = spatial.constraint.mode,
    spatial.mode = spatial.mode,
    metric = metric,
    classifier = classifier,
    backend = backend,
    seed = as.integer(seed),
    folds = as.integer(folds),
    evolution.policy = .evolution.policy,
    visual.init = isTRUE(visual.init),
    progress.file = progress.file,
    apply.kodama.dissimilarity = isTRUE(apply.kodama.dissimilarity),
    return.graph = if (graph_output == 2L) "handle" else graph_output == 1L
  )
  progress_state <- kodama_begin_progress(progress, progress.file)
  on.exit(kodama_end_progress(progress_state), add = TRUE)
  result <- kodama_matrix_cpp(
    data = data_matrix,
    spatial = if (is.null(spatial)) NULL else as_kodama_matrix(spatial),
    samples = sample_ids,
    W = if (is.null(W)) NULL else as.integer(W),
    constrain = if (is.null(constrain)) NULL else as.integer(constrain),
    fix = if (is.null(fix)) NULL else as.integer(fix),
    M = as.integer(M),
    Tcycle = as.integer(Tcycle),
    ncomp = as.integer(ncomp),
    landmarks = as.integer(landmarks),
    splitting = as.integer(splitting),
    n_cores = as.integer(n.cores),
    graph_neighbors = as.integer(graph.neighbors),
    knn_k = as.integer(knn.k),
    spatial_resolution = as.numeric(spatial.resolution),
    spatial_graph_mix = isTRUE(spatial.graph.mix),
    spatial_constraint_mode = if (spatial.constraint.mode == "auto") -1L else if (spatial.constraint.mode == "graph") 1L else 0L,
    spatial_coordinate_mode = if (spatial.mode == "population") 1L else 0L,
    metric = metric,
    classifier = classifier,
    backend = backend,
    seed = as.integer(seed),
    progress = isTRUE(progress),
    apply_kodama_dissimilarity = isTRUE(apply.kodama.dissimilarity),
    compute_visual_init = isTRUE(visual.init),
    graph_output = graph_output,
    folds = as.integer(folds),
    evolution_policy = .evolution.policy
  )
  result <- as_kodama_matrix_result(result, parameters)
  result$progress_file <- progress_state$path
  result
}

#' @export
KODAMA.matrix <- kodama_matrix

#' KODAMA matrix from a precomputed KNN graph
#'
#' @param indices Integer matrix of neighbor indices, or a list with `indices`
#'   and `distances`.
#' @param distances Numeric matrix of neighbor distances.
#' @param data Optional original data matrix. When supplied, KODAMA uses this
#'   float32 geometry for landmark selection and initial splitting, while
#'   reusing the supplied graph for KNN classification/projection and the final
#'   KODAMA graph. When omitted, a self-tuning graph Laplacian geometry is used.
#' @param spatial Optional spatial or external coordinate matrix used only for
#'   constrained grouping when supplied.
#' @param samples Optional sample or slide identifier used to separate spatial
#'   coordinate 1 before spatial graph and constraint construction.
#' @param W Optional starting labels.
#' @param constrain Optional group vector.
#' @param fix Optional fixed-label mask.
#' @param M Number of independent runs.
#' @param Tcycle Number of optimization cycles per run.
#' @param ncomp Number of graph features/components used by the PLS-LDA path.
#' @param landmarks Maximum landmarks optimized directly.
#' @param splitting Initial number of classes.
#' @param n.cores CPU workers. For CUDA or Metal, `0` enables backend-specific
#'   automatic independent-run lane selection.
#' @param graph.neighbors Number of neighbors retained in the returned graph.
#' @param knn.k Number of neighbors used by the KNN classifier.
#' @param spatial.resolution Resolution used when spatial constraints are
#'   derived inside the core.
#' @param spatial.graph.mix Whether to combine spatial and feature graphs for
#'   grouping.
#' @param spatial.constraint.mode Spatial constraint strategy.
#' @param spatial.mode Coordinate treatment. `"population"` applies the
#'   classic repeated-coordinate regularization independently in every `M`
#'   run before spatial clustering.
#' @param classifier Either `"knn"` or `"pls_lda"`.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param graph.feature.mode Graph-to-feature transform for PLS-LDA and
#'   graph-only initialization. The standard path is
#'   `"laplacian_self_tuning"`.
#' @param graph.feature.components Feature count; `0` uses `ncomp`.
#' @param graph.feature.steps Power iterations used by graph feature extraction.
#' @param seed Integer random seed.
#' @param folds Number of cross-validation folds used by the classifier.
#' @param visual.init Whether to propagate the PCA starts stored in a
#'   `KODAMA.graph` object.
#' @param progress Whether the C++ core records progress.
#' @param progress.file Optional progress-log path. When `progress=TRUE`, the
#'   native workers append M/Tcycle checkpoints here. A temporary path is
#'   created and announced when this is `NULL`.
#' @param apply.kodama.dissimilarity Whether to return the KODAMA-corrected
#'   graph.
#' @param return.graph `FALSE` omits the graph, `TRUE` materializes matrices,
#'   and `"handle"` returns a reusable external C++ graph pointer.
#' @param ... Reserved internal controls for reproducibility experiments.
#' @return A `kodama_matrix` result with evolved labels, accuracy traces,
#'   diagnostics, timing, and the corrected graph or graph handle when
#'   requested.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' g <- KODAMA.graph(x, k = 10, storage = "matrix", n.cores = 1)
#' fit <- KODAMA.matrix.graph(
#'   g$indices, g$distances, data = x, M = 1, Tcycle = 1,
#'   landmarks = 100, splitting = 10, knn.k = 5, n.cores = 1,
#'   return.graph = FALSE
#' )
#' fit$acc
#' @aliases KODAMA.matrix.graph
#' @export
kodama_matrix_graph <- function(indices,
                                distances = NULL,
                                data = NULL,
                                spatial = NULL,
                                samples = NULL,
                                W = NULL,
                                constrain = NULL,
                                fix = NULL,
                                M = 100L,
                                Tcycle = 20L,
                                ncomp = NULL,
                                landmarks = 10000L,
                                splitting = NULL,
                                n.cores = 4L,
                                graph.neighbors = NULL,
                                knn.k = 30L,
                                spatial.resolution = 0.4,
                                spatial.graph.mix = FALSE,
                                spatial.constraint.mode = c("kmeans", "graph", "auto"),
                                spatial.mode = c("standard", "population"),
                                classifier = c("knn", "pls_lda"),
                                backend = NULL,
                                graph.feature.mode = "laplacian_self_tuning",
                                graph.feature.components = 0L,
                                graph.feature.steps = 3L,
                                seed = 1234L,
                                folds = 5L,
                                visual.init = TRUE,
                                progress = TRUE,
                                progress.file = NULL,
                                apply.kodama.dissimilarity = TRUE,
                                return.graph = FALSE,
                                ...) {
  experimental <- list(...)
  unknown_experimental <- setdiff(names(experimental), ".evolution.policy")
  if (length(unknown_experimental)) {
    stop("Unknown reserved KODAMA.matrix control: ", unknown_experimental[[1L]])
  }
  .evolution.policy <- if (is.null(experimental$.evolution.policy)) {
    "full"
  } else {
    experimental$.evolution.policy
  }
  graph_object <- if (is.list(indices)) indices else NULL
  supplied_graph <- extract_kodama_graph(indices)
  graph_handle <- NULL
  if (!is.null(supplied_graph)) {
    if (kodama_graph_is_handle(supplied_graph)) {
      graph_handle <- supplied_graph$handle
    } else {
      distances <- supplied_graph$distances
      indices <- supplied_graph$indices
    }
  }
  if (is.null(graph_handle) && (is.null(indices) || is.null(distances))) {
    stop("indices and distances, or a KODAMA graph handle, are required.")
  }
  classifier <- match.arg(classifier)
  backend <- kodama_resolve_backend(backend)
  graph_output <- kodama_graph_output_mode(return.graph)
  if (is.null(ncomp)) ncomp <- if (is.null(data)) 50L else min(50L, ncol(data))
  n_samples <- if (is.null(graph_handle)) nrow(indices) else kodama_graph_samples(supplied_graph)
  neighbors <- if (is.null(graph_handle)) ncol(indices) else kodama_graph_neighbors(supplied_graph)
  if (is.null(splitting)) {
    splitting <- ifelse(n_samples < 40000, 100L, 300L)
  }
  sample_ids <- as_kodama_samples(samples, n_samples)
  if (is.null(graph.neighbors)) graph.neighbors <- neighbors
  spatial.constraint.mode <- match.arg(spatial.constraint.mode)
  spatial.mode <- match.arg(spatial.mode)
  graph.feature.mode <- match.arg(graph.feature.mode)
  .evolution.policy <- match.arg(.evolution.policy, c(
    "full", "no_prediction_guidance", "fixed_proposal_budget",
    "no_transition_proposal", "greedy_acceptance", "raw_cv_score",
    "no_pls_transition_coarsening", "no_pls_fragmentation_penalty"
  ))
  visual_init <- NULL
  if (isTRUE(visual.init) && !is.null(graph_object$visual_init)) {
    if (identical(as.character(graph_object$visual_init$backend), backend)) {
      visual_init <- graph_object$visual_init
    } else if (!is.null(data)) {
      visual_init <- kodama_make_visual_init(
        data,
        seed = seed,
        backend = backend,
        n.cores = n.cores
      )
    }
  }
  parameters <- list(
    M = as.integer(M),
    Tcycle = as.integer(Tcycle),
    ncomp = as.integer(ncomp),
    landmarks = as.integer(landmarks),
    splitting = as.integer(splitting),
    n.cores = as.integer(n.cores),
    graph.neighbors = as.integer(graph.neighbors),
    knn.k = as.integer(knn.k),
    spatial.resolution = as.numeric(spatial.resolution),
    samples = if (is.null(sample_ids)) 0L else length(unique(sample_ids)),
    spatial.graph.mix = isTRUE(spatial.graph.mix),
    spatial.constraint.mode = spatial.constraint.mode,
    spatial.mode = spatial.mode,
    classifier = classifier,
    backend = backend,
    graph.feature.mode = graph.feature.mode,
    graph.feature.components = as.integer(graph.feature.components),
    graph.feature.steps = as.integer(graph.feature.steps),
    graph.uses.data.geometry = !is.null(data),
    seed = as.integer(seed),
    folds = as.integer(folds),
    evolution.policy = .evolution.policy,
    visual.init = isTRUE(visual.init),
    progress.file = progress.file,
    apply.kodama.dissimilarity = isTRUE(apply.kodama.dissimilarity),
    return.graph = if (graph_output == 2L) "handle" else graph_output == 1L
  )
  common_args <- list(
    data = if (is.null(data)) NULL else as_kodama_matrix(data),
    spatial = if (is.null(spatial)) NULL else as_kodama_matrix(spatial),
    samples = sample_ids,
    W = if (is.null(W)) NULL else as.integer(W),
    constrain = if (is.null(constrain)) NULL else as.integer(constrain),
    fix = if (is.null(fix)) NULL else as.integer(fix),
    M = as.integer(M),
    Tcycle = as.integer(Tcycle),
    ncomp = as.integer(ncomp),
    landmarks = as.integer(landmarks),
    splitting = as.integer(splitting),
    n_cores = as.integer(n.cores),
    graph_neighbors = as.integer(graph.neighbors),
    knn_k = as.integer(knn.k),
    spatial_resolution = as.numeric(spatial.resolution),
    spatial_graph_mix = isTRUE(spatial.graph.mix),
    spatial_constraint_mode = if (spatial.constraint.mode == "auto") -1L else if (spatial.constraint.mode == "graph") 1L else 0L,
    spatial_coordinate_mode = if (spatial.mode == "population") 1L else 0L,
    classifier = classifier,
    backend = backend,
    graph_feature_mode = graph.feature.mode,
    graph_feature_components = as.integer(graph.feature.components),
    graph_feature_steps = as.integer(graph.feature.steps),
    seed = as.integer(seed),
    progress = isTRUE(progress),
    apply_kodama_dissimilarity = isTRUE(apply.kodama.dissimilarity),
    graph_output = graph_output,
    folds = as.integer(folds),
    evolution_policy = .evolution.policy
  )
  progress_state <- kodama_begin_progress(progress, progress.file)
  on.exit(kodama_end_progress(progress_state), add = TRUE)
  result <- if (is.null(graph_handle)) {
    do.call(kodama_matrix_graph_cpp, c(list(
      indices = as.matrix(indices),
      distances = as.matrix(distances),
      spatial_indices = if (is.null(graph_object$spatial_knn)) NULL else
        as.matrix(graph_object$spatial_knn$indices),
      spatial_distances = if (is.null(graph_object$spatial_knn)) NULL else
        as.matrix(graph_object$spatial_knn$distances),
      spatial_jitter = if (is.null(graph_object$spatial_jitter)) NULL else
        as.numeric(graph_object$spatial_jitter),
      prepared_spatial_dimensions = if (is.null(graph_object$spatial_dimensions)) 0L else
        as.integer(graph_object$spatial_dimensions)
    ), common_args))
  } else {
    do.call(kodama_matrix_graph_handle_cpp, c(list(
      graph_handle = graph_handle
    ), common_args))
  }
  result <- as_kodama_matrix_result(result, parameters, visual_init = visual_init)
  result$progress_file <- progress_state$path
  result
}

#' @export
KODAMA.matrix.graph <- kodama_matrix_graph

#' Return a compact timing table for a KODAMA result
#'
#' @param x A `kodama_matrix` result or compatible list with timing fields.
#'   Matrix results include separate landmark-selection aggregates and
#'   classifier optimization timings.
#' @return A data frame with analysis steps, elapsed seconds, and percentage
#'   of the recorded total runtime.
#' @examples
#' KODAMA.timing(list(timing = list(graph = 0.5, runtime_seconds = 2)))
#' @aliases kodama_timing
#' @export
KODAMA.timing <- function(x) {
  timing <- if (is.list(x) && !is.null(x$timing)) x$timing else NULL
  if (is.null(timing) && is.list(x) && !is.null(x$runtime_seconds)) {
    timing <- list(runtime_seconds = x$runtime_seconds)
  }
  if (is.null(timing)) stop("No timing information found.")
  seconds <- as.numeric(unlist(timing, use.names = FALSE))
  names <- names(timing)
  total <- if ("runtime_seconds" %in% names) seconds[match("runtime_seconds", names)] else sum(seconds, na.rm = TRUE)
  data.frame(
    step = names,
    seconds = seconds,
    percent = if (is.finite(total) && total > 0) 100 * seconds / total else NA_real_,
    row.names = NULL
  )
}

#' @export
kodama_timing <- KODAMA.timing

#' Diagnose wrapper runtime libraries and environment
#'
#' @param all If `TRUE`, return all linked shared libraries reported by the
#'   platform linker tool. If `FALSE`, keep only likely runtime dependencies.
#' @return A `kodama_diagnostics` list describing the package version,
#'   platform, linked libraries, relevant environment variables, and any
#'   recommended runtime preload libraries.
#' @examples
#' KODAMA.diagnostics()
#' @aliases kodama_diagnostics
#' @export
KODAMA.diagnostics <- function(all = FALSE) {
  lib <- system.file("libs", paste0("KODAMA", .Platform$dynlib.ext), package = "KODAMA")
  linker <- if (.Platform$OS.type == "unix" && Sys.info()[["sysname"]] == "Darwin") "otool" else "ldd"
  args <- if (linker == "otool") c("-L", lib) else lib
  linked <- character()
  if (nzchar(lib) && nzchar(Sys.which(linker))) {
    linked <- tryCatch(system2(linker, args, stdout = TRUE, stderr = TRUE), error = function(e) character())
  }
  if (!all && length(linked)) {
    linked <- grep("omp|gomp|blas|openblas|mkl|cuda|cublas|cufft|stdc", linked, ignore.case = TRUE, value = TRUE)
  }
  env <- Sys.getenv(c(
    "CONDA_PREFIX", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"
  ), unset = "")
  conda <- env[["CONDA_PREFIX"]]
  recommended <- character()
  if (nzchar(conda) && .Platform$OS.type == "unix" && Sys.info()[["sysname"]] != "Darwin") {
    candidates <- file.path(conda, "lib", c("libgomp.so", "libopenblasp-r0.3.33.so", "libstdc++.so.6"))
    recommended <- candidates[file.exists(candidates)]
  }
  out <- list(
    package = as.character(utils::packageVersion("KODAMA")),
    platform = paste(R.version$platform, R.version$version.string, sep = " / "),
    shared_object = lib,
    linked_libraries = linked,
    environment = env,
    recommended_ld_preload = recommended
  )
  class(out) <- "kodama_diagnostics"
  out
}

#' @export
kodama_diagnostics <- KODAMA.diagnostics

#' @export
print.kodama_matrix <- function(x, ...) {
  cat("KODAMA matrix result\n")
  cat("  classifier:", x$classifier, "\n")
  cat("  backend:", x$backend, "\n")
  cat("  runs:", x$parameters$M, " cycles:", x$parameters$Tcycle, "\n")
  cat("  samples:", ncol(x$res), " graph neighbors:", if (!is.null(x$knn$indices)) ncol(x$knn$indices) else NA_integer_, "\n")
  if (!is.na(x$best_run)) {
    cat("  best run:", x$best_run, " acc:", format(x$acc[[x$best_run]], digits = 4),
        " classes:", x$class_counts[[x$best_run]], "\n")
  }
  cat("  runtime:", format(x$runtime_seconds, digits = 4), "sec\n")
  invisible(x)
}

#' @export
print.kodama_diagnostics <- function(x, ...) {
  cat("KODAMA diagnostics\n")
  cat("  package:", x$package, "\n")
  cat("  shared object:", x$shared_object, "\n")
  if (length(x$recommended_ld_preload)) {
    cat("  recommended LD_PRELOAD:\n")
    cat("    ", paste(x$recommended_ld_preload, collapse = ":"), "\n", sep = "")
  }
  if (length(x$linked_libraries)) {
    cat("  linked libraries:\n")
    cat(paste0("    ", x$linked_libraries), sep = "\n")
    cat("\n")
  }
  invisible(x)
}

#' Build the reusable KODAMA graph and visualization initialization
#'
#' @param data Numeric matrix with samples in rows and variables in columns.
#' @param spatial Optional spatial coordinates. When supplied, their invariant
#'   nearest-neighbor graph and jitter scale are prepared once here and reused
#'   by every classifier passed to `KODAMA.matrix`.
#' @param samples Optional sample or slide identifier. Multiple samples are
#'   separated along spatial coordinate 1 using the original KODAMA rule.
#' @param k Number of nearest neighbors to retain.
#' @param metric Distance or similarity metric.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of worker threads. `NULL` uses
#'   `options(n.cores = ...)`, then `N_CORES`, and otherwise 4.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @param seed Integer seed used by the backend-specific PCA initialization.
#' @param storage Graph representation. `"matrix"` returns conventional R
#'   matrices; `"handle"` keeps one float32 graph behind a process-local
#'   external pointer. Materialize a handle before saving it for another R
#'   session.
#' @return A `kodama_graph` object containing either graph matrices or a graph
#'   handle, plus PCA starts for UMAP and openTSNE. Raw data are not retained.
#' @aliases KODAMA.makeSNNGraph makeSNNGraph
#' @examples
#' g <- KODAMA.graph(as.matrix(iris[, 1:4]), k = 5, n.cores = 1)
#' g$parameters
#' @export
KODAMA.graph <- function(data,
                         spatial = NULL,
                         samples = NULL,
                         k = 100L,
                         metric = c("euclidean", "cosine", "inner_product"),
                         backend = NULL,
                         n.cores = NULL,
                         gpu.device = 0L,
                         seed = 1234L,
                         storage = c("handle", "matrix")) {
  metric <- match.arg(metric)
  backend <- kodama_resolve_backend(backend)
  n.cores <- kodama_resolve_n_cores(n.cores, default = 4L)
  storage <- match.arg(storage)
  data_matrix <- as_kodama_matrix(data)
  sample_ids <- as_kodama_samples(samples, nrow(data_matrix))
  result <- kodama_knn_graph_cpp(
    data_matrix,
    if (is.null(spatial)) NULL else as_kodama_matrix(spatial),
    sample_ids,
    as.integer(k),
    metric,
    backend,
    as.integer(n.cores),
    as.integer(gpu.device),
    as.integer(seed),
    storage
  )
  result$parameters <- list(
    k = as.integer(k),
    spatial = !is.null(spatial),
    samples = if (is.null(sample_ids)) 0L else length(unique(sample_ids)),
    metric = metric,
    backend = backend,
    n.cores = as.integer(n.cores),
    gpu.device = as.integer(gpu.device),
    seed = as.integer(seed),
    storage = storage
  )
  class(result) <- unique(c("kodama_graph", class(result)))
  result
}

#' Materialize a handle-backed KODAMA graph
#'
#' @param graph A graph returned by `KODAMA.graph(storage = "handle")` or a
#'   KODAMA result whose `knn` member uses handle storage.
#' @return A graph list containing R index and distance matrices.
#'   Materialized graphs can be serialized normally.
#' @examples
#' g <- KODAMA.graph(as.matrix(iris[, 1:4]), k = 5, storage = "handle")
#' materialized <- KODAMA.graph.materialize(g)
#' dim(materialized$indices)
#' @export
KODAMA.graph.materialize <- function(graph) {
  graph <- extract_kodama_graph(graph)
  if (is.null(graph)) stop("graph is not a KODAMA graph.")
  if (!kodama_graph_is_handle(graph)) return(graph)
  out <- kodama_graph_materialize_cpp(graph$handle)
  metadata <- graph[setdiff(names(graph), c("handle", "indices", "distances"))]
  for (name in names(metadata)) out[[name]] <- metadata[[name]]
  out$storage <- "matrix"
  class(out) <- unique(c("kodama_graph", class(out)))
  out
}

#' @export
KODAMA.makeSNNGraph <- KODAMA.graph

#' @export
makeSNNGraph <- KODAMA.graph

#' Backend-native float32 PCA
#'
#' @param data Numeric matrix with observations in rows.
#' @param ncomp Number of principal components.
#' @param center Whether to center columns.
#' @param scale Whether to scale centered columns to unit sample standard deviation.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of worker threads. `NULL` uses
#'   `options(n.cores = ...)`, then `N_CORES`, and otherwise 1.
#' @param gpu.device Accelerator device id.
#' @param seed Random seed for the Gaussian subspace sketch.
#' @param oversample Optional randomized-SVD oversampling width. `NULL` uses the
#'   standalone backend's automatic policy.
#' @param power Optional randomized-SVD power count. `NULL` uses the standalone
#'   backend's automatic policy.
#' @return A list containing scores, loadings, singular values, explained
#'   variance, preprocessing vectors, backend metadata, and runtime.
#' @examples
#' fit <- KODAMA.pca(as.matrix(iris[, 1:4]), ncomp = 2)
#' dim(fit$scores)
#' @export
kodama_pca <- function(data,
                       ncomp = 2L,
                       center = TRUE,
                       scale = FALSE,
                       backend = NULL,
                       n.cores = NULL,
                       gpu.device = 0L,
                       seed = 4L,
                       oversample = NULL,
                       power = NULL) {
  backend <- kodama_resolve_backend(backend)
  n.cores <- kodama_resolve_n_cores(n.cores, default = 1L)
  kodama_pca_cpp(
    as_kodama_matrix(data),
    ncomp = as.integer(ncomp),
    center = isTRUE(center),
    scale = isTRUE(scale),
    backend = backend,
    seed = as.integer(seed),
    n_threads = as.integer(n.cores),
    gpu_device = as.integer(gpu.device),
    oversample = if (is.null(oversample)) -1L else as.integer(oversample),
    power = if (is.null(power)) -1L else as.integer(power)
  )
}

#' @rdname kodama_pca
#' @export
KODAMA.pca <- kodama_pca

#' Visualize a matrix or KODAMA graph with UMAP or openTSNE
#'
#' @param x Input matrix, KODAMA result, or KNN graph list.
#' @param method Embedding method.
#' @param init Optional two-column initialization matrix.
#' @param raw.data Optional raw data used for backend-native PCA
#'   initialization. When `x` is a raw matrix, that matrix is used
#'   automatically. A stored initialization from `KODAMA.matrix()` is reused
#'   first when its backend matches `backend`; `raw.data` is used only when a
#'   compatible stored start is unavailable.
#' @param initialize.from.raw Whether raw-data PCA initialization is the
#'   default when no explicit `init` is supplied.
#' @param k Number of graph neighbors used by the embedding.
#' @param metric Distance or similarity metric used when `x` is a matrix.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"` for
#'   UMAP and openTSNE.
#' @param n.cores Number of worker threads. `NULL` uses
#'   `options(n.cores = ...)`, then `N_CORES`, and otherwise 4.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @param graph.mode UMAP graph weighting mode. `"fuzzy"` is the default;
#'   `"binary"` remains available as an explicit compatibility mode.
#' @param n.epochs Number of UMAP optimization epochs.
#' @param n.iter Number of openTSNE optimization iterations.
#' @param perplexity openTSNE perplexity.
#' @param seed Integer random seed.
#' @param ... Additional embedding parameters forwarded to the C++ wrapper.
#' @return A two-column numeric matrix with attributes describing the selected
#'   backend, optimizer, initialization source and backend, runtime, and (for
#'   UMAP) fuzzy-graph edge count and maximum weight. The graph diagnostics are
#'   zero for openTSNE because its sparse probability matrix is not a UMAP
#'   fuzzy graph.
#' @export
KODAMA.visualization <- function(x,
                                 method = c("UMAP", "t-SNE", "opentsne"),
                                 init = NULL,
                                 raw.data = NULL,
                                 initialize.from.raw = TRUE,
                                 k = 30L,
                                 metric = c("euclidean", "cosine", "inner_product"),
                                 backend = NULL,
                                 n.cores = NULL,
                                 gpu.device = 0L,
                                 n.epochs = 200L,
                                 n.iter = 500L,
                                 perplexity = 30,
                                 graph.mode = c("fuzzy", "binary"),
                                 seed = 4L,
                                 ...) {
  method <- match.arg(method)
  metric <- match.arg(metric)
  backend <- kodama_resolve_backend(backend)
  n.cores <- kodama_resolve_n_cores(n.cores, default = 4L)
  graph.mode <- match.arg(graph.mode)
  raw_matrix <- if (!is.null(raw.data)) {
    as_kodama_matrix(raw.data)
  } else if (is.matrix(x) || is.data.frame(x)) {
    as_kodama_matrix(x)
  } else NULL
  graph <- extract_kodama_graph(x)
  if (is.null(graph)) {
    graph <- KODAMA.graph(
      x, k = k, metric = metric, backend = backend, n.cores = n.cores,
      gpu.device = gpu.device, storage = "handle"
    )
  }
  init_source <- if (is.null(init)) "" else "explicit"
  init_backend <- if (is.null(init)) "auto" else backend
  if (is.null(init)) {
    method_key <- if (method == "UMAP") "umap" else "opentsne"
    if (isTRUE(initialize.from.raw)) {
      init <- kodama_visual_init(x, method_key, backend = backend)
      if (!is.null(init)) {
        init_source <- "raw_pca"
        init_backend <- backend
      } else if (!is.null(raw_matrix)) {
        generated <- kodama_make_visual_init(
          raw_matrix,
          seed = seed,
          backend = backend,
          n.cores = n.cores,
          gpu.device = gpu.device
        )
        init <- generated[[method_key]]
        init_source <- "raw_pca"
        init_backend <- generated$backend
      } else if (is.list(x) && is.list(x$visual_init) &&
                 !is.null(x$visual_init$backend) &&
                 !identical(as.character(x$visual_init$backend), backend)) {
        warning(
          "The stored raw-data PCA initialization uses backend '",
          x$visual_init$backend, "', not '", backend,
          "'. Pass raw.data to recompute it on the selected backend; ",
          "using the graph-only fallback for this call.",
          call. = FALSE
        )
      }
    }
  }
  if (method == "UMAP") {
    if (kodama_graph_is_handle(graph)) {
      return(kodama_umap_graph_handle_cpp(
        graph$handle,
        if (is.null(init)) NULL else as_kodama_matrix(init),
        as.integer(k),
        as.integer(n.epochs),
        backend = backend,
        n_threads = as.integer(n.cores),
        seed = as.integer(seed),
        gpu_device = as.integer(gpu.device),
        graph_mode = graph.mode,
        init_source = init_source,
        init_backend = init_backend,
        ...
      ))
    }
    return(kodama_umap_cpp(
      graph$indices,
      graph$distances,
      if (is.null(init)) NULL else as_kodama_matrix(init),
      as.integer(k),
      as.integer(n.epochs),
      backend = backend,
      n_threads = as.integer(n.cores),
      seed = as.integer(seed),
      gpu_device = as.integer(gpu.device),
      graph_mode = graph.mode,
      init_source = init_source,
      init_backend = init_backend,
      ...
    ))
  }
  if (kodama_graph_is_handle(graph)) {
    return(kodama_opentsne_graph_handle_cpp(
      graph$handle,
      if (is.null(init)) NULL else as_kodama_matrix(init),
      n_neighbors = as.integer(k),
      perplexity = as.numeric(perplexity),
      n_iter = as.integer(n.iter),
      backend = backend,
      n_threads = as.integer(n.cores),
      seed = as.integer(seed),
      gpu_device = as.integer(gpu.device),
      init_source = init_source,
      init_backend = init_backend,
      ...
    ))
  }
  kodama_opentsne_cpp(
    graph$indices,
    graph$distances,
    if (is.null(init)) NULL else as_kodama_matrix(init),
    n_neighbors = as.integer(k),
    perplexity = as.numeric(perplexity),
    n_iter = as.integer(n.iter),
    backend = backend,
    n_threads = as.integer(n.cores),
    seed = as.integer(seed),
    gpu_device = as.integer(gpu.device),
    init_source = init_source,
    init_backend = init_backend,
    ...
  )
}

#' Cluster a graph or embedding with fastEmbedR
#'
#' This is a KODAMA adapter around [fastEmbedR::knn_graph()] and
#' [fastEmbedR::graph_cluster()]. It does not maintain a second clustering
#' implementation. KODAMA and precomputed KNN graphs are converted once to
#' fastEmbedR's compact graph representation before clustering.
#'
#' @param x Input embedding matrix, KODAMA result, KNN graph, or
#'   `fastEmbedR_graph`.
#' @param method Clustering method: `"leiden"`, `"louvain"`, or `"walktrap"`.
#' @param k Number of neighbors used when a graph must be built from a matrix.
#' @param metric Distance metric used only when neighbors must be calculated.
#' @param weight Graph edge weighting used during graph construction.
#' @param mutual Keep only reciprocal KNN edges.
#' @param prune Remove graph edges with weight less than or equal to this value.
#' @param graph.backend Backend used to construct a graph from a matrix.
#' @param backend Backend used by Louvain or Leiden. Walktrap is CPU-only.
#' @param resolution Modularity resolution for Louvain and Leiden.
#' @param n.iterations Maximum local-moving iterations.
#' @param n.runs Independent seeded Louvain or Leiden runs.
#' @param steps Random-walk length for Walktrap.
#' @param n.cores CPU workers used to construct the graph.
#' @param seed Reproducible seed for Louvain and Leiden.
#' @return A `fastEmbedR_graph_cluster` result containing one-based membership,
#'   modularity, implementation metadata, and timing.
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(40, -2), 20, 2), matrix(rnorm(40, 2), 20, 2))
#' fit <- KODAMA.clustering(x, method = "leiden", k = 5, n.cores = 1)
#' table(fit$membership)
#' @export
KODAMA.clustering <- function(x,
                              method = c("leiden", "louvain", "walktrap"),
                              k = 30L,
                              metric = c("euclidean", "cosine", "correlation"),
                              weight = c("snn", "distance", "binary"),
                              mutual = FALSE,
                              prune = 0,
                              graph.backend = NULL,
                              backend = NULL,
                              resolution = 1,
                              n.iterations = 10L,
                              n.runs = 1L,
                              steps = 4L,
                              n.cores = NULL,
                              seed = 1L) {
  method <- match.arg(method)
  weight <- match.arg(weight)
  metric <- match.arg(metric)
  n.cores <- kodama_resolve_n_cores(n.cores, default = 1L)
  graph_input <- extract_kodama_graph(x)
  if (!is.null(graph_input)) {
    if (kodama_graph_is_handle(graph_input)) {
      graph_input <- KODAMA.graph.materialize(graph_input)
    }
    x <- list(
      indices = graph_input$indices,
      distances = graph_input$distances
    )
  }
  graph <- fastEmbedR::knn_graph(
    x,
    k = k,
    backend = graph.backend,
    metric = metric,
    weight = weight,
    mutual = mutual,
    prune = prune,
    n.cores = n.cores
  )
  fastEmbedR::graph_cluster(
    graph,
    method = method,
    backend = backend,
    resolution = resolution,
    n_iterations = n.iterations,
    n_runs = n.runs,
    steps = steps,
    seed = seed
  )
}
