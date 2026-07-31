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

extract_kodama_graph <- function(x) {
  if (is.list(x) && !is.null(x$knn)) return(x$knn)
  if (is.list(x) && !is.null(x$indices) && !is.null(x$distances)) return(x)
  NULL
}

kodama_class_counts <- function(res) {
  if (is.null(res)) return(integer())
  apply(res, 1L, function(z) length(unique(z)))
}

kodama_best_run <- function(acc) {
  if (length(acc) == 0L || all(is.na(acc))) return(NA_integer_)
  as.integer(which.max(acc))
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
#' @export
KNNCV <- function(data,
                  labels,
                  constrain = NULL,
                  folds = 10L,
                  stratified = TRUE,
                  seed = 1L,
                  k = 10L,
                  metric = c("cosine", "inner_product", "euclidean"),
                  backend = c("cpu", "cuda", "metal"),
                  n.cores = 1L,
                  gpu.device = 0L) {
  metric <- match.arg(metric)
  backend <- match.arg(backend)
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
                     backend = c("cpu", "cuda", "metal"),
                     n.cores = 1L,
                     gpu.device = 0L) {
  backend <- match.arg(backend)
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
                    backend = c("cpu", "cuda", "metal"),
                    n.cores = 4L,
                    gpu.device = 0L) {
  metric <- match.arg(metric)
  backend <- match.arg(backend)
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
                       backend = c("cpu", "cuda", "metal"),
                       n.cores = 4L,
                       gpu.device = 0L) {
  backend <- match.arg(backend)
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
#' @param W Optional integer vector.
#' @param constrain Optional integer vector of sample constraints.
#' @param fix Optional integer vector marking fixed samples.
#' @param M Number of independent KODAMA runs.
#' @param Tcycle Number of optimization cycles per run.
#' @param ncomp Number of PLS components for the PLS-LDA classifier.
#' @param landmarks Maximum number of samples optimized directly in each run.
#' @param splitting Initial number of label classes used for each run.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param graph.neighbors Number of neighbors retained in the returned graph.
#' @param knn.k Number of neighbors used by the KNN classifier.
#' @param spatial.resolution Resolution parameter for constrained grouping when
#'   optional coordinate constraints are supplied.
#' @param spatial.graph.mix Logical flag passed to the C++ constraint builder.
#' @param spatial.constraint.mode Constraint construction mode.
#' @param metric Distance or similarity metric.
#' @param classifier Either `"knn"` or `"pls_lda"`.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param seed Integer random seed.
#' @param visual.init Whether the same native matrix call performs one float32
#'   PCA and stores both UMAP and openTSNE initializations for reuse by
#'   `KODAMA.visualization`.
#' @param progress Whether the C++ core prints run/cycle progress.
#' @param apply.kodama.dissimilarity Whether to return the KODAMA-corrected
#'   neighbor graph rather than only the base graph.
#' @return A list returned by the kodama-cpp core. The full-data graph is built
#'   once before all `M` runs. The result contains one `knn` graph;
#'   `knn_is_kodama_corrected`, `graph_storage_bytes`, `graph_builds`, and
#'   `timing` expose its lifecycle.
#' @aliases KODAMA.matrix
#' @export
kodama_matrix <- function(data = NULL,
                          graph = NULL,
                          spatial = NULL,
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
                          metric = "euclidean",
                          classifier = c("knn", "pls_lda"),
                          backend = c("cpu", "cuda", "metal"),
                          seed = 1234L,
                          visual.init = TRUE,
                          progress = TRUE,
                          apply.kodama.dissimilarity = TRUE) {
  backend_was_missing <- missing(backend)
  classifier <- match.arg(classifier)
  backend <- match.arg(backend)
  spatial.constraint.mode <- match.arg(spatial.constraint.mode)
  if (!is.null(graph)) {
    graph_input <- extract_kodama_graph(graph)
    if (is.null(graph_input)) {
      stop("graph must be a KODAMA.graph result or a list with indices and distances.")
    }
    if (!is.null(data) && !is.null(extract_kodama_graph(data))) {
      stop("data must be the raw numeric matrix; pass graph inputs through graph.")
    }
    if (backend_was_missing && !is.null(graph$backend)) {
      backend <- match.arg(as.character(graph$backend), c("cpu", "cuda", "metal"))
    }
    raw_data <- if (is.null(data)) NULL else as_kodama_matrix(data)
    samples <- nrow(graph_input$indices)
    if (!is.null(raw_data) && nrow(raw_data) != samples) {
      stop("data and graph must contain the same number of samples.")
    }
    if (is.null(ncomp)) {
      ncomp <- if (is.null(raw_data)) 50L else min(50L, ncol(raw_data))
    }
    if (is.null(splitting)) splitting <- ifelse(samples < 40000, 100L, 300L)
    if (is.null(graph.neighbors)) graph.neighbors <- ncol(graph_input$indices)
    return(kodama_matrix_graph(
      indices = graph,
      data = raw_data,
      spatial = spatial,
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
      classifier = classifier,
      backend = backend,
      seed = seed,
      visual.init = visual.init,
      progress = progress,
      apply.kodama.dissimilarity = apply.kodama.dissimilarity
    ))
  }

  if (is.null(data)) {
    stop("data or graph is required.")
  }
  if (!is.null(extract_kodama_graph(data))) {
    stop("data must be the raw numeric matrix; pass graph inputs through graph.")
  }
  data_matrix <- as_kodama_matrix(data)
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
    spatial.graph.mix = isTRUE(spatial.graph.mix),
    spatial.constraint.mode = spatial.constraint.mode,
    metric = metric,
    classifier = classifier,
    backend = backend,
    seed = as.integer(seed),
    visual.init = isTRUE(visual.init),
    apply.kodama.dissimilarity = isTRUE(apply.kodama.dissimilarity)
  )
  result <- kodama_matrix_cpp(
    data = data_matrix,
    spatial = if (is.null(spatial)) NULL else as_kodama_matrix(spatial),
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
    metric = metric,
    classifier = classifier,
    backend = backend,
    seed = as.integer(seed),
    progress = isTRUE(progress),
    apply_kodama_dissimilarity = isTRUE(apply.kodama.dissimilarity),
    compute_visual_init = isTRUE(visual.init)
  )
  as_kodama_matrix_result(result, parameters)
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
#' @param W Optional starting labels.
#' @param constrain Optional group vector.
#' @param fix Optional fixed-label mask.
#' @param M Number of independent runs.
#' @param Tcycle Number of optimization cycles per run.
#' @param ncomp Number of graph features/components used by the PLS-LDA path.
#' @param landmarks Maximum landmarks optimized directly.
#' @param splitting Initial number of classes.
#' @param n.cores CPU workers or CUDA automatic lanes when backend permits.
#' @param graph.neighbors Number of neighbors retained in the returned graph.
#' @param knn.k Number of neighbors used by the KNN classifier.
#' @param spatial.resolution Resolution used when spatial constraints are
#'   derived inside the core.
#' @param spatial.graph.mix Whether to combine spatial and feature graphs for
#'   grouping.
#' @param spatial.constraint.mode Spatial constraint strategy.
#' @param classifier Either `"knn"` or `"pls_lda"`.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param graph.feature.mode Graph-to-feature transform for PLS-LDA and
#'   graph-only initialization. The standard path is
#'   `"laplacian_self_tuning"`.
#' @param graph.feature.components Feature count; `0` uses `ncomp`.
#' @param graph.feature.steps Power iterations used by graph feature extraction.
#' @param seed Integer random seed.
#' @param visual.init Whether to propagate the PCA starts stored in a
#'   `KODAMA.graph` object.
#' @param progress Whether the C++ core prints progress.
#' @param apply.kodama.dissimilarity Whether to return the KODAMA-corrected
#'   graph.
#' @aliases KODAMA.matrix.graph
#' @export
kodama_matrix_graph <- function(indices,
                                distances = NULL,
                                data = NULL,
                                spatial = NULL,
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
                                classifier = c("knn", "pls_lda"),
                                backend = c("cpu", "cuda", "metal"),
                                graph.feature.mode = "laplacian_self_tuning",
                                graph.feature.components = 0L,
                                graph.feature.steps = 3L,
                                seed = 1234L,
                                visual.init = TRUE,
                                progress = TRUE,
                                apply.kodama.dissimilarity = TRUE) {
  graph_object <- if (is.list(indices)) indices else NULL
  supplied_graph <- extract_kodama_graph(indices)
  backend_was_missing <- missing(backend)
  if (!is.null(supplied_graph)) {
    distances <- supplied_graph$distances
    indices <- supplied_graph$indices
  }
  if (is.null(indices) || is.null(distances)) stop("indices and distances are required.")
  classifier <- match.arg(classifier)
  backend <- match.arg(backend)
  if (backend_was_missing && !is.null(graph_object$backend)) {
    backend <- match.arg(as.character(graph_object$backend), c("cpu", "cuda", "metal"))
  }
  if (is.null(ncomp)) ncomp <- if (is.null(data)) 50L else min(50L, ncol(data))
  if (is.null(splitting)) {
    splitting <- ifelse(nrow(indices) < 40000, 100L, 300L)
  }
  if (is.null(graph.neighbors)) graph.neighbors <- ncol(indices)
  spatial.constraint.mode <- match.arg(spatial.constraint.mode)
  graph.feature.mode <- match.arg(graph.feature.mode)
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
    spatial.graph.mix = isTRUE(spatial.graph.mix),
    spatial.constraint.mode = spatial.constraint.mode,
    classifier = classifier,
    backend = backend,
    graph.feature.mode = graph.feature.mode,
    graph.feature.components = as.integer(graph.feature.components),
    graph.feature.steps = as.integer(graph.feature.steps),
    graph.uses.data.geometry = !is.null(data),
    seed = as.integer(seed),
    visual.init = isTRUE(visual.init),
    apply.kodama.dissimilarity = isTRUE(apply.kodama.dissimilarity)
  )
  result <- kodama_matrix_graph_cpp(
    indices = as.matrix(indices),
    distances = as.matrix(distances),
    data = if (is.null(data)) NULL else as_kodama_matrix(data),
    spatial = if (is.null(spatial)) NULL else as_kodama_matrix(spatial),
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
    classifier = classifier,
    backend = backend,
    graph_feature_mode = graph.feature.mode,
    graph_feature_components = as.integer(graph.feature.components),
    graph_feature_steps = as.integer(graph.feature.steps),
    seed = as.integer(seed),
    progress = isTRUE(progress),
    apply_kodama_dissimilarity = isTRUE(apply.kodama.dissimilarity)
  )
  as_kodama_matrix_result(result, parameters, visual_init = visual_init)
}

#' @export
KODAMA.matrix.graph <- kodama_matrix_graph

#' Return a compact timing table for a KODAMA result
#'
#' @param x A `kodama_matrix` result or compatible list with timing fields.
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
#' @aliases kodama_diagnostics
#' @export
KODAMA.diagnostics <- function(all = FALSE) {
  lib <- system.file("libs", paste0("kodamaR", .Platform$dynlib.ext), package = "kodamaR")
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
    package = as.character(utils::packageVersion("kodamaR")),
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
  cat("kodamaR diagnostics\n")
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
#' @param k Number of nearest neighbors to retain.
#' @param metric Distance or similarity metric.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU worker threads used for native HNSW
#'   construction and graph querying.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @param seed Integer seed used by the backend-specific PCA initialization.
#' @return A memory-light `kodama_graph` object containing the KNN indices and
#'   distances plus PCA starts for UMAP and openTSNE. The raw matrix is not
#'   retained.
#' @aliases KODAMA.makeSNNGraph makeSNNGraph
#' @export
KODAMA.graph <- function(data,
                         k = 100L,
                         metric = c("euclidean", "cosine", "inner_product"),
                         backend = c("cpu", "cuda", "metal"),
                         n.cores = 4L,
                         gpu.device = 0L,
                         seed = 1234L) {
  metric <- match.arg(metric)
  backend <- match.arg(backend)
  data_matrix <- as_kodama_matrix(data)
  result <- kodama_knn_graph_cpp(
    data_matrix,
    as.integer(k),
    metric,
    backend,
    as.integer(n.cores),
    as.integer(gpu.device),
    as.integer(seed)
  )
  result$parameters <- list(
    k = as.integer(k),
    metric = metric,
    backend = backend,
    n.cores = as.integer(n.cores),
    gpu.device = as.integer(gpu.device),
    seed = as.integer(seed)
  )
  class(result) <- unique(c("kodama_graph", class(result)))
  result
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
#' @param n.cores Number of CPU worker threads.
#' @param gpu.device Accelerator device id.
#' @param seed Random seed for the Gaussian subspace sketch.
#' @param oversample Optional randomized-SVD oversampling width. `NULL` uses the
#'   backend policy from fastEmbedR.
#' @param power Optional randomized-SVD power count. `NULL` uses the backend
#'   policy from fastEmbedR.
#' @return A list containing scores, loadings, singular values, explained
#'   variance, preprocessing vectors, backend metadata, and runtime.
#' @export
kodama_pca <- function(data,
                       ncomp = 2L,
                       center = TRUE,
                       scale = FALSE,
                       backend = c("cpu", "cuda", "metal"),
                       n.cores = 1L,
                       gpu.device = 0L,
                       seed = 4L,
                       oversample = NULL,
                       power = NULL) {
  backend <- match.arg(backend)
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
#' @param backend Execution backend, either `"cpu"` or `"cuda"`.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param gpu.device CUDA device id when `backend = "cuda"`.
#' @param graph.mode UMAP graph weighting mode. `"fuzzy"` is the default;
#'   `"binary"` remains available as an explicit compatibility mode.
#' @param n.epochs Number of UMAP optimization epochs.
#' @param n.iter Number of openTSNE optimization iterations.
#' @param perplexity openTSNE perplexity.
#' @param seed Integer random seed.
#' @param ... Additional embedding parameters forwarded to the C++ wrapper.
#' @export
KODAMA.visualization <- function(x,
                                 method = c("UMAP", "t-SNE", "opentsne"),
                                 init = NULL,
                                 raw.data = NULL,
                                 initialize.from.raw = TRUE,
                                 k = 30L,
                                 metric = c("euclidean", "cosine", "inner_product"),
                                 backend = c("cpu", "cuda"),
                                 n.cores = 4L,
                                 gpu.device = 0L,
                                 n.epochs = 200L,
                                 n.iter = 500L,
                                 perplexity = 30,
                                 graph.mode = c("fuzzy", "binary"),
                                 seed = 4L,
                                 ...) {
  method <- match.arg(method)
  metric <- match.arg(metric)
  backend <- match.arg(backend)
  graph.mode <- match.arg(graph.mode)
  raw_matrix <- if (!is.null(raw.data)) {
    as_kodama_matrix(raw.data)
  } else if (is.matrix(x) || is.data.frame(x)) {
    as_kodama_matrix(x)
  } else NULL
  graph <- extract_kodama_graph(x)
  if (is.null(graph)) {
    graph <- KODAMA.graph(x, k = k, metric = metric, backend = backend, n.cores = n.cores, gpu.device = gpu.device)
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

#' Cluster a graph or embedding with random walks
#'
#' @param x Input embedding matrix, KODAMA result, or KNN graph list.
#' @param n.clusters Optional target number of clusters. Random-walk clustering
#'   reports an error when it cannot produce the requested count exactly.
#' @param weight Graph edge-weighting rule.
#' @param k Number of neighbors used when `x` is an embedding matrix.
#' @param metric Distance or similarity metric used when `x` is a matrix.
#' @param graph.backend Backend used to construct a graph from an embedding:
#'   `"cpu"`, `"cuda"`, or `"metal"`. Clustering itself runs on CPU.
#' @param n.cores Number of CPU worker threads requested by the wrapper.
#' @param n.iterations Number of clustering refinement iterations.
#' @param random.walk.steps Number of random-walk steps.
#' @param gpu.device CUDA device id when `graph.backend = "cuda"`.
#' @export
KODAMA.clustering <- function(x,
                              n.clusters = 0L,
                              weight = c("distance", "snn", "adaptive", "binary"),
                              k = 30L,
                              metric = c("euclidean", "cosine", "inner_product"),
                              graph.backend = c("cpu", "cuda", "metal"),
                              n.cores = 4L,
                              n.iterations = 10L,
                              random.walk.steps = 4L,
                              gpu.device = 0L) {
  weight <- match.arg(weight)
  metric <- match.arg(metric)
  graph.backend <- match.arg(graph.backend)
  graph <- extract_kodama_graph(x)
  if (!is.null(graph)) {
    return(kodama_graph_cluster_cpp(
      graph$indices,
      graph$distances,
      weight,
      as.integer(n.cores),
      as.integer(n.iterations),
      as.integer(random.walk.steps),
      as.integer(n.clusters),
      0,
      FALSE
    ))
  }
  kodama_embedding_cluster_cpp(
    as_kodama_matrix(x),
    graph.backend,
    weight,
    metric,
    as.integer(k),
    as.integer(n.cores),
    as.integer(n.iterations),
    as.integer(random.walk.steps),
    as.integer(n.clusters),
    0,
    FALSE,
    as.integer(gpu.device)
  )
}
