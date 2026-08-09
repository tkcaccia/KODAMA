# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

test_that("KNN classifier defaults are benchmark-selected without changing graph defaults", {
  expect_identical(formals(CoreKNN)$k, 10L)
  expect_identical(formals(kodama_matrix)$knn.k, 10L)
  expect_identical(formals(kodama_matrix_graph)$knn.k, 10L)
  expect_identical(formals(KODAMA.graph)$k, 100L)
  expect_identical(formals(KODAMA.visualization)$k, 30L)
})

test_that("kodama_matrix runs KNN and PLS-LDA on a small matrix", {
  set.seed(1)
  x <- matrix(rnorm(90 * 6), 90, 6)
  spatial <- matrix(rnorm(90 * 2), 90, 2)
  labels <- rep(1:3, length.out = nrow(x))

  knn <- kodama_matrix(
    x,
    spatial = spatial,
    M = 1,
    Tcycle = 1,
    landmarks = 60,
    classifier = "knn",
    backend = "cpu",
    progress = FALSE,
    return.graph = TRUE
  )
  pls <- kodama_matrix(
    x,
    spatial = spatial,
    M = 1,
    Tcycle = 1,
    ncomp = 3,
    landmarks = 60,
    classifier = "pls_lda",
    backend = "cpu",
    progress = FALSE
  )

  expect_equal(ncol(knn$res), nrow(x))
  expect_equal(ncol(pls$res), nrow(x))
  expect_equal(knn$analysis_storage, "float32")
  expect_equal(pls$analysis_storage, "float32")
  expect_s3_class(knn, "kodama_matrix")
  expect_identical(knn$visual_init$backend, "cpu")
  expect_match(knn$visual_init$method, "kodama_cpp_cpu_")
  expect_identical(knn$visual_init$precision, "float32")
  expect_equal(dim(knn$visual_init$umap), c(nrow(x), 2L))
  expect_equal(dim(knn$visual_init$opentsne), c(nrow(x), 2L))
  expect_identical(knn$graph_builds, 1L)
  expect_identical(knn$n.cores, 4L)
  expect_false(knn$gpu_auto_workers)
  expect_false(knn$gpu_scheduler_enabled)
  expect_identical(knn$gpu_scheduler_lanes, 0L)
  expect_equal(knn$gpu_worker_memory_estimate_mb, 0)
  expect_true(isTRUE(knn$knn_is_kodama_corrected))
  expect_null(knn$base_knn)
  expect_gte(
    knn$graph_storage_bytes,
    length(knn$knn$indices) * 4 + length(knn$knn$distances) * 4
  )
  expect_true(knn$timing$visual_init_seconds >= 0)
  expect_length(knn$landmark_seconds, 1L)
  expect_length(knn$landmark_occupied_strata, 1L)
  expect_length(knn$landmark_represented_strata, 1L)
  expect_length(knn$landmark_grid_bins, 1L)
  expect_equal(knn$timing$landmark_sum_seconds, sum(knn$landmark_seconds))
  expect_equal(knn$timing$landmark_mean_seconds, mean(knn$landmark_seconds))
  expect_equal(knn$timing$landmark_median_seconds, median(knn$landmark_seconds))
  expect_length(knn$best_labels, nrow(x))
  expect_length(knn$class_counts, 1L)
  expect_equal(knn$parameters$classifier, "knn")
  expect_true(all(c("landmark_sum_seconds", "runtime_seconds") %in% KODAMA.timing(knn)$step))
  emb <- KODAMA.visualization(
    knn,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu"
  )
  tsne <- KODAMA.visualization(
    knn,
    method = "opentsne",
    k = 5,
    perplexity = 1,
    n.iter = 2,
    early_exaggeration_iter = 1,
    backend = "cpu"
  )
  expect_identical(attr(emb, "initialization"), "raw_pca")
  expect_identical(attr(emb, "initialization_backend"), "cpu")
  expect_identical(attr(tsne, "initialization"), "raw_pca")
  expect_identical(attr(tsne, "initialization_backend"), "cpu")

  local_mocked_bindings(
    kodama_make_visual_init = function(...) {
      stop("stored initialization was not reused")
    },
    .package = "kodamaR"
  )
  emb_with_raw <- KODAMA.visualization(
    knn,
    method = "UMAP",
    raw.data = x,
    k = 5,
    n.epochs = 1,
    backend = "cpu"
  )
  expect_identical(attr(emb_with_raw, "initialization"), "raw_pca")
})

test_that("public API wrappers are exposed", {
  set.seed(2)
  x <- matrix(rnorm(60 * 5), 60, 5)
  labels <- rep(1:3, length.out = nrow(x))

  knncv <- KNNCV(x, labels, folds = 3, k = 3, backend = "cpu")
  pls <- PLSLDACV(x, labels, folds = 3, ncomp = 2, backend = "cpu")
  core_knn <- CoreKNN(x, labels, cycles = 1, folds = 3, k = 3, backend = "cpu")
  core_pls <- CorePLSLDA(x, labels, cycles = 1, folds = 3, ncomp = 2, backend = "cpu")
  pca <- KODAMA.pca(x, ncomp = 3, backend = "cpu", seed = 4)
  graph <- KODAMA.graph(x, k = 5, backend = "cpu")
  expect_s3_class(graph$timings, "data.frame")
  expect_true(all(c("knn_graph", "total") %in% graph$timings$step))
  emb_default <- KODAMA.visualization(
    graph,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu"
  )
  emb_fuzzy <- KODAMA.visualization(
    graph,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu",
    graph.mode = "fuzzy"
  )
  emb_binary <- KODAMA.visualization(
    graph,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu",
    graph.mode = "binary"
  )
  emb_raw <- KODAMA.visualization(
    x,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu"
  )
  tsne_raw <- KODAMA.visualization(
    x,
    method = "opentsne",
    k = 5,
    perplexity = 1,
    n.iter = 2,
    early_exaggeration_iter = 1,
    backend = "cpu"
  )
  clu <- KODAMA.clustering(graph, n.iterations = 2, random.walk.steps = 2)

  expect_length(knncv$predicted, nrow(x))
  expect_length(pls$predicted, nrow(x))
  expect_length(core_knn$clbest, nrow(x))
  expect_length(core_pls$clbest, nrow(x))
  transition_fields <- c(
    "proposals_evaluated", "best_state_updates", "current_state_accepts",
    "stochastic_state_attempts", "stochastic_state_accepts",
    "current_state_rejections",
    "coarsening_moves", "absorption_moves"
  )
  expect_true(all(transition_fields %in% names(core_knn)))
  expect_true(all(transition_fields %in% names(core_pls)))
  expect_identical(core_knn$proposals_evaluated, core_knn$cycles_completed)
  expect_identical(core_pls$proposals_evaluated, core_pls$cycles_completed)
  expect_identical(core_knn$current_state_accepts + core_knn$current_state_rejections, 0L)
  expect_identical(core_pls$current_state_accepts + core_pls$current_state_rejections, 0L)
  expect_lte(core_knn$stochastic_state_accepts, core_knn$stochastic_state_attempts)
  expect_lte(core_pls$stochastic_state_accepts, core_pls$stochastic_state_attempts)
  expect_identical(knncv$backend, "cpu")
  expect_identical(pls$backend, "cpu")
  expect_identical(core_knn$backend, "cpu")
  expect_identical(core_pls$backend, "cpu")
  expect_equal(dim(pca$scores), c(nrow(x), 3L))
  expect_equal(dim(pca$loadings), c(ncol(x), 3L))
  expect_equal(pca$precision, "float32")
  expect_true(all(diff(pca$singular_values) <= 1e-5))
  expect_equal(dim(KODAMA.graph.materialize(graph)$indices), c(nrow(x), 5L))
  expect_s3_class(graph$handle, "kodama_graph_handle")
  expect_identical(graph$backend, "cpu")
  expect_s3_class(graph, "kodama_graph")
  expect_null(graph$data)
  expect_equal(dim(graph$visual_init$umap), c(nrow(x), 2L))
  expect_equal(dim(graph$visual_init$opentsne), c(nrow(x), 2L))
  expect_equal(dim(emb_default), c(nrow(x), 2L))
  expect_equal(dim(emb_fuzzy), c(nrow(x), 2L))
  expect_equal(dim(emb_binary), c(nrow(x), 2L))
  expect_equal(as.numeric(emb_default), as.numeric(emb_fuzzy), tolerance = 0)
  expect_identical(attr(emb_default, "initialization"), "raw_pca")
  expect_identical(attr(emb_default, "backend"), "cpu")
  expect_identical(attr(emb_default, "optimizer"), "csr_epoch_schedule")
  expect_gt(attr(emb_default, "graph_edges"), 0)
  expect_equal(attr(emb_default, "graph_max_weight"), 1)
  expect_gte(attr(emb_default, "runtime_seconds"), 0)
  expect_identical(attr(emb_raw, "initialization"), "raw_pca")
  expect_identical(
    attr(tsne_raw, "optimizer"),
    "opentsne_fitsne_fft_grid_sparse_knn_float32"
  )
  expect_identical(attr(tsne_raw, "backend"), "cpu")
  expect_equal(attr(tsne_raw, "graph_edges"), 0)
  expect_equal(attr(tsne_raw, "graph_max_weight"), 0)
  expect_identical(attr(emb_raw, "initialization_backend"), "cpu")
  expect_identical(attr(tsne_raw, "initialization"), "raw_pca")
  expect_true(all(is.finite(emb_default)))
  expect_true(all(is.finite(emb_fuzzy)))
  expect_true(all(is.finite(emb_binary)))
  expect_true(all(is.finite(emb_raw)))
  expect_true(all(is.finite(tsne_raw)))
  expect_length(clu$membership, nrow(x))
  metal_graph <- KODAMA.graph.materialize(graph)
  metal_umap <- tryCatch(
    kodamaR:::kodama_umap_cpp(
      metal_graph$indices,
      metal_graph$distances,
      n_neighbors = 5L,
      n_epochs = 3L,
      backend = "metal"
    ),
    error = identity
  )
  if (inherits(metal_umap, "error")) {
    skip(paste(
      "the current process cannot access a Metal device:",
      conditionMessage(metal_umap)
    ))
  }
  expect_false(inherits(metal_umap, "error"))
  expect_equal(dim(metal_umap), c(nrow(x), 2L))
  expect_identical(attr(metal_umap, "backend"), "metal")
  expect_identical(
    attr(metal_umap, "optimizer"), "metal_clean_atomic_edge_sampler"
  )
  expect_gt(attr(metal_umap, "graph_edges"), 0)
  expect_equal(attr(metal_umap, "graph_max_weight"), 1)
  expect_true(all(is.finite(metal_umap)))
  metal_tsne <- kodamaR:::kodama_opentsne_cpp(
      metal_graph$indices,
      metal_graph$distances,
      perplexity = 3,
      early_exaggeration_iter = 2L,
      n_iter = 3L,
      backend = "metal"
  )
  expect_equal(dim(metal_tsne), c(nrow(x), 2L))
  expect_identical(attr(metal_tsne, "backend"), "metal")
  expect_identical(
    attr(metal_tsne, "optimizer"),
    "metal_opentsne_fft_grid_sparse_knn_float32"
  )
  expect_equal(attr(metal_tsne, "graph_edges"), 0)
  expect_equal(attr(metal_tsne, "graph_max_weight"), 0)
  expect_true(all(is.finite(metal_tsne)))
})

test_that("KODAMA.matrix accepts all raw and graph input forms", {
  set.seed(3)
  x <- matrix(rnorm(60 * 6), 60, 6)
  prepared <- KODAMA.graph(
    x,
    k = 15,
    backend = "cpu",
    n.cores = 1,
    seed = 9
  )
  prepared_matrix <- KODAMA.graph.materialize(prepared)
  bare <- list(
    indices = prepared_matrix$indices,
    distances = prepared_matrix$distances
  )
  common <- list(
    M = 1L,
    Tcycle = 1L,
    ncomp = 3L,
    landmarks = 40L,
    splitting = 5L,
    n.cores = 1L,
    graph.neighbors = 15L,
    knn.k = 5L,
    classifier = "knn",
    backend = "cpu",
    seed = 9L,
    progress = FALSE
  )

  from_raw <- do.call(KODAMA.matrix, c(list(data = x), common))
  from_prepared <- do.call(KODAMA.matrix, c(list(graph = prepared), common))
  from_prepared_data <- do.call(
    KODAMA.matrix,
    c(list(data = x, graph = prepared), common)
  )
  from_bare <- do.call(KODAMA.matrix, c(list(graph = bare), common))
  labels_only <- do.call(
    KODAMA.matrix,
    c(list(data = x, return.graph = FALSE), common)
  )

  expect_identical(from_raw$graph_builds, 1L)
  expect_identical(from_prepared$graph_builds, 0L)
  expect_identical(from_prepared_data$graph_builds, 0L)
  expect_identical(from_bare$graph_builds, 0L)
  expect_equal(from_prepared$parameters$graph.uses.data.geometry, FALSE)
  expect_equal(from_prepared_data$parameters$graph.uses.data.geometry, TRUE)
  expect_equal(from_bare$parameters$graph.uses.data.geometry, FALSE)
  expect_equal(dim(from_prepared_data$res), dim(from_raw$res))
  expect_null(labels_only$knn)
  expect_equal(labels_only$res, from_raw$res)
  expect_equal(
    as.numeric(from_prepared$visual_init$umap),
    as.numeric(prepared$visual_init$umap)
  )
  expect_null(from_bare$visual_init)
  expect_error(
    do.call(KODAMA.matrix, c(list(data = prepared), common)),
    "pass graph inputs through graph"
  )
})

test_that("handle-backed graphs avoid R matrices and preserve results", {
  set.seed(31)
  x <- matrix(rnorm(240 * 7), 240, 7)
  handle_graph <- KODAMA.graph(
    x, k = 15, backend = "cpu", n.cores = 2, seed = 12,
    storage = "handle"
  )

  expect_s3_class(handle_graph, "kodama_graph")
  expect_s3_class(handle_graph$handle, "kodama_graph_handle")
  expect_null(handle_graph$indices)
  expect_null(handle_graph$distances)
  expect_equal(handle_graph$samples, nrow(x))
  expect_equal(handle_graph$neighbors, 15L)

  invalid_handle <- new("externalptr")
  class(invalid_handle) <- "kodama_graph_handle"
  expect_error(
    KODAMA.graph.materialize(list(handle = invalid_handle)),
    "no longer valid"
  )

  materialized <- KODAMA.graph.materialize(handle_graph)
  matrix_graph <- materialized
  matrix_graph$visual_init <- handle_graph$visual_init
  matrix_graph$backend <- handle_graph$backend
  expect_identical(KODAMA.graph.materialize(handle_graph)$indices, matrix_graph$indices)
  expect_equal(
    KODAMA.graph.materialize(handle_graph)$distances,
    matrix_graph$distances,
    tolerance = 0
  )

  common <- list(
    M = 1L, Tcycle = 1L, landmarks = 160L, splitting = 8L,
    graph.neighbors = 15L, knn.k = 5L, backend = "cpu",
    seed = 12L, progress = FALSE
  )
  from_matrix <- do.call(KODAMA.matrix, c(list(graph = matrix_graph), common))
  from_handle <- do.call(
    KODAMA.matrix,
    c(list(graph = handle_graph, return.graph = "handle"), common)
  )
  expect_identical(from_handle$res, from_matrix$res)
  expect_s3_class(from_handle$knn$handle, "kodama_graph_handle")

  matrix_umap <- KODAMA.visualization(
    matrix_graph, method = "UMAP", k = 10, n.epochs = 3,
    backend = "cpu", seed = 12
  )
  handle_umap <- KODAMA.visualization(
    handle_graph, method = "UMAP", k = 10, n.epochs = 3,
    backend = "cpu", seed = 12
  )
  expect_equal(as.numeric(handle_umap), as.numeric(matrix_umap), tolerance = 0)
  expect_identical(attr(handle_umap, "initialization"), "raw_pca")
})

test_that("KODAMA.matrix accepts all raw and graph input forms", {
  set.seed(3)
  x <- matrix(rnorm(60 * 6), 60, 6)
  prepared <- KODAMA.graph(
    x,
    k = 15,
    backend = "cpu",
    n.cores = 1,
    seed = 9
  )
  materialized <- KODAMA.graph.materialize(prepared)
  bare <- list(indices = materialized$indices, distances = materialized$distances)
  common <- list(
    M = 1L,
    Tcycle = 1L,
    ncomp = 3L,
    landmarks = 40L,
    splitting = 5L,
    n.cores = 1L,
    graph.neighbors = 15L,
    knn.k = 5L,
    classifier = "knn",
    backend = "cpu",
    seed = 9L,
    progress = FALSE
  )

  from_raw <- do.call(KODAMA.matrix, c(list(data = x), common))
  from_prepared <- do.call(KODAMA.matrix, c(list(graph = prepared), common))
  from_prepared_data <- do.call(
    KODAMA.matrix,
    c(list(data = x, graph = prepared), common)
  )
  from_bare <- do.call(KODAMA.matrix, c(list(graph = bare), common))

  expect_identical(from_raw$graph_builds, 1L)
  expect_identical(from_prepared$graph_builds, 0L)
  expect_identical(from_prepared_data$graph_builds, 0L)
  expect_identical(from_bare$graph_builds, 0L)
  expect_equal(from_prepared$parameters$graph.uses.data.geometry, FALSE)
  expect_equal(from_prepared_data$parameters$graph.uses.data.geometry, TRUE)
  expect_equal(from_bare$parameters$graph.uses.data.geometry, FALSE)
  expect_equal(dim(from_prepared_data$res), dim(from_raw$res))
  expect_equal(
    as.numeric(from_prepared$visual_init$umap),
    as.numeric(prepared$visual_init$umap)
  )
  expect_null(from_bare$visual_init)
  expect_error(
    do.call(KODAMA.matrix, c(list(data = prepared), common)),
    "pass graph inputs through graph"
  )
})

test_that("diagnostics report wrapper runtime information", {
  diag <- KODAMA.diagnostics()
  expect_s3_class(diag, "kodama_diagnostics")
  expect_true(nzchar(diag$package))
  expect_true("CONDA_PREFIX" %in% names(diag$environment))
})

test_that("shared k-means landmark atlas is reported", {
  set.seed(47)
  x <- matrix(rnorm(120 * 5), 120, 5)
  prepared <- KODAMA.graph(
    x, k = 12L, backend = "cpu", n.cores = 2L, storage = "handle"
  )
  result <- KODAMA.matrix(
    data = x, graph = prepared, M = 1L, Tcycle = 1L,
    landmarks = 80L, splitting = 8L, graph.neighbors = 12L,
    knn.k = 5L,
    backend = "cpu", progress = FALSE
  )
  expect_true(result$shared_landmark_partition_used)
  expect_identical(result$shared_landmark_partition_strata, 8L)
})

test_that("reviewer evolution policies are reproducible but hidden", {
  set.seed(53)
  x <- matrix(rnorm(150 * 6), 150, 6)
  prepared <- KODAMA.graph(
    x, k = 12L, backend = "cpu", n.cores = 2L, seed = 19L,
    storage = "handle"
  )
  common <- list(
    data = x, graph = prepared, M = 1L, Tcycle = 3L,
    landmarks = 100L, splitting = 8L, graph.neighbors = 12L,
    knn.k = 5L, folds = 4L, backend = "cpu", seed = 19L,
    progress = FALSE, visual.init = FALSE
  )

  ordinary <- do.call(KODAMA.matrix, common)
  explicit_full <- do.call(
    KODAMA.matrix,
    c(common, list(.evolution.policy = "full"))
  )
  no_transition <- do.call(
    KODAMA.matrix,
    c(common, list(.evolution.policy = "no_transition_proposal"))
  )

  expect_identical(ordinary$res, explicit_full$res)
  expect_identical(ordinary$best_labels, explicit_full$best_labels)
  expect_equal(nrow(ordinary$run_diagnostics), 1L)
  expect_equal(nrow(ordinary$cycle_diagnostics), 3L)
  expect_identical(ordinary$run_diagnostics$cv_evaluations, 4L)
  expect_identical(
    ordinary$run_diagnostics$landmark_rows_hash,
    no_transition$run_diagnostics$landmark_rows_hash
  )
  expect_identical(
    ordinary$run_diagnostics$initial_labels_hash,
    no_transition$run_diagnostics$initial_labels_hash
  )
  expect_identical(
    ordinary$run_diagnostics$fold_assignments_hash,
    no_transition$run_diagnostics$fold_assignments_hash
  )
  expect_identical(no_transition$run_diagnostics$transition_attempted, 0L)
  expect_false(".evolution.policy" %in% names(formals(KODAMA.matrix)))
  expect_true("..." %in% names(formals(KODAMA.matrix)))
  expect_error(
    do.call(KODAMA.matrix, c(common, list(.evolution.policy = "not_a_policy"))),
    "should be one of"
  )
  expect_error(
    do.call(KODAMA.matrix, c(common, list(.unknown.control = TRUE))),
    "Unknown reserved KODAMA.matrix control"
  )
})
