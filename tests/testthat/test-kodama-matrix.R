# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

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
    progress = FALSE
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
  expect_true(isTRUE(knn$knn_is_kodama_corrected))
  expect_null(knn$base_knn)
  expect_gte(
    knn$graph_storage_bytes,
    length(knn$knn$indices) * 4 + length(knn$knn$distances) * 4
  )
  expect_true(knn$timing$visual_init_seconds >= 0)
  expect_length(knn$best_labels, nrow(x))
  expect_length(knn$class_counts, 1L)
  expect_equal(knn$parameters$classifier, "knn")
  expect_true("runtime_seconds" %in% KODAMA.timing(knn)$step)
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
  expect_identical(knncv$backend, "cpu")
  expect_identical(pls$backend, "cpu")
  expect_identical(core_knn$backend, "cpu")
  expect_identical(core_pls$backend, "cpu")
  expect_equal(dim(pca$scores), c(nrow(x), 3L))
  expect_equal(dim(pca$loadings), c(ncol(x), 3L))
  expect_equal(pca$precision, "float32")
  expect_true(all(diff(pca$singular_values) <= 1e-5))
  expect_equal(dim(graph$indices), c(nrow(x), 5L))
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
  expect_identical(attr(emb_raw, "initialization"), "raw_pca")
  expect_identical(attr(emb_raw, "initialization_backend"), "cpu")
  expect_identical(attr(tsne_raw, "initialization"), "raw_pca")
  expect_true(all(is.finite(emb_default)))
  expect_true(all(is.finite(emb_fuzzy)))
  expect_true(all(is.finite(emb_binary)))
  expect_true(all(is.finite(emb_raw)))
  expect_true(all(is.finite(tsne_raw)))
  expect_length(clu$membership, nrow(x))
  expect_error(
    kodamaR:::kodama_umap_cpp(
      graph$indices,
      graph$distances,
      backend = "metal"
    ),
    "not Metal"
  )
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
  bare <- list(indices = prepared$indices, distances = prepared$distances)
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
