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
  expect_length(knn$best_labels, nrow(x))
  expect_length(knn$class_counts, 1L)
  expect_equal(knn$parameters$classifier, "knn")
  expect_true("runtime_seconds" %in% KODAMA.timing(knn)$step)
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
  emb <- KODAMA.visualization(
    graph,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu",
    graph.mode = "binary"
  )
  emb_fuzzy <- KODAMA.visualization(
    graph,
    method = "UMAP",
    k = 5,
    n.epochs = 3,
    backend = "cpu",
    graph.mode = "fuzzy"
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
  expect_equal(dim(emb), c(nrow(x), 2L))
  expect_equal(dim(emb_fuzzy), c(nrow(x), 2L))
  expect_true(all(is.finite(emb)))
  expect_true(all(is.finite(emb_fuzzy)))
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

test_that("diagnostics report wrapper runtime information", {
  diag <- KODAMA.diagnostics()
  expect_s3_class(diag, "kodama_diagnostics")
  expect_true(nzchar(diag$package))
  expect_true("CONDA_PREFIX" %in% names(diag$environment))
})
