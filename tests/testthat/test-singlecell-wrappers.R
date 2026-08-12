# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

small_kodama_arguments <- function() {
  list(
    M = 1L, Tcycle = 1L, landmarks = 18L, splitting = 6L,
    graph.neighbors = 5L, knn.k = 3L, classifier = "knn",
    backend = "cpu", n.cores = 1L, seed = 4L,
    visual.init = TRUE, progress = FALSE
  )
}

test_that("direct single-cell generics preserve the matrix API", {
  set.seed(20)
  x <- matrix(rnorm(24 * 5), 24, 5)
  graph <- RunKODAMAgraph(
    x, k = 5L, backend = "cpu", n.cores = 1L, storage = "matrix"
  )
  fit <- do.call(
    RunKODAMAmatrix,
    c(list(object = x, graph = graph), small_kodama_arguments())
  )
  visualization <- RunKODAMAvisualization(
    fit, method = "UMAP", backend = "cpu", n.cores = 1L,
    n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_s3_class(graph, "kodama_graph")
  expect_s3_class(fit, "kodama_matrix")
  expect_equal(dim(visualization), c(nrow(x), 2L))

  graph_only <- do.call(
    RunKODAMAmatrix,
    c(list(object = graph), small_kodama_arguments())
  )
  expect_s3_class(graph_only, "kodama_matrix")
})

test_that("SingleCellExperiment stores and reuses KODAMA state", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  set.seed(21)
  pca <- matrix(
    rnorm(24 * 5), 24, 5,
    dimnames = list(paste0("cell", seq_len(24)), paste0("PC", seq_len(5)))
  )
  counts <- matrix(
    rpois(8 * 24, 3), 8, 24,
    dimnames = list(paste0("gene", seq_len(8)), rownames(pca))
  )
  object <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts),
    reducedDims = S4Vectors::SimpleList(PCA = pca)
  )
  object <- RunKODAMAgraph(
    object, dims = 5L, k = 5L, backend = "cpu", n.cores = 1L
  )
  graph_handle <- S4Vectors::metadata(object)$KODAMA$KODAMA$graph$handle
  expect_type(graph_handle, "externalptr")

  object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  state <- S4Vectors::metadata(object)$KODAMA$KODAMA
  expect_identical(state$graph$handle, graph_handle)
  expect_s3_class(state$matrix, "kodama_matrix")
  expect_type(state$matrix$knn$handle, "externalptr")

  object <- RunKODAMAvisualization(
    object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_true("KODAMA" %in% SingleCellExperiment::reducedDimNames(object))
  expect_equal(
    dim(SingleCellExperiment::reducedDim(object, "KODAMA")),
    c(24L, 2L)
  )
})

test_that("SpatialExperiment forwards coordinates and slide identities", {
  skip_if_not_installed("SpatialExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  set.seed(22)
  cells <- paste0("cell", seq_len(24))
  pca <- matrix(rnorm(24 * 5), 24, 5, dimnames = list(cells, NULL))
  counts <- matrix(rpois(8 * 24, 3), 8, 24,
                   dimnames = list(paste0("gene", seq_len(8)), cells))
  coordinates <- cbind(x = rep(seq_len(6), 4), y = rep(seq_len(4), each = 6))
  object <- SpatialExperiment::SpatialExperiment(
    assays = list(counts = counts, logcounts = log1p(counts)),
    reducedDims = S4Vectors::SimpleList(PCA = pca),
    spatialCoords = coordinates,
    colData = S4Vectors::DataFrame(sample_id = rep(c("slide1", "slide2"), each = 12))
  )
  tutorial_object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  tutorial_object <- RunKODAMAvisualization(
    tutorial_object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_equal(
    dim(SingleCellExperiment::reducedDim(tutorial_object, "KODAMA")),
    c(24L, 2L)
  )

  object <- RunKODAMAgraph(
    object, dims = 5L, k = 5L, backend = "cpu", n.cores = 1L,
    storage = "matrix"
  )
  graph <- S4Vectors::metadata(object)$KODAMA$KODAMA$graph
  expect_true(!is.null(graph$spatial_knn))
  expect_identical(graph$parameters$samples, 2L)

  object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  state <- S4Vectors::metadata(object)$KODAMA$KODAMA
  expect_identical(state$matrix$parameters$samples, 2L)
  object <- RunKODAMAvisualization(
    object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_equal(
    dim(SingleCellExperiment::reducedDim(object, "KODAMA")),
    c(24L, 2L)
  )

  features <- RunSpatialFeatureSelection(
    object, assay.type = "logcounts", n.cores = 1L
  )
  expect_s3_class(features, "kodama_spatial_features")
  expect_length(features$score, nrow(counts))
  expect_identical(features$sample.labels, c("slide1", "slide2"))
})

test_that("Seurat stores state separately from the final reduction", {
  skip_if_not_installed("SeuratObject")
  set.seed(23)
  cells <- paste0("cell", seq_len(24))
  counts <- matrix(rpois(8 * 24, 3), 8, 24,
                   dimnames = list(paste0("gene", seq_len(8)), cells))
  pca <- matrix(rnorm(24 * 5), 24, 5,
                dimnames = list(cells, paste0("PC_", seq_len(5))))
  object <- suppressWarnings(SeuratObject::CreateSeuratObject(counts = counts))
  object <- Seurat::NormalizeData(object, verbose = FALSE)
  coordinates <- data.frame(
    x = rep(seq_len(6), 4), y = rep(seq_len(4), each = 6),
    row.names = cells
  )
  object <- suppressWarnings({
    object[["slice1"]] <- SeuratObject::CreateFOV(
      coords = coordinates, type = "centroids"
    )
    object
  })
  object[["pca"]] <- SeuratObject::CreateDimReducObject(
    embeddings = pca, key = "PC_", assay = SeuratObject::DefaultAssay(object)
  )
  tutorial_object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  tutorial_object <- RunKODAMAvisualization(
    tutorial_object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_equal(
    dim(SeuratObject::Embeddings(tutorial_object, "KODAMA")),
    c(24L, 2L)
  )

  object <- RunKODAMAgraph(
    object, dims = 5L, k = 5L,
    backend = "cpu", n.cores = 1L
  )
  state <- SeuratObject::Misc(object, slot = "KODAMA")$KODAMA
  expect_s3_class(state$graph, "kodama_graph")
  expect_false("KODAMA" %in% names(methods::slot(object, "reductions")))

  object <- do.call(
    RunKODAMAmatrix,
    c(
      list(object = object, dims = 5L),
      small_kodama_arguments()
    )
  )
  object <- RunKODAMAvisualization(
    object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  expect_true("KODAMA" %in% names(methods::slot(object, "reductions")))
  expect_equal(dim(SeuratObject::Embeddings(object, "KODAMA")), c(24L, 2L))

  expect_identical(SpatialFeatureSelection, RunSpatialFeatureSelection)
  features <- SpatialFeatureSelection(
    object, layer = "data", n.cores = 1L
  )
  expect_s3_class(features, "kodama_spatial_features")
  expect_length(features$score, nrow(counts))
  expect_identical(features$sample.labels, "slice1")
})

test_that("Giotto methods use public dimensional-reduction storage", {
  skip_if_not_installed("Giotto")
  skip_if_not_installed("GiottoClass")
  old_cores <- getOption("mc.cores")
  old_check_core <- getOption("giotto.check_core")
  old_check_version <- getOption("giotto.check_version")
  options(mc.cores = 2)
  options(giotto.check_core = FALSE, giotto.check_version = FALSE)
  on.exit(
    options(
      mc.cores = old_cores,
      giotto.check_core = old_check_core,
      giotto.check_version = old_check_version
    ),
    add = TRUE
  )
  giotto_was_attached <- "package:Giotto" %in% search()
  if (!giotto_was_attached) {
    suppressPackageStartupMessages(library("Giotto", character.only = TRUE))
    on.exit(detach("package:Giotto", unload = FALSE), add = TRUE)
  }
  set.seed(24)
  cells <- paste0("cell", seq_len(24))
  expression <- matrix(
    rpois(8 * 24, 3), 8, 24,
    dimnames = list(paste0("gene", seq_len(8)), cells)
  )
  spatial <- data.frame(
    sdimx = rep(seq_len(6), 4), sdimy = rep(seq_len(4), each = 6),
    row.names = cells
  )
  object <- suppressWarnings(getExportedValue("Giotto", "createGiottoObject")(
    expression = expression, spatial_locs = spatial,
    cores = 1L, verbose = FALSE
  ))
  object <- suppressWarnings(Giotto::normalizeGiotto(
    object, scalefactor = 6000, verbose = FALSE
  ))
  pca <- matrix(rnorm(24 * 5), 24, 5, dimnames = list(cells, NULL))
  pca_object <- getExportedValue("GiottoClass", "createDimObj")(
    coordinates = pca, name = "pca", spat_unit = "cell", feat_type = "rna",
    method = "PCA", reduction = "cells", provenance = NULL, misc = list(),
    my_rownames = cells
  )
  object <- getExportedValue("Giotto", "setDimReduction")(
    object, x = pca_object, name = "pca", reduction = "cells",
    reduction_method = "pca", verbose = FALSE
  )
  tutorial_object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  tutorial_object <- RunKODAMAvisualization(
    tutorial_object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  tutorial_visualization <- getExportedValue("Giotto", "getDimReduction")(
    tutorial_object, spat_unit = "cell", feat_type = "rna",
    reduction = "cells", reduction_method = "kodama", name = "KODAMA",
    output = "matrix", set_defaults = FALSE
  )
  expect_equal(dim(tutorial_visualization), c(24L, 2L))

  object <- RunKODAMAgraph(
    object, dims = 5L, k = 5L, backend = "cpu", n.cores = 1L
  )
  state_object <- getExportedValue("Giotto", "getDimReduction")(
    object, spat_unit = "cell", feat_type = "rna", reduction = "cells",
    reduction_method = "kodama", name = "KODAMA", output = "dimObj",
    set_defaults = FALSE
  )
  expect_s3_class(methods::slot(state_object, "misc")$graph, "kodama_graph")

  object <- do.call(
    RunKODAMAmatrix,
    c(list(object = object, dims = 5L), small_kodama_arguments())
  )
  state_object <- getExportedValue("Giotto", "getDimReduction")(
    object, spat_unit = "cell", feat_type = "rna", reduction = "cells",
    reduction_method = "kodama", name = "KODAMA", output = "dimObj",
    set_defaults = FALSE
  )
  expect_s3_class(methods::slot(state_object, "misc")$matrix, "kodama_matrix")
  object <- RunKODAMAvisualization(
    object, dims = 5L, method = "UMAP", backend = "cpu",
    n.cores = 1L, n.epochs = 5L, k = 3L, seed = 4L
  )
  visualization <- getExportedValue("Giotto", "getDimReduction")(
    object, spat_unit = "cell", feat_type = "rna", reduction = "cells",
    reduction_method = "kodama", name = "KODAMA", output = "matrix",
    set_defaults = FALSE
  )
  expect_equal(dim(visualization), c(24L, 2L))

  features <- RunSpatialFeatureSelection(
    object, values = "normalized", n.cores = 1L
  )
  expect_s3_class(features, "kodama_spatial_features")
  expect_length(features$score, nrow(expression))
})
