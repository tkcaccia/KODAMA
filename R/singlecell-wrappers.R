# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT
# Adapted from tkcaccia/KODAMAextra R/main.R by its author.

.kodama_require_namespace <- function(package, framework) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(
      framework, " support requires the optional package '", package, "'.",
      call. = FALSE
    )
  }
}

.kodama_export <- function(package, function_name, framework = package) {
  .kodama_require_namespace(package, framework)
  getExportedValue(package, function_name)
}

.kodama_select_dimensions <- function(data, dims, context) {
  data <- as.matrix(data)
  if (!is.numeric(data) || length(dim(data)) != 2L || ncol(data) < 1L) {
    stop(context, " must contain a numeric dimensional reduction.", call. = FALSE)
  }
  if (length(dims) == 1L) {
    dims <- as.integer(dims)
    if (is.na(dims) || dims < 1L) stop("dims must be positive.", call. = FALSE)
    if (dims > ncol(data)) {
      message("dims reduced to the ", ncol(data), " available dimensions.")
      dims <- ncol(data)
    }
    columns <- seq_len(dims)
  } else {
    columns <- as.integer(dims)
    if (anyNA(columns) || any(columns < 1L) || any(columns > ncol(data))) {
      stop("dims contains unavailable dimensions.", call. = FALSE)
    }
  }
  data[, columns, drop = FALSE]
}

.kodama_bioc_state <- function(object, name) {
  .kodama_require_namespace("S4Vectors", "Bioconductor")
  metadata <- S4Vectors::metadata(object)
  all_state <- metadata$KODAMA
  if (is.null(all_state) || !is.list(all_state)) all_state <- list()
  state <- all_state[[name]]
  if (is.null(state)) list() else state
}

.kodama_set_bioc_state <- function(object, name, state) {
  metadata <- S4Vectors::metadata(object)
  all_state <- metadata$KODAMA
  if (is.null(all_state) || !is.list(all_state)) all_state <- list()
  all_state[[name]] <- state
  metadata$KODAMA <- all_state
  S4Vectors::metadata(object) <- metadata
  object
}

.kodama_bioc_data <- function(object, reduction, dims) {
  .kodama_require_namespace("SingleCellExperiment", "SingleCellExperiment")
  available <- SingleCellExperiment::reducedDimNames(object)
  if (!reduction %in% available) {
    stop(
      "Reduced dimension '", reduction, "' was not found. Available values: ",
      paste(available, collapse = ", "), ".",
      call. = FALSE
    )
  }
  .kodama_select_dimensions(
    SingleCellExperiment::reducedDim(object, reduction), dims,
    paste0("Reduced dimension '", reduction, "'")
  )
}

.kodama_spatialexperiment_inputs <- function(object, sample.column) {
  .kodama_require_namespace("SpatialExperiment", "SpatialExperiment")
  spatial <- as.matrix(SpatialExperiment::spatialCoords(object))
  samples <- NULL
  if (!is.null(sample.column)) {
    .kodama_require_namespace("SummarizedExperiment", "SpatialExperiment")
    column_data <- SummarizedExperiment::colData(object)
    if (sample.column %in% colnames(column_data)) {
      samples <- as.vector(column_data[[sample.column]])
    }
  }
  list(spatial = spatial, samples = samples)
}

.kodama_seurat_data <- function(object, reduction, dims) {
  .kodama_require_namespace("SeuratObject", "Seurat")
  if (!inherits(object, "Seurat")) stop("object is not a Seurat object.", call. = FALSE)
  reductions <- names(methods::slot(object, "reductions"))
  if (!reduction %in% reductions) {
    stop("Reduction '", reduction, "' was not found in the Seurat object.", call. = FALSE)
  }
  .kodama_select_dimensions(
    SeuratObject::Embeddings(object, reduction = reduction), dims,
    paste0("Seurat reduction '", reduction, "'")
  )
}

.kodama_seurat_spatial <- function(object, cells) {
  .kodama_require_namespace("Seurat", "Seurat")
  images <- methods::slot(object, "images")
  if (!length(images)) {
    stop("use.spatial = TRUE requires at least one Seurat image.", call. = FALSE)
  }
  spatial <- matrix(NA_real_, nrow = length(cells), ncol = 2L,
                    dimnames = list(cells, c("x", "y")))
  samples <- rep(NA_character_, length(cells))
  for (image_name in names(images)) {
    coordinates <- as.data.frame(Seurat::GetTissueCoordinates(images[[image_name]]))
    coordinate_names <- intersect(c("x", "y"), colnames(coordinates))
    if (length(coordinate_names) < 2L) {
      numeric_columns <- which(vapply(coordinates, is.numeric, logical(1)))
      if (length(numeric_columns) < 2L) {
        stop("A Seurat image does not expose two numeric coordinates.", call. = FALSE)
      }
      coordinate_names <- colnames(coordinates)[numeric_columns[seq_len(2L)]]
    }
    if ("cell" %in% colnames(coordinates)) {
      cell_names <- as.character(coordinates$cell)
    } else {
      cell_names <- rownames(coordinates)
    }
    positions <- match(cell_names, cells)
    keep <- !is.na(positions)
    spatial[positions[keep], ] <- as.matrix(coordinates[keep, coordinate_names, drop = FALSE])
    samples[positions[keep]] <- image_name
  }
  if (anyNA(spatial) || anyNA(samples)) {
    stop(
      "Spatial coordinates were not found for every cell in the selected Seurat reduction.",
      call. = FALSE
    )
  }
  list(spatial = spatial, samples = samples)
}

.kodama_seurat_state <- function(object, name) {
  .kodama_require_namespace("SeuratObject", "Seurat")
  all_state <- SeuratObject::Misc(object, slot = "KODAMA")
  if (is.null(all_state) || !is.list(all_state)) all_state <- list()
  state <- all_state[[name]]
  if (is.null(state)) list() else state
}

.kodama_set_seurat_state <- function(object, name, state) {
  all_state <- SeuratObject::Misc(object, slot = "KODAMA")
  if (is.null(all_state) || !is.list(all_state)) all_state <- list()
  all_state[[name]] <- state
  suppressWarnings(
    SeuratObject::`Misc<-`(object, slot = "KODAMA", value = all_state)
  )
}

.kodama_set_seurat_visualization <- function(object, name, visualization, state) {
  key <- paste0(gsub("[^A-Za-z0-9]", "", name), "_")
  colnames(visualization) <- paste0(key, seq_len(ncol(visualization)))
  assay <- SeuratObject::DefaultAssay(object)
  object[[name]] <- SeuratObject::CreateDimReducObject(
    embeddings = visualization,
    key = key,
    assay = assay,
    misc = state
  )
  object
}

.kodama_giotto_data <- function(object, reduction, dims) {
  .kodama_require_namespace("Giotto", "Giotto")
  get_reduction <- .kodama_export("Giotto", "getDimReduction")
  data <- tryCatch(
    get_reduction(
      object,
      reduction = "cells",
      reduction_method = reduction,
      name = reduction,
      output = "matrix",
      set_defaults = TRUE
    ),
    error = function(error) NULL
  )
  if (is.null(data)) {
    data <- get_reduction(
      object,
      spat_unit = "cell",
      feat_type = "rna",
      reduction = "cells",
      reduction_method = "kodama",
      name = reduction,
      output = "matrix",
      set_defaults = FALSE
    )
  }
  .kodama_select_dimensions(data, dims, paste0("Giotto reduction '", reduction, "'"))
}

.kodama_giotto_spatial <- function(object, cells) {
  locations <- as.data.frame(.kodama_export("Giotto", "getSpatialLocations")(
    object, output = "data.table", copy_obj = TRUE,
    verbose = FALSE, set_defaults = TRUE
  ))
  cell_column <- intersect(c("cell_ID", "cell", "cell_id"), colnames(locations))
  location_cells <- if (length(cell_column)) {
    as.character(locations[[cell_column[[1L]]]])
  } else {
    rownames(locations)
  }
  numeric_columns <- which(vapply(locations, is.numeric, logical(1)))
  if (length(numeric_columns) < 2L) {
    stop("Giotto spatial locations do not contain two numeric coordinates.", call. = FALSE)
  }
  positions <- match(cells, location_cells)
  if (anyNA(positions)) {
    stop("Giotto spatial locations do not cover every selected cell.", call. = FALSE)
  }
  as.matrix(locations[positions, numeric_columns[seq_len(2L)], drop = FALSE])
}

.kodama_giotto_samples <- function(object, cells, sample.column) {
  if (is.null(sample.column)) return(NULL)
  metadata <- as.data.frame(.kodama_export("Giotto", "getCellMetadata")(
    object, output = "data.table", copy_obj = TRUE, set_defaults = TRUE
  ))
  if (!sample.column %in% colnames(metadata)) {
    stop(
      "Giotto cell metadata does not contain sample column '",
      sample.column, "'.",
      call. = FALSE
    )
  }
  cell_column <- intersect(c("cell_ID", "cell", "cell_id"), colnames(metadata))
  metadata_cells <- if (length(cell_column)) {
    as.character(metadata[[cell_column[[1L]]]])
  } else {
    rownames(metadata)
  }
  positions <- match(cells, metadata_cells)
  if (anyNA(positions)) {
    stop("Giotto cell metadata does not cover every selected cell.", call. = FALSE)
  }
  as.vector(metadata[[sample.column]][positions])
}

.kodama_feature_rows <- function(expression, cells, framework) {
  if (length(dim(expression)) != 2L ||
      (is.matrix(expression) && !is.numeric(expression))) {
    stop(framework, " expression data must be a numeric matrix.", call. = FALSE)
  }
  if (ncol(expression) != length(cells)) {
    stop(
      framework, " expression columns do not match its spatial observations.",
      call. = FALSE
    )
  }
  expression_cells <- colnames(expression)
  if (!is.null(expression_cells)) {
    positions <- match(cells, expression_cells)
    if (anyNA(positions)) {
      stop(
        framework, " expression data do not cover every spatial observation.",
        call. = FALSE
      )
    }
    expression <- expression[, positions, drop = FALSE]
  }
  if (inherits(expression, "Matrix")) {
    .kodama_require_namespace("Matrix", framework)
    Matrix::t(expression)
  } else {
    t(expression)
  }
}

.kodama_giotto_state <- function(object, name) {
  .kodama_require_namespace("Giotto", "Giotto")
  dim_object <- tryCatch(
    .kodama_export("Giotto", "getDimReduction")(
      object, spat_unit = "cell", feat_type = "rna", reduction = "cells",
      reduction_method = "kodama", name = name, output = "dimObj",
      set_defaults = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(dim_object)) return(list())
  state <- methods::slot(dim_object, "misc")
  if (is.null(state) || !is.list(state)) list() else state
}

.kodama_set_giotto_state <- function(object, name, coordinates, state) {
  .kodama_require_namespace("GiottoClass", "Giotto")
  dim_object <- .kodama_export("GiottoClass", "createDimObj", "Giotto")(
    coordinates = coordinates,
    name = name,
    spat_unit = "cell",
    feat_type = "rna",
    method = "KODAMA",
    reduction = "cells",
    provenance = NULL,
    misc = state,
    my_rownames = rownames(coordinates)
  )
  .kodama_export("Giotto", "setDimReduction")(
    object, x = dim_object, name = name, reduction = "cells",
    reduction_method = "kodama", verbose = FALSE
  )
}

.kodama_is_seurat_list <- function(object) {
  is.list(object) && length(object) > 0L &&
    all(vapply(object, inherits, logical(1), what = "Seurat"))
}

.kodama_subset_feature_rows <- function(expression, features, framework) {
  if (is.null(features)) return(expression)
  if (is.character(features)) {
    available <- rownames(expression)
    if (is.null(available) || anyNA(match(features, available))) {
      stop(framework, " does not contain every requested feature.", call. = FALSE)
    }
  }
  expression[features, , drop = FALSE]
}

.kodama_fit_expression_pca <- function(expression, cells, features,
                                       framework, ...) {
  expression <- .kodama_subset_feature_rows(expression, features, framework)
  feature_names <- rownames(expression)
  observations <- .kodama_feature_rows(expression, cells, framework)
  fit <- fastEmbedR::pca(observations, ...)
  fit$sdev <- fit$singular_values / sqrt(max(1, nrow(observations) - 1L))
  component_names <- paste0("PC", seq_len(ncol(fit$scores)))
  rownames(fit$scores) <- cells
  colnames(fit$scores) <- component_names
  rownames(fit$loadings) <- feature_names
  colnames(fit$loadings) <- component_names
  fit
}

.kodama_pca_misc <- function(fit) {
  fit[c(
    "singular_values", "sdev", "variance", "variance_explained",
    "cumulative_variance_explained", "total_variance", "center", "scale",
    "ncomp", "oversample", "power", "backend", "precision",
    "runtime_seconds"
  )]
}

.kodama_spatial_feature_metadata <- function(result) {
  rank <- integer(length(result$ranking))
  rank[result$ranking] <- seq_along(result$ranking)
  data.frame(
    KODAMA_spatial_score = result$score,
    KODAMA_spatial_p_value = result$p.value,
    KODAMA_spatial_adjusted_p_value = result$adjusted.p.value,
    KODAMA_spatial_rank = rank,
    row.names = names(result$score),
    check.names = FALSE
  )
}

.kodama_ranked_features <- function(rank, feature_names, nfeatures, framework) {
  if (is.null(nfeatures)) return(NULL)
  nfeatures <- suppressWarnings(as.integer(nfeatures))
  if (length(nfeatures) != 1L || is.na(nfeatures) || nfeatures < 1L) {
    stop("nfeatures must be a positive integer.", call. = FALSE)
  }
  if (is.null(rank) || length(rank) != length(feature_names) || anyNA(rank)) {
    stop(
      framework, " does not contain a SpatialFeatureSelection ranking. ",
      "Run SpatialFeatureSelection() first.", call. = FALSE
    )
  }
  feature_names[order(rank, method = "radix")][
    seq_len(min(nfeatures, length(feature_names)))
  ]
}

#' Fast native PCA for matrices and single-cell containers
#'
#' Uses [fastEmbedR::pca()] for the numerical PCA and stores its scores directly
#' in supported single-cell containers. Expression assays are transposed
#' internally because their rows are features and their columns are cells.
#'
#' @param object Numeric matrix, `SingleCellExperiment`, `SpatialExperiment`,
#'   Seurat object, Giotto object, or list of Seurat objects.
#' @param ... Arguments passed to [fastEmbedR::pca()].
#' @return A native PCA result for a matrix input, otherwise the updated
#'   container with a PCA dimensional reduction.
#' @export
RunFastPCA <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunFastPCA.Seurat, ...))
  }
  UseMethod("RunFastPCA")
}

#' @rdname RunFastPCA
#' @param features Optional feature names, indices, or logical selector.
#' @export
RunFastPCA.default <- function(object, features = NULL, ...) {
  if (!is.null(features)) object <- object[, features, drop = FALSE]
  fastEmbedR::pca(object, ...)
}

#' @rdname RunFastPCA
#' @param assay.type Assay to use. `NULL` prefers `logcounts`, then
#'   `normcounts`, then `counts`, and finally the first available assay.
#' @param nfeatures Number of top features from a preceding
#'   [SpatialFeatureSelection()] call. Cannot be combined with `features`.
#' @param ncomp Number of principal components to calculate and store.
#' @param reduction.name Name used to store PCA scores.
#' @export
RunFastPCA.SingleCellExperiment <- function(
    object, assay.type = NULL, features = NULL, nfeatures = NULL, ncomp = 50L,
    reduction.name = "PCA", ...) {
  .kodama_require_namespace("SummarizedExperiment", "SingleCellExperiment")
  assay_names <- SummarizedExperiment::assayNames(object)
  if (!length(assay_names)) {
    stop("SingleCellExperiment does not contain an expression assay.", call. = FALSE)
  }
  if (is.null(assay.type)) {
    preferred <- intersect(c("logcounts", "normcounts", "counts"), assay_names)
    assay.type <- if (length(preferred)) preferred[[1L]] else assay_names[[1L]]
  }
  if (!assay.type %in% assay_names) {
    stop(
      "Assay '", assay.type, "' was not found. Available values: ",
      paste(assay_names, collapse = ", "), ".", call. = FALSE
    )
  }
  if (!is.null(features) && !is.null(nfeatures)) {
    stop("Specify either features or nfeatures, not both.", call. = FALSE)
  }
  if (is.null(features) && !is.null(nfeatures)) {
    feature_data <- SummarizedExperiment::rowData(object)
    rank <- if ("KODAMA_spatial_rank" %in% colnames(feature_data)) {
      feature_data$KODAMA_spatial_rank
    } else NULL
    features <- .kodama_ranked_features(
      rank, rownames(object), nfeatures, "SingleCellExperiment"
    )
  }
  fit <- .kodama_fit_expression_pca(
    SummarizedExperiment::assay(object, assay.type), colnames(object),
    features, "SingleCellExperiment", ncomp = ncomp, ...
  )
  SingleCellExperiment::reducedDim(object, reduction.name) <- fit$scores
  object
}

#' @rdname RunFastPCA
#' @export
RunFastPCA.SpatialExperiment <- function(
    object, assay.type = NULL, features = NULL, nfeatures = NULL, ncomp = 50L,
    reduction.name = "PCA", ...) {
  RunFastPCA.SingleCellExperiment(
    object, assay.type = assay.type, features = features,
    nfeatures = nfeatures, ncomp = ncomp, reduction.name = reduction.name, ...
  )
}

#' @rdname RunFastPCA
#' @param assay Seurat assay containing the features.
#' @param layer Seurat assay layer, normally `"data"`.
#' @export
RunFastPCA.Seurat <- function(
    object, assay = NULL, layer = "data", features = NULL, nfeatures = NULL,
    ncomp = 50L,
    reduction.name = "pca", ...) {
  .kodama_require_namespace("SeuratObject", "Seurat")
  if (!inherits(object, "Seurat")) {
    stop("object is not a Seurat object.", call. = FALSE)
  }
  if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
  if (!assay %in% names(object)) {
    stop("Assay '", assay, "' was not found in the Seurat object.", call. = FALSE)
  }
  if (!is.null(features) && !is.null(nfeatures)) {
    stop("Specify either features or nfeatures, not both.", call. = FALSE)
  }
  if (is.null(features) && !is.null(nfeatures)) {
    feature_data <- object[[assay]][[]]
    rank <- if ("KODAMA_spatial_rank" %in% colnames(feature_data)) {
      feature_data$KODAMA_spatial_rank
    } else NULL
    features <- .kodama_ranked_features(
      rank, rownames(feature_data), nfeatures, "Seurat"
    )
  }
  expression <- SeuratObject::LayerData(object[[assay]], layer = layer)
  fit <- .kodama_fit_expression_pca(
    expression, colnames(expression), features, "Seurat",
    ncomp = ncomp, ...
  )
  key <- paste0(gsub("[^A-Za-z0-9]", "", toupper(reduction.name)), "_")
  object[[reduction.name]] <- SeuratObject::CreateDimReducObject(
    embeddings = fit$scores, loadings = fit$loadings, assay = assay,
    stdev = fit$sdev, key = key, misc = .kodama_pca_misc(fit)
  )
  object
}

#' @rdname RunFastPCA
#' @param values Giotto expression-values selection. `NULL` uses the object's
#'   default expression values.
#' @export
RunFastPCA.giotto <- function(
    object, values = NULL, features = NULL, nfeatures = NULL, ncomp = 50L,
    reduction.name = "pca", ...) {
  if (!is.null(features) && !is.null(nfeatures)) {
    stop("Specify either features or nfeatures, not both.", call. = FALSE)
  }
  if (is.null(features) && !is.null(nfeatures)) {
    feature_data <- as.data.frame(.kodama_export("Giotto", "getFeatureMetadata")(
      object, output = "data.table", copy_obj = TRUE, set_defaults = TRUE
    ))
    feature_column <- intersect(
      c("feat_ID", "gene_ID", "gene", "feature_ID"), colnames(feature_data)
    )
    feature_names <- if (length(feature_column)) {
      as.character(feature_data[[feature_column[[1L]]]])
    } else rownames(feature_data)
    rank <- if ("KODAMA_spatial_rank" %in% colnames(feature_data)) {
      feature_data$KODAMA_spatial_rank
    } else NULL
    features <- .kodama_ranked_features(rank, feature_names, nfeatures, "Giotto")
  }
  expression <- .kodama_export("Giotto", "getExpression")(
    object, values = values, output = "matrix", set_defaults = TRUE
  )
  cells <- colnames(expression)
  if (is.null(cells)) {
    stop("Giotto expression data must provide cell column names.", call. = FALSE)
  }
  fit <- .kodama_fit_expression_pca(
    expression, cells, features, "Giotto", ncomp = ncomp, ...
  )
  dim_object <- .kodama_export("GiottoClass", "createDimObj", "Giotto")(
    coordinates = fit$scores, name = reduction.name,
    spat_unit = "cell", feat_type = "rna", method = "PCA",
    reduction = "cells", provenance = NULL,
    misc = .kodama_pca_misc(fit), my_rownames = cells
  )
  .kodama_export("Giotto", "setDimReduction")(
    object, x = dim_object, name = reduction.name, reduction = "cells",
    reduction_method = "pca", verbose = FALSE
  )
}

.kodama_object_matrix <- function(data, graph = NULL, spatial = NULL,
                                  samples = NULL, arguments = list()) {
  if (!is.null(arguments$return.graph) &&
      !identical(arguments$return.graph, "handle")) {
    warning(
      "Object wrappers retain the corrected graph as a handle for visualization; ",
      "return.graph was set to 'handle'.",
      call. = FALSE
    )
  }
  arguments$return.graph <- "handle"
  do.call(
    KODAMA.matrix,
    c(
      list(data = data, graph = graph, spatial = spatial, samples = samples),
      arguments
    )
  )
}

#' Run KODAMA graph construction on matrices and single-cell containers
#'
#' The object methods extract an existing dimensional reduction, construct one
#' reusable KODAMA graph, and store it in the container. The graph does not
#' retain the original expression matrix.
#'
#' @param object Numeric matrix, `SingleCellExperiment`, `SpatialExperiment`,
#'   Seurat object, Giotto object, or list of Seurat objects.
#' @param ... Arguments passed to [KODAMA.graph()].
#' @return A graph for a matrix input, otherwise the updated container.
#' @export
RunKODAMAgraph <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunKODAMAgraph.Seurat, ...))
  }
  UseMethod("RunKODAMAgraph")
}

#' @rdname RunKODAMAgraph
#' @export
RunKODAMAgraph.default <- function(object, ...) KODAMA.graph(data = object, ...)

#' @rdname RunKODAMAgraph
#' @param reduction Name of the input dimensional reduction.
#' @param dims Number of leading dimensions, or an integer vector selecting
#'   dimensions.
#' @param graph.name Name under which the reusable graph is stored.
#' @export
RunKODAMAgraph.SingleCellExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA", ...) {
  data <- .kodama_bioc_data(object, reduction, dims)
  state <- .kodama_bioc_state(object, graph.name)
  state$graph <- KODAMA.graph(data = data, ...)
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAgraph
#' @param use.spatial Whether spatial coordinates should constrain a spatial
#'   container analysis.
#' @param sample.column Column in `colData` identifying slides or samples.
#' @export
RunKODAMAgraph.SpatialExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA",
    use.spatial = TRUE, sample.column = "sample_id", ...) {
  data <- .kodama_bioc_data(object, reduction, dims)
  spatial_input <- if (isTRUE(use.spatial)) {
    .kodama_spatialexperiment_inputs(object, sample.column)
  } else list(spatial = NULL, samples = NULL)
  state <- .kodama_bioc_state(object, graph.name)
  state$graph <- KODAMA.graph(
    data = data, spatial = spatial_input$spatial,
    samples = spatial_input$samples, ...
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAgraph
#' @param reduction.save Name used to store KODAMA state and the final
#'   visualization.
#' @export
RunKODAMAgraph.Seurat <- function(
    object, reduction = "pca", dims = 50L, use.spatial = TRUE,
    reduction.save = "KODAMA", ...) {
  data <- .kodama_seurat_data(object, reduction, dims)
  spatial_input <- if (isTRUE(use.spatial)) {
    .kodama_seurat_spatial(object, rownames(data))
  } else list(spatial = NULL, samples = NULL)
  state <- .kodama_seurat_state(object, reduction.save)
  state$graph <- KODAMA.graph(
    data = data, spatial = spatial_input$spatial,
    samples = spatial_input$samples, ...
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_seurat_state(object, reduction.save, state)
}

#' @rdname RunKODAMAgraph
#' @export
RunKODAMAgraph.giotto <- function(
    object, reduction = "pca", dims = 50L, use.spatial = TRUE,
    reduction.save = "KODAMA", ...) {
  data <- .kodama_giotto_data(object, reduction, dims)
  spatial <- if (isTRUE(use.spatial)) .kodama_giotto_spatial(object, rownames(data)) else NULL
  state <- .kodama_giotto_state(object, reduction.save)
  state$graph <- KODAMA.graph(data = data, spatial = spatial, ...)
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_giotto_state(
    object, reduction.save, data[, seq_len(min(2L, ncol(data))), drop = FALSE], state
  )
}

#' Run KODAMA optimization on matrices and single-cell containers
#'
#' Object methods reuse a graph previously stored by [RunKODAMAgraph()] unless
#' `reuse.graph = FALSE`. The optimization result is stored as KODAMA state;
#' the actual reduced dimension is created by [RunKODAMAvisualization()].
#'
#' @inheritParams RunKODAMAgraph
#' @param graph Optional graph supplied for a matrix input.
#' @param reuse.graph Reuse a graph stored in the object when available.
#' @param ... Arguments passed to [KODAMA.matrix()].
#' @return A KODAMA result for a matrix input, otherwise the updated container.
#' @export
RunKODAMAmatrix <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunKODAMAmatrix.Seurat, ...))
  }
  UseMethod("RunKODAMAmatrix")
}

#' @rdname RunKODAMAmatrix
#' @export
RunKODAMAmatrix.default <- function(object, graph = NULL, ...) {
  if (is.null(graph) && !is.null(extract_kodama_graph(object))) {
    graph <- object
    object <- NULL
  }
  .kodama_object_matrix(data = object, graph = graph, arguments = list(...))
}

#' @rdname RunKODAMAmatrix
#' @export
RunKODAMAmatrix.SingleCellExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA",
    reuse.graph = TRUE, ...) {
  data <- .kodama_bioc_data(object, reduction, dims)
  state <- .kodama_bioc_state(object, graph.name)
  graph <- if (isTRUE(reuse.graph)) state$graph else NULL
  state$matrix <- .kodama_object_matrix(
    data = data, graph = graph, arguments = list(...)
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAmatrix
#' @export
RunKODAMAmatrix.SpatialExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA",
    use.spatial = TRUE, sample.column = "sample_id", reuse.graph = TRUE, ...) {
  data <- .kodama_bioc_data(object, reduction, dims)
  spatial_input <- if (isTRUE(use.spatial)) {
    .kodama_spatialexperiment_inputs(object, sample.column)
  } else list(spatial = NULL, samples = NULL)
  state <- .kodama_bioc_state(object, graph.name)
  graph <- if (isTRUE(reuse.graph)) state$graph else NULL
  state$matrix <- .kodama_object_matrix(
    data = data, graph = graph, spatial = spatial_input$spatial,
    samples = spatial_input$samples, arguments = list(...)
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAmatrix
#' @export
RunKODAMAmatrix.Seurat <- function(
    object, reduction = "pca", dims = 50L, use.spatial = TRUE,
    reduction.save = "KODAMA", reuse.graph = TRUE, ...) {
  data <- .kodama_seurat_data(object, reduction, dims)
  spatial_input <- if (isTRUE(use.spatial)) {
    .kodama_seurat_spatial(object, rownames(data))
  } else list(spatial = NULL, samples = NULL)
  state <- .kodama_seurat_state(object, reduction.save)
  graph <- if (isTRUE(reuse.graph)) state$graph else NULL
  state$matrix <- .kodama_object_matrix(
    data = data, graph = graph, spatial = spatial_input$spatial,
    samples = spatial_input$samples, arguments = list(...)
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_seurat_state(object, reduction.save, state)
}

#' @rdname RunKODAMAmatrix
#' @export
RunKODAMAmatrix.giotto <- function(
    object, reduction = "pca", dims = 50L, use.spatial = TRUE,
    reduction.save = "KODAMA", reuse.graph = TRUE, ...) {
  data <- .kodama_giotto_data(object, reduction, dims)
  spatial <- if (isTRUE(use.spatial)) .kodama_giotto_spatial(object, rownames(data)) else NULL
  state <- .kodama_giotto_state(object, reduction.save)
  graph <- if (isTRUE(reuse.graph)) state$graph else NULL
  state$matrix <- .kodama_object_matrix(
    data = data, graph = graph, spatial = spatial, arguments = list(...)
  )
  state$input.reduction <- reduction
  state$input.dims <- dims
  .kodama_set_giotto_state(
    object, reduction.save, data[, seq_len(min(2L, ncol(data))), drop = FALSE], state
  )
}

#' Visualize KODAMA results stored in single-cell containers
#'
#' @inheritParams RunKODAMAmatrix
#' @param ... Arguments passed to [KODAMA.visualization()].
#' @return A matrix for a direct KODAMA input, otherwise the updated container
#'   with a genuine KODAMA reduced dimension.
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' fit <- RunKODAMAvisualization(
#'   x, method = "UMAP", k = 5, n.epochs = 5, n.cores = 1
#' )
#' dim(fit)
#' @export
RunKODAMAvisualization <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunKODAMAvisualization.Seurat, ...))
  }
  UseMethod("RunKODAMAvisualization")
}

#' @rdname RunKODAMAvisualization
#' @export
RunKODAMAvisualization.default <- function(object, ...) {
  KODAMA.visualization(object, ...)
}

#' @rdname RunKODAMAvisualization
#' @export
RunKODAMAvisualization.SingleCellExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA",
    reduction.save = "KODAMA", ...) {
  state <- .kodama_bioc_state(object, graph.name)
  if (is.null(state$matrix)) {
    stop("RunKODAMAmatrix() must be called before visualization.", call. = FALSE)
  }
  data <- .kodama_bioc_data(object, reduction, dims)
  visualization <- KODAMA.visualization(state$matrix, raw.data = data, ...)
  rownames(visualization) <- rownames(data)
  SingleCellExperiment::reducedDim(object, reduction.save) <- visualization
  state$visualization <- visualization
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAvisualization
#' @export
RunKODAMAvisualization.SpatialExperiment <- function(
    object, reduction = "PCA", dims = 50L, graph.name = "KODAMA",
    reduction.save = "KODAMA", ...) {
  RunKODAMAvisualization.SingleCellExperiment(
    object, reduction = reduction, dims = dims, graph.name = graph.name,
    reduction.save = reduction.save, ...
  )
}

#' @rdname RunKODAMAvisualization
#' @export
RunKODAMAvisualization.Seurat <- function(
    object, reduction = "pca", dims = 50L, reduction.save = "KODAMA", ...) {
  state <- .kodama_seurat_state(object, reduction.save)
  if (is.null(state$matrix)) {
    stop("RunKODAMAmatrix() must be called before visualization.", call. = FALSE)
  }
  data <- .kodama_seurat_data(object, reduction, dims)
  visualization <- KODAMA.visualization(state$matrix, raw.data = data, ...)
  rownames(visualization) <- rownames(data)
  state$visualization <- visualization
  object <- .kodama_set_seurat_state(object, reduction.save, state)
  .kodama_set_seurat_visualization(object, reduction.save, visualization, state)
}

#' @rdname RunKODAMAvisualization
#' @export
RunKODAMAvisualization.giotto <- function(
    object, reduction = "pca", dims = 50L, reduction.save = "KODAMA", ...) {
  state <- .kodama_giotto_state(object, reduction.save)
  if (is.null(state$matrix)) {
    stop("RunKODAMAmatrix() must be called before visualization.", call. = FALSE)
  }
  data <- .kodama_giotto_data(object, reduction, dims)
  visualization <- KODAMA.visualization(state$matrix, raw.data = data, ...)
  rownames(visualization) <- rownames(data)
  state$visualization <- visualization
  .kodama_set_giotto_state(object, reduction.save, visualization, state)
}

#' Cluster KODAMA reductions in matrices and single-cell containers
#'
#' This object-aware wrapper extracts a stored dimensional reduction, delegates
#' graph construction and Louvain, Leiden, or Walktrap clustering to
#' [KODAMA.clustering()], and writes the resulting membership back to the
#' container. The clustering algorithms themselves are implemented only by
#' [fastEmbedR::graph_cluster()].
#'
#' @param object Numeric matrix, KODAMA result, `SingleCellExperiment`,
#'   `SpatialExperiment`, Seurat object, Giotto object, or list of Seurat
#'   objects.
#' @param ... Arguments passed to [KODAMA.clustering()].
#' @return A `fastEmbedR_graph_cluster` result for direct matrix or graph input;
#'   otherwise the updated container with cluster membership stored in its
#'   observation metadata.
#' @export
RunKODAMAclustering <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunKODAMAclustering.Seurat, ...))
  }
  UseMethod("RunKODAMAclustering")
}

#' @rdname RunKODAMAclustering
#' @param graph Optional precomputed KODAMA, KNN, or fastEmbedR graph for a
#'   direct matrix call.
#' @export
RunKODAMAclustering.default <- function(object, graph = NULL, ...) {
  KODAMA.clustering(if (is.null(graph)) object else graph, ...)
}

#' @rdname RunKODAMAclustering
#' @param reduction Name of the dimensional reduction to cluster.
#' @param dims Number of leading dimensions, or an integer vector selecting
#'   dimensions.
#' @param cluster.name Observation-metadata column used to store membership.
#' @param graph.name Name of the KODAMA state entry used to store complete
#'   clustering diagnostics.
#' @export
RunKODAMAclustering.SingleCellExperiment <- function(
    object, reduction = "KODAMA", dims = 2L,
    cluster.name = "KODAMA_clusters", graph.name = "KODAMA", ...) {
  data <- .kodama_bioc_data(object, reduction, dims)
  fit <- KODAMA.clustering(data, ...)
  membership <- factor(fit$membership)
  names(membership) <- rownames(data)
  column_data <- SummarizedExperiment::colData(object)
  column_data[[cluster.name]] <- membership
  SummarizedExperiment::colData(object) <- column_data
  state <- .kodama_bioc_state(object, graph.name)
  if (is.null(state$clustering) || !is.list(state$clustering)) {
    state$clustering <- list()
  }
  state$clustering[[cluster.name]] <- fit
  .kodama_set_bioc_state(object, graph.name, state)
}

#' @rdname RunKODAMAclustering
#' @export
RunKODAMAclustering.SpatialExperiment <- function(
    object, reduction = "KODAMA", dims = 2L,
    cluster.name = "KODAMA_clusters", graph.name = "KODAMA", ...) {
  RunKODAMAclustering.SingleCellExperiment(
    object,
    reduction = reduction,
    dims = dims,
    cluster.name = cluster.name,
    graph.name = graph.name,
    ...
  )
}

#' @rdname RunKODAMAclustering
#' @param reduction.save Name of the Seurat or Giotto KODAMA state entry.
#' @export
RunKODAMAclustering.Seurat <- function(
    object, reduction = "KODAMA", dims = 2L,
    cluster.name = "KODAMA_clusters", reduction.save = "KODAMA", ...) {
  data <- .kodama_seurat_data(object, reduction, dims)
  fit <- KODAMA.clustering(data, ...)
  membership <- factor(fit$membership)
  names(membership) <- rownames(data)
  object <- SeuratObject::AddMetaData(
    object, metadata = membership, col.name = cluster.name
  )
  state <- .kodama_seurat_state(object, reduction.save)
  if (is.null(state$clustering) || !is.list(state$clustering)) {
    state$clustering <- list()
  }
  state$clustering[[cluster.name]] <- fit
  .kodama_set_seurat_state(object, reduction.save, state)
}

#' @rdname RunKODAMAclustering
#' @export
RunKODAMAclustering.giotto <- function(
    object, reduction = "KODAMA", dims = 2L,
    cluster.name = "KODAMA_clusters", reduction.save = "KODAMA", ...) {
  data <- .kodama_giotto_data(object, reduction, dims)
  fit <- KODAMA.clustering(data, ...)
  membership <- factor(fit$membership)
  names(membership) <- rownames(data)
  .kodama_export("Giotto", "addCellMetadata")(
    object,
    new_metadata = membership,
    vector_name = cluster.name
  )
}

#' Spatial feature selection for matrices and spatial containers
#'
#' Extracts expression values, coordinates, and slide identities from supported
#' containers, then calls the multicore CPU [spatial_feature_selection()]
#' implementation. Expression values are passed without normalization or
#' transformation.
#'
#' @param object Numeric matrix, SpatialExperiment, Seurat object, Giotto
#'   object, or list of Seurat objects.
#' @param ... Additional arguments passed to [spatial_feature_selection()].
#' @return A `kodama_spatial_features` result for a matrix input. Container
#'   methods return the updated object with statistics in feature metadata.
#' @aliases SpatialFeatureSelection
#' @export
RunSpatialFeatureSelection <- function(object, ...) {
  if (.kodama_is_seurat_list(object)) {
    return(lapply(object, RunSpatialFeatureSelection.Seurat, ...))
  }
  UseMethod("RunSpatialFeatureSelection")
}

#' @rdname RunSpatialFeatureSelection
#' @param spatial Coordinate matrix for a matrix input.
#' @param samples Optional slide or sample identifier for each observation.
#' @export
RunSpatialFeatureSelection.default <- function(
    object, spatial, samples = NULL, ...) {
  spatial_feature_selection(
    data = object, spatial = spatial, samples = samples, ...
  )
}

#' @rdname RunSpatialFeatureSelection
#' @param assay.type Name of the SpatialExperiment assay to screen.
#' @param sample.column Column in colData identifying slides or samples.
#' @export
RunSpatialFeatureSelection.SpatialExperiment <- function(
    object, assay.type = "logcounts", sample.column = "sample_id",
    samples = NULL, ...) {
  .kodama_require_namespace("SummarizedExperiment", "SpatialExperiment")
  assay_names <- SummarizedExperiment::assayNames(object)
  if (!assay.type %in% assay_names) {
    stop(
      "Assay '", assay.type, "' was not found. Available values: ",
      paste(assay_names, collapse = ", "), ".",
      call. = FALSE
    )
  }
  cells <- colnames(object)
  data <- .kodama_feature_rows(
    SummarizedExperiment::assay(object, assay.type),
    cells,
    "SpatialExperiment"
  )
  spatial_input <- .kodama_spatialexperiment_inputs(object, sample.column)
  if (!is.null(sample.column) && is.null(spatial_input$samples)) {
    stop(
      "SpatialExperiment colData does not contain sample column '",
      sample.column, "'.",
      call. = FALSE
    )
  }
  if (is.null(samples)) samples <- spatial_input$samples
  result <- spatial_feature_selection(
    data = data, spatial = spatial_input$spatial, samples = samples, ...
  )
  feature_metadata <- .kodama_spatial_feature_metadata(result)
  positions <- match(rownames(object), rownames(feature_metadata))
  if (anyNA(positions)) {
    stop("Spatial feature results do not match SpatialExperiment rows.", call. = FALSE)
  }
  row_data <- SummarizedExperiment::rowData(object)
  for (column in colnames(feature_metadata)) {
    row_data[[column]] <- feature_metadata[[column]][positions]
  }
  SummarizedExperiment::rowData(object) <- row_data
  metadata <- S4Vectors::metadata(object)
  if (is.null(metadata$KODAMA) || !is.list(metadata$KODAMA)) {
    metadata$KODAMA <- list()
  }
  metadata$KODAMA$SpatialFeatureSelection <- result
  S4Vectors::metadata(object) <- metadata
  object
}

#' @rdname RunSpatialFeatureSelection
#' @param assay Seurat assay containing the features to screen.
#' @param layer Seurat assay layer, normally "data".
#' @export
RunSpatialFeatureSelection.Seurat <- function(
    object, assay = NULL, layer = "data", samples = NULL, ...) {
  .kodama_require_namespace("SeuratObject", "Seurat")
  if (!inherits(object, "Seurat")) {
    stop("object is not a Seurat object.", call. = FALSE)
  }
  if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
  if (!assay %in% names(object)) {
    stop("Assay '", assay, "' was not found in the Seurat object.", call. = FALSE)
  }
  expression <- SeuratObject::LayerData(object[[assay]], layer = layer)
  cells <- colnames(expression)
  data <- .kodama_feature_rows(expression, cells, "Seurat")
  spatial_input <- .kodama_seurat_spatial(object, cells)
  if (is.null(samples)) samples <- spatial_input$samples
  result <- spatial_feature_selection(
    data = data, spatial = spatial_input$spatial, samples = samples, ...
  )
  feature_metadata <- .kodama_spatial_feature_metadata(result)
  assay_features <- rownames(object[[assay]])
  feature_metadata <- feature_metadata[assay_features, , drop = FALSE]
  object[[assay]] <- SeuratObject::AddMetaData(
    object[[assay]], metadata = feature_metadata
  )
  state <- SeuratObject::Misc(object, slot = "KODAMA")
  if (is.null(state) || !is.list(state)) state <- list()
  state$SpatialFeatureSelection <- result
  suppressWarnings(
    SeuratObject::`Misc<-`(object, slot = "KODAMA", value = state)
  )
}

#' @rdname RunSpatialFeatureSelection
#' @param values Giotto expression-values selection. NULL uses the object's
#'   default expression values.
#' @export
RunSpatialFeatureSelection.giotto <- function(
    object, values = NULL, sample.column = NULL, samples = NULL, ...) {
  expression <- .kodama_export("Giotto", "getExpression")(
    object, values = values, output = "matrix", set_defaults = TRUE
  )
  cells <- colnames(expression)
  if (is.null(cells)) {
    stop("Giotto expression data must provide cell column names.", call. = FALSE)
  }
  data <- .kodama_feature_rows(expression, cells, "Giotto")
  spatial <- .kodama_giotto_spatial(object, cells)
  if (is.null(samples)) {
    samples <- .kodama_giotto_samples(object, cells, sample.column)
  }
  result <- spatial_feature_selection(
    data = data, spatial = spatial, samples = samples, ...
  )
  feature_metadata <- .kodama_spatial_feature_metadata(result)
  feature_metadata$feat_ID <- rownames(feature_metadata)
  object <- .kodama_export("Giotto", "addFeatMetadata")(
    object, new_metadata = feature_metadata, by_column = TRUE,
    column_feat_ID = "feat_ID"
  )
  attr(object, "KODAMA_spatial_features") <- result
  object
}

#' @rdname RunSpatialFeatureSelection
#' @export
SpatialFeatureSelection <- RunSpatialFeatureSelection
