# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and Leonardo Tenori
# SPDX-License-Identifier: MIT
# Compatibility port of tkcaccia/KODAMA R/preprocessing.R by its authors.

#' Normalize samples
#'
#' Compatibility implementation of the preprocessing utility exposed by the
#' KODAMA R package. Samples are stored in rows and variables in columns.
#'
#' @param Xtrain Numeric training matrix.
#' @param Xtest Optional numeric test matrix with the same variables as
#'   `Xtrain`.
#' @param method Normalization method. One of `"pqn"`, `"sum"`, `"median"`,
#'   `"sqrt"`, or `"none"`; an unambiguous partial match is accepted.
#' @param ref Optional reference spectrum for probabilistic quotient
#'   normalization. When omitted, the variable-wise training median is used.
#' @param backend Native backend: `"cpu"`, `"cuda"`, `"metal"`, or `"auto"`.
#' @param n.cores Number of CPU worker threads.
#' @param gpu.device CUDA device identifier; Metal currently uses device zero.
#' @return A list containing `newXtrain` and `coeXtrain`, and also `newXtest`
#'   and `coeXtest` when `Xtest` is supplied.
#' @references Dieterle F, Ross A, Schlotterbeck G, Senn H. Probabilistic
#'   quotient normalization as robust method to account for dilution of
#'   complex biological mixtures. *Analytical Chemistry* 78, 4281-4290
#'   (2006).
#' @seealso [scaling()]
#' @examples
#' x <- abs(as.matrix(iris[, 1:4])) + 1
#' fit <- normalization(x, method = "pqn")
#' dim(fit$newXtrain)
#' @export
normalization <- function(Xtrain, Xtest = NULL, method = "pqn", ref = NULL,
                          backend = NULL, n.cores = 1L, gpu.device = 0L) {
  backend <- kodama_resolve_backend(backend)
  methods <- c("pqn", "sum", "median", "sqrt", "none")
  method <- pmatch(method, methods)
  if (is.na(method)) stop("invalid normalization method")
  out <- kodama_normalization_cpp(
    Xtrain, Xtest, methods[[method]], ref, backend, n.cores, gpu.device
  )
  out[c("backend", "runtime_seconds", "precision", "reference")] <- NULL
  out
}

#' Scale variables
#'
#' Compatibility implementation of the scaling utility exposed by the KODAMA
#' R package. Training statistics are reused for an optional test matrix.
#'
#' @param Xtrain Numeric training matrix.
#' @param Xtest Optional numeric test matrix with the same variables as
#'   `Xtrain`.
#' @param method Scaling method. One of `"none"`, `"centering"`,
#'   `"autoscaling"`, `"rangescaling"`, or `"paretoscaling"`; an
#'   unambiguous partial match is accepted.
#' @param backend Native backend: `"cpu"`, `"cuda"`, `"metal"`, or `"auto"`.
#' @param n.cores Number of CPU worker threads.
#' @param gpu.device CUDA device identifier; Metal currently uses device zero.
#' @return A list containing `newXtrain`, and also `newXtest` when `Xtest` is
#'   supplied.
#' @references van den Berg RA, Hoefsloot HCJ, Westerhuis JA, et al. Centering,
#'   scaling, and transformations: improving the biological information
#'   content of metabolomics data. *BMC Genomics* 7, 142 (2006).
#' @seealso [normalization()]
#' @examples
#' fit <- scaling(as.matrix(iris[, 1:4]), method = "autoscaling")
#' dim(fit$newXtrain)
#' @export
scaling <- function(Xtrain, Xtest = NULL, method = "autoscaling",
                    backend = NULL, n.cores = 1L, gpu.device = 0L) {
  backend <- kodama_resolve_backend(backend)
  methods <- c(
    "none", "centering", "autoscaling", "rangescaling", "paretoscaling"
  )
  method <- pmatch(method, methods)
  if (is.na(method)) stop("invalid scaling method")

  out <- kodama_scaling_cpp(
    Xtrain, Xtest, methods[[method]], backend, n.cores, gpu.device
  )
  if (method != 1L) {
    attr(out$newXtrain, "scaled:center") <- out$center
    if (!is.null(out$newXtest)) attr(out$newXtest, "scaled:center") <- out$center
  }
  if (method %in% 3:5) {
    attr(out$newXtrain, "scaled:scale") <- out$scale
    if (!is.null(out$newXtest)) attr(out$newXtest, "scaled:scale") <- out$scale
  }
  out[c("backend", "runtime_seconds", "precision", "center", "scale")] <- NULL
  out
}

#' Spatial message-passing preprocessing
#'
#' Reproduces the self-inclusive, distance-weighted `passing.message`
#' preprocessing operation from KODAMAextra using the native exact 2D/3D grid
#' search. The weighted sum is intentionally not normalized by its total
#' weight.
#'
#' @param data Numeric expression matrix with observations in rows.
#' @param spatial Numeric matrix containing two or three coordinates per row.
#' @param number_knn Number of self-inclusive spatial nearest neighbors.
#' @param samples Optional sample, tissue, or slide vector. Neighbor search and
#'   distance scaling are performed independently for every distinct value.
#' @param backend Native backend: `"cpu"`, `"cuda"`, `"metal"`, or `"auto"`.
#' @param n.cores Number of CPU worker threads.
#' @param gpu.device CUDA device identifier; Metal currently uses device zero.
#' @return A numeric matrix with the dimensions and dimnames of `data`. Timing,
#'   backend, precision, and sample-group metadata are attached as attributes.
#' @export
passing.message <- function(data, spatial, number_knn = 15L, samples = NULL,
                            backend = NULL, n.cores = 4L,
                            gpu.device = 0L) {
  backend <- kodama_resolve_backend(backend)
  data <- as.matrix(data)
  spatial <- as.matrix(spatial)
  if (nrow(data) != nrow(spatial)) {
    stop("data and spatial must have the same number of rows.")
  }
  if (!is.null(samples)) {
    if (length(samples) != nrow(data) || anyNA(samples)) {
      stop("samples must contain one non-missing value per data row.")
    }
    samples <- as.integer(as.factor(samples))
  }
  result <- kodama_passing_message_cpp(
    data, spatial, as.integer(number_knn), samples, backend,
    as.integer(n.cores), as.integer(gpu.device)
  )
  output <- result$data
  dimnames(output) <- dimnames(data)
  attr(output, "backend") <- result$backend
  attr(output, "precision") <- result$precision
  attr(output, "sample_groups") <- result$sample_groups
  attr(output, "sample_max_distances") <- result$sample_max_distances
  attr(output, "timing") <- result[c(
    "graph_seconds", "aggregation_seconds", "runtime_seconds"
  )]
  output
}

#' Fast multi-slide spatial feature screening
#'
#' Independently screens variables with low-rank spatial covariance projections
#' spanning linear, smooth, and periodic coordinate bases. Basis-level evidence
#' is combined with the Cauchy rule; slides are processed independently. This
#' implementation was written independently and does not use SPARK-X code.
#'
#' @param data Numeric matrix with observations in rows and variables in columns.
#'   Sparse and delayed matrix-like inputs are converted to dense float32 work
#'   buffers in bounded feature batches rather than densified in full.
#' @param spatial Two- or three-column spatial-coordinate matrix.
#' @param samples Optional slide/sample vector. Statistics are calculated
#'   independently within each value.
#' @param n.cores Number of CPU threads. `NULL` uses
#'   `options(n.cores = ...)`, then `N_CORES`, and otherwise 4.
#' @param require.nonzero.each.sample Exclude variables that are identically
#'   zero in any slide, matching the filtering convention of `multi_SPARKX()`.
#' @return A list with rankings, scores, raw and BH-adjusted p-values,
#'   per-slide results, basis/statistic timings, resolved core count, and
#'   backend metadata.
#' @export
spatial_feature_selection <- function(
    data, spatial, samples = NULL,
    n.cores = NULL, require.nonzero.each.sample = TRUE) {
  n.cores <- kodama_resolve_n_cores(n.cores, default = 4L)
  data_dimensions <- dim(data)
  if (length(data_dimensions) != 2L ||
      (is.matrix(data) && !is.numeric(data)) ||
      anyNA(data_dimensions) || any(data_dimensions < 1L)) {
    stop("data must be a non-empty numeric matrix-like object.")
  }
  spatial <- as.matrix(spatial)
  if (!is.numeric(spatial) || length(dim(spatial)) != 2L ||
      nrow(spatial) != data_dimensions[[1L]]) {
    stop("spatial must be a numeric matrix with one row per data observation.")
  }
  sample_levels <- NULL
  sample_ids <- NULL
  if (!is.null(samples)) {
    if (length(samples) != data_dimensions[[1L]] || anyNA(samples)) {
      stop("samples must contain one non-missing value per data row.")
    }
    samples <- factor(samples)
    sample_levels <- levels(samples)
    sample_ids <- as.integer(samples)
  }
  run_dense <- function(x) {
    x <- as.matrix(x)
    if (!is.numeric(x)) {
      stop("data must contain numeric values.")
    }
    kodama_spatial_features_cpp(
      x, spatial, sample_ids, as.integer(n.cores),
      isTRUE(require.nonzero.each.sample)
    )
  }
  if (is.matrix(data)) {
    result <- run_dense(data)
    result$batches <- 1L
    result$input.storage <- "dense"
  } else {
    # R's dense representation uses eight bytes per value. Bound each
    # temporary conversion to approximately 64 MiB and retain feature order.
    target_bytes <- 64 * 1024^2
    columns_per_batch <- max(
      1L,
      min(data_dimensions[[2L]],
          floor(target_bytes / (8 * data_dimensions[[1L]])))
    )
    batches <- split(
      seq_len(data_dimensions[[2L]]),
      ceiling(seq_len(data_dimensions[[2L]]) / columns_per_batch)
    )
    partial <- lapply(batches, function(columns) {
      run_dense(as.matrix(data[, columns, drop = FALSE]))
    })
    result <- partial[[1L]]
    result$score <- unlist(lapply(partial, `[[`, "score"), use.names = FALSE)
    result$p.value <- unlist(
      lapply(partial, `[[`, "p.value"), use.names = FALSE
    )
    result$adjusted.p.value <- stats::p.adjust(result$p.value, method = "BH")
    result$ranking <- order(-result$score, seq_along(result$score), method = "radix")
    result$per.sample.score <- do.call(
      cbind, lapply(partial, `[[`, "per.sample.score")
    )
    result$per.sample.p.value <- do.call(
      cbind, lapply(partial, `[[`, "per.sample.p.value")
    )
    result$variables <- data_dimensions[[2L]]
    result$basis.seconds <- sum(vapply(
      partial, function(x) x$basis.seconds, numeric(1L)
    ))
    result$statistic.seconds <- sum(vapply(
      partial, function(x) x$statistic.seconds, numeric(1L)
    ))
    result$runtime.seconds <- sum(vapply(
      partial, function(x) x$runtime.seconds, numeric(1L)
    ))
    result$batches <- length(batches)
    result$input.storage <- "sparse_or_delayed"
  }
  variable_names <- colnames(data)
  if (!is.null(variable_names)) {
    names(result$score) <- variable_names
    names(result$p.value) <- variable_names
    names(result$adjusted.p.value) <- variable_names
    colnames(result$per.sample.score) <- variable_names
    colnames(result$per.sample.p.value) <- variable_names
    result$features <- variable_names[result$ranking]
  } else {
    result$features <- result$ranking
  }
  if (!is.null(sample_levels)) {
    rownames(result$per.sample.score) <- sample_levels
    rownames(result$per.sample.p.value) <- sample_levels
    result$sample.labels <- sample_levels
  }
  result$n.cores <- n.cores
  class(result) <- c("kodama_spatial_features", "list")
  result
}

#' @rdname spatial_feature_selection
#' @export
KODAMA.spatial.features <- spatial_feature_selection
