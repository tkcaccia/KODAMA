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
#' @export
normalization <- function(Xtrain, Xtest = NULL, method = "pqn", ref = NULL,
                          backend = "cpu", n.cores = 1L, gpu.device = 0L) {
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
#' @export
scaling <- function(Xtrain, Xtest = NULL, method = "autoscaling",
                    backend = "cpu", n.cores = 1L, gpu.device = 0L) {
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
