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
#' @return A list containing `newXtrain` and `coeXtrain`, and also `newXtest`
#'   and `coeXtest` when `Xtest` is supplied.
#' @references Dieterle F, Ross A, Schlotterbeck G, Senn H. Probabilistic
#'   quotient normalization as robust method to account for dilution of
#'   complex biological mixtures. *Analytical Chemistry* 78, 4281-4290
#'   (2006).
#' @seealso [scaling()]
#' @export
normalization <- function(Xtrain, Xtest = NULL, method = "pqn", ref = NULL) {
  methods <- c("pqn", "sum", "median", "sqrt", "none")
  method <- pmatch(method, methods)
  if (is.na(method)) stop("invalid normalization method")

  if (method == 1L) {
    rSXtrain <- rowSums(abs(Xtrain), na.rm = TRUE)
    Xtrain <- Xtrain / rSXtrain
    if (is.null(ref)) ref <- apply(Xtrain, 2L, stats::median, na.rm = TRUE)
    newXtrain <- t(t(Xtrain) / ref)
    coeXtrain <- apply(newXtrain, 1L, stats::median, na.rm = TRUE)
    if (!is.null(Xtest)) {
      rSXtest <- rowSums(abs(Xtest), na.rm = TRUE)
      Xtest <- Xtest / rSXtest
      newXtest <- t(t(Xtest) / ref)
      coeXtest <- apply(newXtest, 1L, stats::median, na.rm = TRUE)
      return(list(
        newXtrain = Xtrain / coeXtrain,
        coeXtrain = coeXtrain * rSXtrain,
        newXtest = Xtest / coeXtest,
        coeXtest = coeXtest * rSXtest
      ))
    }
    return(list(
      newXtrain = Xtrain / coeXtrain,
      coeXtrain = coeXtrain * rSXtrain
    ))
  }

  if (method == 2L) {
    w1 <- apply(Xtrain, 1L, sum)
    data_train <- Xtrain / w1
    if (!is.null(Xtest)) {
      w2 <- apply(Xtest, 1L, function(x) sum(abs(x), na.rm = TRUE))
      return(list(
        newXtrain = data_train,
        coeXtrain = w1,
        newXtest = Xtest / w2,
        coeXtest = w2
      ))
    }
    return(list(newXtrain = data_train, coeXtrain = w1))
  }

  if (method == 3L) {
    w1 <- apply(Xtrain, 1L, stats::median)
    data_train <- Xtrain / w1
    if (!is.null(Xtest)) {
      w2 <- apply(Xtest, 1L, function(x) stats::median(x, na.rm = TRUE))
      return(list(
        newXtrain = data_train,
        coeXtrain = w1,
        newXtest = Xtest / w2,
        coeXtest = w2
      ))
    }
    return(list(newXtrain = data_train, coeXtrain = w1))
  }

  if (method == 4L) {
    w1 <- apply(Xtrain, 1L, function(x) sqrt(sum(x^2)))
    data_train <- Xtrain / w1
    if (!is.null(Xtest)) {
      w2 <- apply(Xtest, 1L, function(x) sqrt(sum(x^2, na.rm = TRUE)))
      return(list(
        newXtrain = data_train,
        coeXtrain = w1,
        newXtest = Xtest / w2,
        coeXtest = w2
      ))
    }
    return(list(newXtrain = data_train, coeXtrain = w1))
  }

  w1 <- rep(1, nrow(Xtrain))
  if (!is.null(Xtest)) {
    w2 <- rep(1, nrow(Xtest))
    return(list(
      newXtrain = Xtrain,
      coeXtrain = w1,
      newXtest = Xtest,
      coeXtest = w2
    ))
  }
  list(newXtrain = Xtrain, coeXtrain = w1)
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
#' @return A list containing `newXtrain`, and also `newXtest` when `Xtest` is
#'   supplied.
#' @references van den Berg RA, Hoefsloot HCJ, Westerhuis JA, et al. Centering,
#'   scaling, and transformations: improving the biological information
#'   content of metabolomics data. *BMC Genomics* 7, 142 (2006).
#' @seealso [normalization()]
#' @export
scaling <- function(Xtrain, Xtest = NULL, method = "autoscaling") {
  methods <- c(
    "none", "centering", "autoscaling", "rangescaling", "paretoscaling"
  )
  method <- pmatch(method, methods)
  if (is.na(method)) stop("invalid scaling method")

  if (!is.null(Xtest)) {
    if (method == 2L) {
      Xtrain <- scale(Xtrain, center = TRUE, scale = FALSE)
      Xtest <- scale(
        Xtest, center = attr(Xtrain, "scaled:center"), scale = FALSE
      )
      return(list(newXtrain = Xtrain, newXtest = Xtest))
    }
    if (method == 3L) {
      Xtrain <- scale(Xtrain, center = TRUE, scale = TRUE)
      Xtest <- scale(
        Xtest,
        center = attr(Xtrain, "scaled:center"),
        scale = attr(Xtrain, "scaled:scale")
      )
    }
    if (method == 4L) {
      ran <- apply(Xtrain, 2L, function(x) max(x) - min(x))
      Xtrain <- scale(Xtrain, center = TRUE, scale = ran)
      Xtest <- scale(Xtest, center = attr(Xtrain, "scaled:center"), scale = ran)
    }
    if (method == 5L) {
      ssd <- sqrt(apply(Xtrain, 2L, stats::sd))
      Xtrain <- scale(Xtrain, center = TRUE, scale = ssd)
      Xtest <- scale(Xtest, center = attr(Xtrain, "scaled:center"), scale = ssd)
    }
    return(list(newXtrain = Xtrain, newXtest = Xtest))
  }

  if (method == 2L) {
    Xtrain <- scale(Xtrain, center = TRUE, scale = FALSE)
  }
  if (method == 3L) {
    Xtrain <- scale(Xtrain, center = TRUE, scale = TRUE)
  }
  if (method == 4L) {
    ran <- apply(Xtrain, 2L, function(x) max(x) - min(x))
    Xtrain <- scale(Xtrain, center = TRUE, scale = ran)
  }
  if (method == 5L) {
    ssd <- sqrt(apply(Xtrain, 2L, stats::sd))
    Xtrain <- scale(Xtrain, center = TRUE, scale = ssd)
  }
  list(newXtrain = Xtrain)
}
