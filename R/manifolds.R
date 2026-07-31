# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and Leonardo Tenori
# SPDX-License-Identifier: MIT
# Compatibility port of tkcaccia/KODAMA R/datasets.R by its authors.

#' Generate Dini's surface
#'
#' Generates points distributed on a Dini surface using the construction from
#' the KODAMA R package.
#'
#' @param N Number of data points.
#' @return An `N` by 3 numeric matrix with columns `x`, `y`, and `z`.
#' @seealso [helicoid()], [spirals()], [swissroll()]
#' @export
dinisurface <- function(N = 1000) {
  u <- sort(stats::runif(N) * 4 * pi)
  v <- stats::runif(N)
  a <- 1
  b <- 0.2
  x <- a * cos(u) * sin(v)
  y <- a * sin(u) * sin(v)
  z <- a * (cos(v) + log(tan(v / 2))) + b * u
  cbind(x, y, z)
}

#' Generate a helicoid surface
#'
#' Generates points distributed on a helicoid using the construction from the
#' KODAMA R package.
#'
#' @inheritParams dinisurface
#' @return An `N` by 3 numeric matrix with columns `x`, `y`, and `z`.
#' @seealso [dinisurface()], [spirals()], [swissroll()]
#' @export
helicoid <- function(N = 1000) {
  a <- 1
  p <- sample(seq(1, -1, length.out = N))
  t <- seq(-pi, pi, length.out = N)
  x <- p * cos(a * t)
  y <- p * sin(a * t)
  z <- t
  cbind(x, y, z)
}

#' Generate spiral clusters
#'
#' Produces two-dimensional spiral clusters using the construction from the
#' KODAMA R package.
#'
#' @param n Integer vector giving the number of points in each spiral.
#' @param sd Numeric vector giving the radial noise standard deviation for each
#'   spiral.
#' @return A `sum(n)` by 2 numeric matrix with columns `x` and `y`.
#' @seealso [dinisurface()], [helicoid()], [swissroll()]
#' @export
spirals <- function(n = c(100, 100, 100), sd = c(0, 0, 0)) {
  clusters <- length(n)
  x <- NULL
  y <- NULL
  for (i in seq_len(clusters)) {
    t <- seq(1 / (4 * pi), 1, length.out = n[i])^0.5 * 2 * pi
    a <- stats::rnorm(n[i], sd = sd[i])
    x <- c(x, cos(t + (2 * pi * i) / clusters) * (t + a))
    y <- c(y, sin(t + (2 * pi * i) / clusters) * (t + a))
  }
  cbind(x, y)
}

#' Generate a Swiss roll
#'
#' Generates the three-dimensional Swiss roll used by the KODAMA R package.
#'
#' @inheritParams dinisurface
#' @return An `N` by 3 numeric matrix with columns `x`, `y`, and `z`.
#' @seealso [dinisurface()], [helicoid()], [spirals()]
#' @export
swissroll <- function(N = 1000) {
  tt <- sort((3 * pi / 2) * (1 + 2 * stats::runif(N)))
  height <- 21 * stats::runif(N)
  x <- tt * cos(tt)
  y <- height
  z <- tt * sin(tt)
  cbind(x, y, z)
}
