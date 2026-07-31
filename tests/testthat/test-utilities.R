# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and Leonardo Tenori
# SPDX-License-Identifier: MIT

test_that("normalization preserves the KODAMA API", {
  Xtrain <- matrix(c(2, 4, 8, 3, 9, 12, 5, 10, 20), nrow = 3, byrow = TRUE)
  Xtest <- matrix(c(4, 8, 16, 6, 12, 24), nrow = 2, byrow = TRUE)

  expect_named(normalization(Xtrain), c("newXtrain", "coeXtrain"))
  expect_named(
    normalization(Xtrain, Xtest),
    c("newXtrain", "coeXtrain", "newXtest", "coeXtest")
  )
  expect_equal(normalization(Xtrain, method = "none")$newXtrain, Xtrain)

  sum_expected <- Xtrain / apply(Xtrain, 1L, sum)
  median_expected <- Xtrain / apply(Xtrain, 1L, median)
  sqrt_expected <- Xtrain / apply(Xtrain, 1L, function(x) sqrt(sum(x^2)))
  expect_equal(normalization(Xtrain, method = "sum")$newXtrain, sum_expected, tolerance = 1e-6)
  expect_equal(
    normalization(Xtrain, method = "median")$newXtrain,
    median_expected, tolerance = 1e-6
  )
  expect_equal(normalization(Xtrain, method = "sqrt")$newXtrain, sqrt_expected, tolerance = 1e-6)

  pqn <- normalization(Xtrain, Xtest, method = "pqn")
  expect_equal(dim(pqn$newXtrain), dim(Xtrain))
  expect_equal(dim(pqn$newXtest), dim(Xtest))
  expect_true(all(is.finite(pqn$newXtrain)))
  expect_error(normalization(Xtrain, method = "bad"), "invalid normalization")
})

test_that("scaling reuses training statistics", {
  Xtrain <- matrix(c(1, 4, 2, 8, 4, 12, 8, 16), ncol = 2, byrow = TRUE)
  Xtest <- matrix(c(3, 6, 6, 14), ncol = 2, byrow = TRUE)

  expect_equal(scaling(Xtrain, method = "none")$newXtrain, Xtrain)
  centered <- scaling(Xtrain, Xtest, method = "centering")
  expect_equal(colMeans(centered$newXtrain), c(0, 0), tolerance = 1e-15)
  expect_equal(
    unclass(centered$newXtest),
    unclass(scale(Xtest, center = colMeans(Xtrain), scale = FALSE)), tolerance = 1e-6
  )

  autoscaled <- scaling(Xtrain, Xtest, method = "autoscaling")
  expect_equal(
    unclass(autoscaled$newXtrain),
    unclass(scale(Xtrain, center = TRUE, scale = TRUE)), tolerance = 1e-6
  )
  expect_true(all(is.finite(scaling(Xtrain, method = "rangescaling")$newXtrain)))
  expect_true(all(is.finite(scaling(Xtrain, method = "paretoscaling")$newXtrain)))
  expect_error(scaling(Xtrain, method = "bad"), "invalid scaling")
})

test_that("synthetic manifolds preserve KODAMA constructions", {
  set.seed(17)
  dini <- dinisurface(25)
  set.seed(17)
  u <- sort(runif(25) * 4 * pi)
  v <- runif(25)
  dini_expected <- cbind(
    x = cos(u) * sin(v),
    y = sin(u) * sin(v),
    z = cos(v) + log(tan(v / 2)) + 0.2 * u
  )
  expect_equal(dini, dini_expected)

  set.seed(18)
  helix <- helicoid(25)
  set.seed(18)
  p <- sample(seq(1, -1, length.out = 25))
  angle <- seq(-pi, pi, length.out = 25)
  expect_equal(
    helix,
    cbind(x = p * cos(angle), y = p * sin(angle), z = angle)
  )

  set.seed(19)
  expect_equal(dim(spirals(c(10, 12, 14), c(0.1, 0.2, 0.3))), c(36L, 2L))

  set.seed(20)
  roll <- swissroll(25)
  set.seed(20)
  tt <- sort((3 * pi / 2) * (1 + 2 * runif(25)))
  height <- 21 * runif(25)
  expect_equal(
    roll,
    cbind(x = tt * cos(tt), y = height, z = tt * sin(tt))
  )
})
