# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

test_that("PLS-LDA CV handles a zero-rank supervised fold", {
  x <- matrix(1, nrow = 40, ncol = 3)
  labels <- c(rep(10L, 30), rep(20L, 10))

  fit <- PLSLDACV(
    x,
    labels,
    folds = 5,
    ncomp = 3,
    backend = "cpu"
  )

  expect_length(fit$predicted, nrow(x))
  expect_true(is.finite(fit$accuracy))
  expect_identical(fit$selected_components, 1L)
})
