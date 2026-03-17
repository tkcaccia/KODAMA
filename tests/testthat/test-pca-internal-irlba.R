test_that("pca runs without loading irlba package namespace", {
  pre_loaded <- "irlba" %in% loadedNamespaces()
  x <- matrix(rnorm(120), nrow = 30, ncol = 4)
  out <- pca(x, nv = 3)
  post_loaded <- "irlba" %in% loadedNamespaces()

  expect_s3_class(out, "prcomp")
  expect_equal(post_loaded, pre_loaded)
})

test_that("pca output dimensions are correct", {
  x <- matrix(rnorm(200), nrow = 25, ncol = 8)
  out <- pca(x, nv = 5)

  expect_equal(dim(out$x), c(25, 5))
  expect_equal(dim(out$rotation), c(8, 5))
  expect_equal(length(out$sdev), 5)
})

test_that("pca scores are consistent with truncated SVD up to sign", {
  set.seed(123)
  x <- matrix(rnorm(180), nrow = 30, ncol = 6)
  k <- 4
  out <- pca(x, nv = k)

  sv <- svd(x, nu = k, nv = k)
  ref_scores <- sweep(sv$u, 2, sv$d[seq_len(k)], "*")

  for (j in seq_len(k)) {
    cc <- suppressWarnings(cor(out$x[, j], ref_scores[, j]))
    expect_gt(abs(cc), 0.98)
  }
})
