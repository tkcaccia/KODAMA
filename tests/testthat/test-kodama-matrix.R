test_that("KODAMA.matrix returns expected structure", {
  set.seed(42)
  x <- matrix(rnorm(60), nrow = 12, ncol = 5)

  out <- KODAMA.matrix(
    data = x,
    M = 4,
    Tcycle = 3,
    ncomp = 2,
    landmarks = 6,
    splitting = 4,
    n.cores = 1,
    seed = 101
  )

  expect_type(out, "list")
  expect_true(all(c("acc", "v", "res", "knn_Rnanoflann", "data", "res_constrain") %in% names(out)))
  expect_equal(length(out$acc), 4)
  expect_equal(dim(out$v), c(4, 3))
  expect_equal(dim(out$res), c(4, 12))
  expect_equal(dim(out$res_constrain), c(4, 12))
  expect_equal(dim(out$knn_Rnanoflann$indices), c(12, out$knn_Rnanoflann$neighbors))
  expect_equal(dim(out$knn_Rnanoflann$distances), c(12, out$knn_Rnanoflann$neighbors))
  expect_equal(out$n.cores, 1)
})

test_that("Sequential and parallel runs are numerically consistent", {
  skip_on_cran()
  skip_if(.Platform$OS.type == "windows")

  set.seed(123)
  x <- matrix(rnorm(80), nrow = 16, ncol = 5)

  out_seq <- KODAMA.matrix(
    data = x,
    M = 5,
    Tcycle = 4,
    ncomp = 2,
    landmarks = 8,
    splitting = 4,
    n.cores = 1,
    seed = 999
  )
  out_par <- KODAMA.matrix(
    data = x,
    M = 5,
    Tcycle = 4,
    ncomp = 2,
    landmarks = 8,
    splitting = 4,
    n.cores = 2,
    seed = 999
  )

  expect_equal(out_seq$res, out_par$res, tolerance = 1e-10)
  expect_equal(out_seq$res_constrain, out_par$res_constrain, tolerance = 1e-10)
  expect_equal(out_seq$knn_Rnanoflann$indices, out_par$knn_Rnanoflann$indices)
  expect_equal(out_seq$knn_Rnanoflann$distances, out_par$knn_Rnanoflann$distances, tolerance = 1e-8)
  expect_equal(out_seq$acc, out_par$acc, tolerance = 1e-10)
  expect_equal(out_seq$n.cores, 1)
  expect_equal(out_par$n.cores, 2)
})

test_that("Reference compatibility with KODAMAextra implementation", {
  skip_if_not_installed("KODAMAextra")
  skip("optional integration test; requires external parallel stack")

  set.seed(7)
  x <- matrix(rnorm(70), nrow = 14, ncol = 5)

  ref <- KODAMAextra::KODAMA.matrix.parallel(
    data = x,
    M = 4,
    Tcycle = 3,
    ncomp = 2,
    landmarks = 7,
    splitting = 4,
    n.cores = 1,
    seed = 321
  )
  out <- KODAMA.matrix(
    data = x,
    M = 4,
    Tcycle = 3,
    ncomp = 2,
    landmarks = 7,
    splitting = 4,
    n.cores = 1,
    seed = 321
  )

  expect_equal(out$res, ref$res, tolerance = 1e-10)
  expect_equal(out$res_constrain, ref$res_constrain, tolerance = 1e-10)
  expect_equal(out$knn_Rnanoflann$indices, ref$knn_Rnanoflann$indices)
  expect_equal(out$knn_Rnanoflann$distances, ref$knn_Rnanoflann$distances, tolerance = 1e-8)
})

test_that("Default visualization configs expose define.n.cores", {
  expect_true("define.n.cores" %in% names(config.tsne.default))
  expect_true("define.n.cores" %in% names(config.umap.default))
  expect_false(isTRUE(config.tsne.default$define.n.cores))
  expect_false(isTRUE(config.umap.default$define.n.cores))
})

test_that("PLS backend is no longer user-selectable from KODAMA.matrix", {
  expect_false("FUN" %in% names(formals(KODAMA.matrix)))

  set.seed(321)
  x <- matrix(rnorm(60), nrow = 12, ncol = 5)

  out_fast <- expect_message(
    KODAMA.matrix(
      data = x,
      M = 3,
      Tcycle = 3,
      ncomp = 2,
      landmarks = 6,
      splitting = 4,
      n.cores = 1,
      seed = 555,
      FUN = "fastpls"
    ),
    "`FUN` is deprecated and ignored",
    fixed = TRUE
  )
  out_simpls <- expect_message(
    KODAMA.matrix(
      data = x,
      M = 3,
      Tcycle = 3,
      ncomp = 2,
      landmarks = 6,
      splitting = 4,
      n.cores = 1,
      seed = 555,
      FUN = "simpls"
    ),
    "`FUN` is deprecated and ignored",
    fixed = TRUE
  )

  expect_equal(out_fast$res, out_simpls$res, tolerance = 1e-10)
  expect_equal(out_fast$acc, out_simpls$acc, tolerance = 1e-10)
})
