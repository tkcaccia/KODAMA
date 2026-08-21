test_that("KODAMA backend precedence is explicit, option, environment, CPU", {
  old_global_option <- getOption("backend", NULL)
  old_option <- getOption("KODAMA.backend", NULL)
  old_global_env <- Sys.getenv("BACKEND", unset = NA_character_)
  old_env <- Sys.getenv("KODAMA_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_global_option)
    options(KODAMA.backend = old_option)
    if (is.na(old_global_env)) Sys.unsetenv("BACKEND") else Sys.setenv(BACKEND = old_global_env)
    if (is.na(old_env)) Sys.unsetenv("KODAMA_BACKEND") else Sys.setenv(KODAMA_BACKEND = old_env)
  }, add = TRUE)

  options(backend = NULL, KODAMA.backend = NULL)
  Sys.unsetenv(c("BACKEND", "KODAMA_BACKEND"))
  expect_identical(KODAMA_backend(), "cpu")
  Sys.setenv(BACKEND = "metal")
  expect_identical(KODAMA_backend(), "metal")
  options(backend = "cuda", KODAMA.backend = "cpu")
  expect_identical(KODAMA_backend(), "cuda")
  expect_identical(KODAMA:::kodama_resolve_backend("cpu"), "cpu")
  options(backend = NULL, KODAMA.backend = "metal")
  Sys.unsetenv("BACKEND")
  expect_identical(KODAMA_backend(), "metal")
  expect_error(KODAMA:::kodama_resolve_backend("auto"), "must be one of")
})

test_that("backend-capable KODAMA functions use NULL defaults", {
  functions <- list(
    KNNCV, PLSLDACV, CoreKNN, CorePLSLDA, KODAMA.matrix,
    KODAMA.graph, kodama_pca, KODAMA.visualization,
    normalization, scaling, passing.message
  )
  expect_true(all(vapply(functions, function(fn) is.null(formals(fn)$backend), logical(1))))
  expect_null(formals(KODAMA.clustering)$graph.backend)
  expect_null(formals(KODAMA.clustering)$backend)
})

test_that("pipeline core count uses explicit, option, environment, default precedence", {
  old_option <- getOption("n.cores", NULL)
  old_environment <- Sys.getenv("N_CORES", unset = NA_character_)
  on.exit({
    options(n.cores = old_option)
    if (is.na(old_environment)) {
      Sys.unsetenv("N_CORES")
    } else {
      Sys.setenv(N_CORES = old_environment)
    }
  }, add = TRUE)

  options(n.cores = NULL)
  Sys.unsetenv("N_CORES")
  expect_identical(KODAMA:::kodama_resolve_n_cores(NULL, 3L), 3L)
  Sys.setenv(N_CORES = "5")
  expect_identical(KODAMA:::kodama_resolve_n_cores(NULL, 3L), 5L)
  options(n.cores = 7L)
  expect_identical(KODAMA:::kodama_resolve_n_cores(NULL, 3L), 7L)
  expect_identical(KODAMA:::kodama_resolve_n_cores(2L, 3L), 2L)
  expect_error(KODAMA:::kodama_resolve_n_cores(0L, 3L), "greater than zero")
  expect_identical(
    KODAMA:::kodama_resolve_n_cores(0L, 3L, allow.zero = TRUE), 0L
  )

  pipeline <- list(KODAMA.matrix, KODAMA.graph, kodama_pca, KODAMA.visualization)
  expect_true(all(vapply(
    pipeline, function(fn) is.null(formals(fn)$n.cores), logical(1)
  )))

  options(n.cores = 2L)
  set.seed(11L)
  spatial <- matrix(runif(40), 20L, 2L)
  data <- cbind(spatial[, 1L], matrix(rnorm(20L * 3L), 20L, 3L))
  selection <- SpatialFeatureSelection(data, spatial)
  expect_identical(selection$n.cores, 2L)
  expect_null(formals(spatial_feature_selection)$n.cores)
})
