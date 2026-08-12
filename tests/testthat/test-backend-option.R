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
})
