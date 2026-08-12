test_that("KODAMA backend precedence is explicit, option, environment, CPU", {
  old_option <- getOption("KODAMA.backend", NULL)
  old_env <- Sys.getenv("KODAMA_BACKEND", unset = NA_character_)
  on.exit({
    options(KODAMA.backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("KODAMA_BACKEND") else Sys.setenv(KODAMA_BACKEND = old_env)
  }, add = TRUE)

  options(KODAMA.backend = NULL)
  Sys.unsetenv("KODAMA_BACKEND")
  expect_identical(KODAMA_backend(), "cpu")
  Sys.setenv(KODAMA_BACKEND = "metal")
  expect_identical(KODAMA_backend(), "metal")
  options(KODAMA.backend = "cuda")
  expect_identical(KODAMA_backend(), "cuda")
  expect_identical(KODAMA:::kodama_resolve_backend("cpu"), "cpu")
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
