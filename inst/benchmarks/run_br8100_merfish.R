#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT


suppressPackageStartupMessages(library(kodamaR))

get_env <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) value else default
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)][1]
  if (length(hit) == 0L || is.na(hit)) paths[1] else hit
}

load_spatial_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    ds <- get("dataset", envir = env)
    return(list(data = as.matrix(ds$data), spatial = as.matrix(ds$spatial), labels = as.factor(ds$labels)))
  }
  names <- ls(env)
  data_name <- intersect(c("pca.PM", "pca_Br8100", "data", "x"), names)[1]
  spatial_name <- intersect(c("xyz", "xy_Br8100", "spatial"), names)[1]
  labels_name <- intersect(c("tissue_segments", "labels_Br8100", "labels"), names)[1]
  if (any(is.na(c(data_name, spatial_name, labels_name)))) {
    stop("Cannot identify data/spatial/labels objects in ", path)
  }
  list(
    data = as.matrix(get(data_name, envir = env)),
    spatial = as.matrix(get(spatial_name, envir = env)),
    labels = as.factor(get(labels_name, envir = env))
  )
}

run_one <- function(name, ds, classifier, backend, M, Tcycle, landmarks, knn.k, ncomp) {
  message(sprintf("[%s] %s backend=%s M=%d Tcycle=%d", name, classifier, backend, M, Tcycle))
  t0 <- proc.time()[["elapsed"]]
  result <- kodama_matrix(
    ds$data,
    spatial = ds$spatial,
    M = M,
    Tcycle = Tcycle,
    ncomp = ncomp,
    landmarks = landmarks,
    knn.k = knn.k,
    classifier = classifier,
    backend = backend,
    progress = TRUE
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  classes <- apply(result$res, 1L, function(z) length(unique(z)))
  data.frame(
    dataset = name,
    classifier = classifier,
    backend = backend,
    samples = nrow(ds$data),
    variables = ncol(ds$data),
    M = M,
    Tcycle = Tcycle,
    landmarks = landmarks,
    knn_k = knn.k,
    ncomp = ncomp,
    runtime_seconds = elapsed,
    core_runtime_seconds = result$runtime_seconds,
    median_acc = median(result$acc),
    max_acc = max(result$acc),
    median_classes = median(classes),
    min_classes = min(classes),
    max_classes = max(classes)
  )
}

out_dir <- get_env("KODAMA_R_BENCH_OUT", file.path(getwd(), "kodama-r-benchmarks"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- list(
  MERFISH = get_env("MERFISH_RDATA", first_existing(c(
    file.path("data", "MERFISH.RData"),
    file.path("data", "MERFISH-input.RData")
  ))),
  Br8100 = get_env("BR8100_RDATA", first_existing(c(
    file.path("data", "DLFPC-Br8100.RData"),
    file.path("data", "DLFPC-Br8100-input.RData"),
    file.path("data", "Br8100.RData")
  )))
)

M <- as.integer(get_env("KODAMA_BENCH_M", "20"))
Tcycle <- as.integer(get_env("KODAMA_BENCH_TCYCLE", "20"))
landmarks <- as.integer(get_env("KODAMA_BENCH_LANDMARKS", "100000"))
knn.k <- as.integer(get_env("KODAMA_BENCH_KNN_K", "30"))
ncomp <- as.integer(get_env("KODAMA_BENCH_NCOMP", "50"))
backends <- strsplit(get_env("KODAMA_BENCH_BACKENDS", "cpu,cuda"), ",", fixed = TRUE)[[1]]

rows <- list()
for (dataset_name in names(datasets)) {
  path <- datasets[[dataset_name]]
  if (!file.exists(path)) {
    warning("Skipping missing dataset: ", path)
    next
  }
  ds <- load_spatial_dataset(path)
  for (backend in backends) {
    for (classifier in c("knn", "pls_lda")) {
      rows[[length(rows) + 1L]] <- run_one(dataset_name, ds, classifier, backend, M, Tcycle, landmarks, knn.k, ncomp)
      write.csv(do.call(rbind, rows), file.path(out_dir, "br8100_merfish_summary.csv"), row.names = FALSE)
    }
  }
}

print(do.call(rbind, rows))
