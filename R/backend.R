# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

#' Configure the default KODAMA execution backend
#'
#' KODAMA uses an explicit `backend` argument when supplied. Otherwise it reads
#' `options(backend = ...)`, then `BACKEND`, and finally uses `"cpu"`.
#' Package-specific settings remain supported as compatibility fallbacks.
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @return The active backend. Setting returns the previous value invisibly.
#' @examples
#' KODAMA_backend()
#' old <- KODAMA_backend("cpu")
#' options(backend = old)
#' @export
KODAMA_backend <- function(backend = NULL) {
  if (is.null(backend)) return(kodama_resolve_backend(NULL))
  backend <- kodama_validate_backend(backend, "backend")
  old <- getOption("backend", NULL)
  options(backend = backend)
  invisible(old)
}

kodama_validate_backend <- function(backend, argument = "backend") {
  backend <- tolower(as.character(backend))
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend) ||
      !backend %in% c("cpu", "cuda", "metal")) {
    stop("`", argument, "` must be one of \"cpu\", \"cuda\", or \"metal\".", call. = FALSE)
  }
  backend
}

kodama_resolve_backend <- function(backend = NULL, argument = "backend") {
  if (!is.null(backend) && length(backend) == 1L) {
    return(kodama_validate_backend(backend, argument))
  }
  option <- getOption("backend", NULL)
  if (!is.null(option)) return(kodama_validate_backend(option, "option backend"))
  legacy_option <- getOption("KODAMA.backend", NULL)
  if (!is.null(legacy_option)) return(kodama_validate_backend(legacy_option, "option KODAMA.backend"))
  environment <- Sys.getenv("BACKEND", unset = "")
  if (nzchar(environment)) return(kodama_validate_backend(environment, "BACKEND"))
  legacy_environment <- Sys.getenv("KODAMA_BACKEND", unset = "")
  if (nzchar(legacy_environment)) return(kodama_validate_backend(legacy_environment, "KODAMA_BACKEND"))
  "cpu"
}

kodama_resolve_n_cores <- function(n.cores = NULL, default = 1L,
                                   allow.zero = FALSE) {
  value <- n.cores
  source <- "n.cores"
  if (is.null(value)) {
    value <- getOption("n.cores", NULL)
    source <- "option n.cores"
  }
  if (is.null(value)) {
    environment <- Sys.getenv("N_CORES", unset = "")
    if (nzchar(environment)) {
      value <- environment
      source <- "N_CORES"
    }
  }
  if (is.null(value)) value <- default
  value <- suppressWarnings(as.integer(value))
  minimum <- if (isTRUE(allow.zero)) 0L else 1L
  if (length(value) != 1L || is.na(value) || value < minimum) {
    stop(
      "`", source, "` must be a single integer ",
      if (minimum == 0L) "greater than or equal to zero" else "greater than zero",
      ".", call. = FALSE
    )
  }
  value
}
