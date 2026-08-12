#' Configure the default KODAMA execution backend
#'
#' KODAMA uses an explicit `backend` argument when supplied. Otherwise it reads
#' `options(KODAMA.backend = ...)`, then `KODAMA_BACKEND`, and finally uses
#' `"cpu"`. This function gets or sets the package option.
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @return The active backend. Setting returns the previous value invisibly.
#' @examples
#' KODAMA_backend()
#' old <- KODAMA_backend("cpu")
#' options(KODAMA.backend = old)
#' @export
KODAMA_backend <- function(backend = NULL) {
  if (is.null(backend)) return(kodama_resolve_backend(NULL))
  backend <- kodama_validate_backend(backend, "backend")
  old <- getOption("KODAMA.backend", NULL)
  options(KODAMA.backend = backend)
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
  option <- getOption("KODAMA.backend", NULL)
  if (!is.null(option)) return(kodama_validate_backend(option, "option KODAMA.backend"))
  environment <- Sys.getenv("KODAMA_BACKEND", unset = "")
  if (nzchar(environment)) return(kodama_validate_backend(environment, "KODAMA_BACKEND"))
  "cpu"
}
