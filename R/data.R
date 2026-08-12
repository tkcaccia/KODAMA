#' Nuclear Magnetic Resonance spectra of urine samples
#'
#' A longitudinal metabolomics cohort containing 873 urine spectra from 22
#' healthy donors. Each spectrum contains 450 bins.
#'
#' @format A list with three elements:
#' \describe{
#'   \item{data}{Numeric matrix with 873 rows and 450 columns.}
#'   \item{gender}{Character vector containing donor gender.}
#'   \item{donor}{Character vector containing donor identifiers.}
#' }
#' @references Assfalg M, Bertini I, Colangiuli D, et al. (2008)
#' Evidence of different metabolic phenotypes in humans. PNAS 105:1420-1424.
#' @source Distributed with the original KODAMA R package.
"MetRef"

#' State of the Union data
#'
#' TF-IDF representation of 86 spoken State of the Union addresses from 1900
#' through 2014, using 834 retained words.
#'
#' @format A list with three elements:
#' \describe{
#'   \item{data}{Numeric TF-IDF matrix with 86 rows and 834 columns.}
#'   \item{year}{Numeric vector containing speech years.}
#'   \item{president}{Character vector containing president names.}
#' }
#' @source Distributed with the original KODAMA R package.
"USA"

#' Lymphoma gene-expression data
#'
#' Gene-expression profiles for diffuse large B-cell lymphoma, follicular
#' lymphoma, and B-cell chronic lymphocytic leukemia.
#'
#' @format A list with two elements:
#' \describe{
#'   \item{data}{Numeric matrix with 62 rows and 4,026 columns.}
#'   \item{class}{Factor with levels `B-CLL`, `DLBCL`, and `FL`.}
#' }
#' @references Alizadeh AA, Eisen MB, Davis RE, et al. (2000). Distinct types
#' of diffuse large B-cell lymphoma identified by gene expression profiling.
#' Nature 403:503-511.
#' @source Distributed with the original KODAMA R package.
"lymphoma"
