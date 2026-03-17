\name{KODAMA.matrix}
\alias{KODAMA.matrix}
\title{Knowledge Discovery by Accuracy Maximization}
\description{
Run KODAMA on a numeric data matrix and return the optimized label runs and
nearest-neighbor structure used by \code{\link{KODAMA.visualization}}.
}
\usage{
KODAMA.matrix(
  data,
  spatial = NULL,
  samples = NULL,
  M = 100,
  Tcycle = 20,
  ncomp = min(c(50, ncol(data))),
  W = NULL,
  metrics = "euclidean",
  constrain = NULL,
  fix = NULL,
  landmarks = 10000,
  splitting = ifelse(nrow(data) < 40000, 100, 300),
  spatial.resolution = 0.3,
  n.cores = 1,
  ancestry = FALSE,
  seed = 1234,
  ...
)
}
\arguments{
  \item{data}{Numeric matrix where rows are samples and columns are variables.}
  \item{spatial}{Optional numeric matrix of spatial coordinates with \code{nrow(spatial) == nrow(data)}.}
  \item{samples}{Optional sample identifier vector used to separate multiple spatial samples on a shared coordinate axis.}
  \item{M}{Number of independent KODAMA optimization runs.}
  \item{Tcycle}{Number of optimization cycles for each run.}
  \item{ncomp}{Number of PLS components.}
  \item{W}{Optional starting labels for semi-supervised initialization.}
  \item{metrics}{Distance metric passed to \code{Rnanoflann::nn}.}
  \item{constrain}{Optional grouping constraint vector; entries with the same value are forced to share labels within each run.}
  \item{fix}{Optional logical vector indicating which entries in \code{W} are fixed during optimization.}
  \item{landmarks}{Number of landmark clusters used in each run.}
  \item{splitting}{Number of clusters used for initialization when \code{W} is \code{NULL}.}
  \item{spatial.resolution}{Fraction of landmarks used to define spatial constraint clusters.}
  \item{n.cores}{Number of worker processes. On Unix-like systems forked workers are used; on Windows PSOCK workers are used.}
  \item{ancestry}{Logical; if \code{TRUE}, ancestry-aware spatial processing is used.}
  \item{seed}{Random seed.}
  \item{...}{Ignored legacy arguments. Passing \code{FUN} is deprecated and has no effect.}
}
\details{
The function runs \code{M} independent KODAMA optimizations and builds a
KODAMA-weighted nearest-neighbor structure. Progress bars are shown for both
the optimization stage and dissimilarity update stage.

The PLS backend is selected automatically from the current number of classes:
\code{"plssvd"} (fast mode) when \code{ncomp} is smaller than the number of
classes, otherwise \code{"simpls"}.

When \code{n.cores > 1}, Unix-like systems use fork-based parallelism, which
typically reduces memory duplication through copy-on-write when worker code
treats \code{data} as read-only. On Windows, socket workers are used and the
input matrix is copied to workers by design.
}
\value{
A list with:
\item{acc}{Numeric vector of length \code{M} with final run accuracies.}
\item{v}{Numeric matrix (\code{M x Tcycle}) of accuracy trajectories.}
\item{res}{Numeric matrix (\code{M x nrow(data)}) with optimized labels from each run.}
\item{knn_Rnanoflann}{List containing \code{indices}, \code{distances}, and \code{neighbors}.}
\item{data}{Input data matrix.}
\item{res_constrain}{Numeric matrix (\code{M x nrow(data)}) with effective constraints used in each run.}
\item{n.cores}{Number of cores used by \code{KODAMA.matrix}. This value is reused by \code{\link{KODAMA.visualization}} when visualization config sets \code{define.n.cores = FALSE}.}
}
\author{Stefano Cacciatore and Leonardo Tenori}
\seealso{\code{\link{KODAMA.visualization}}}
\examples{
\donttest{
data(iris)
data_mat <- iris[, -5]
kk <- KODAMA.matrix(data_mat, ncomp = 2, M = 10, n.cores = 1)
embedding <- KODAMA.visualization(kk, "t-SNE")
plot(embedding, col = as.numeric(iris[, 5]), cex = 2)
}
}
