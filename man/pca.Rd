\name{pca}
\alias{pca}
\title{Truncated Principal Components Analysis}
\description{
Performs PCA using KODAMA's internal vendored IRLBA backend (with a small-matrix
\code{svd()} fallback) and returns a \code{prcomp}-compatible object.
}
\usage{
pca(x, nv = min(50L, ncol(x)), ...)
}
\arguments{
  \item{x}{A numeric matrix of data.}
  \item{nv}{Number of principal components to compute.}
  \item{...}{Currently unused, kept for backward compatibility.}
}
\value{
The function returns a list with class \code{prcomp} containing:
\item{sdev}{standard deviations of the retained principal components.}
\item{rotation}{matrix of variable loadings (columns are retained components).}
\item{x}{scores matrix equivalent to \code{u \%*\% diag(d)} from truncated SVD.}
\item{center, scale}{set to \code{FALSE}; centering/scaling are expected upstream when needed.}
\item{txt}{percentage-of-variance labels for each retained component.}
}
\author{Stefano Cacciatore}
\references{
Baglama J, Reichel L. \cr
Augmented implicitly restarted Lanczos bidiagonalization methods. \cr
\emph{SIAM Journal on Scientific Computing} 2005;27(1):19-42.
}
\examples{
data(MetRef)
u <- MetRef$data
u <- u[, -which(colSums(u) == 0)]
u <- normalization(u)$newXtrain
u <- scaling(u)$newXtrain
class <- as.numeric(as.factor(MetRef$gender))
cc <- pca(u, nv = 5)
plot(cc$x, pch = 21, bg = class)
}
\keyword{pca}
