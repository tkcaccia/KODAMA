\docType{data}
\name{config.tsne.default}
\alias{config.tsne.default}
\title{Default configuration for Rtsne}
\format{
  An object of class \code{config.tsne.default} of length 12.
}
\usage{
  config.tsne.default
}
\description{
  A list with parameters customizing a Rtsne embedding. Each component of the
  list is an effective argument for Rtsne_neighbors().
}
\details{
  dims: integer, Output dimensionality 
  
  perplexity: numeric, Perplexity parameter (should not be bigger than 3 * perplexity < nrow(X) - 1, see details for interpretation)
  
  theta: numeric, Speed/accuracy trade-off (increase for less accuracy), set to 0.0 for exact TSNE 
  
  max_iter: integer, Number of iterations 
  
  verbose: logical, Whether progress updates should be printed (default: global "verbose" option, or FALSE if that is not set)
  
  Y_init: matrix, Initial locations of the objects. If NULL, random initialization will be used (default: NULL). Note that when using this, the initial stage with exaggerated perplexity values and a larger momentum term will be skipped.
  
  momentum: numeric, Momentum used in the first part of the optimization 
  
  final_momentum: numeric, Momentum used in the final part of the optimization 
  
  eta: numeric, Learning rate 
  
  exaggeration_factor:
  
  num_threads: integer, Number of threads to use when using OpenMP, default is 1. Setting to 0 corresponds to detecting and using all available cores

  define.n.cores: logical. If FALSE (default), \code{\link{KODAMA.visualization}} overrides \code{num_threads} using \code{kk$n.cores} from \code{\link{KODAMA.matrix}} output.
}
\examples{
  # display all default settings
  config.tsne.default
  
  # create a new settings object with perplexity set to 100
  custom.settings = config.tsne.default
  custom.settings$perplexity = 100
  custom.settings
  
}
\keyword{datasets}
