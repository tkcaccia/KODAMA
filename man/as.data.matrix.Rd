\name{as.data.matrix}
\alias{as.data.matrix}
\title{Convert Character Matrix to Numeric Matrix}
\description{
  Converts a matrix of character or factor entries to a numeric matrix, preserving row and column names.
}
\usage{
as.data.matrix(x)
}
\arguments{
  \item{x}{
    A matrix containing character or factor values that represent numeric entries.
  }
}
\value{
  A numeric matrix of the same dimensions as \code{x}, with the same row and column names.
}
\details{
  This function is useful when a matrix has been read as character or factor and needs to be transformed into numeric format for analysis. It preserves both the row and column names of the input matrix.
}
\author{
  Stefano Cacciatore
}
\examples{
# Create a character matrix
ma <- matrix(c("1", "2", "3", "4"), ncol = 2)

# Convert to numeric matrix
ma_numeric <- as.data.matrix(ma)

# Print result
ma_numeric
}
\keyword{data}
