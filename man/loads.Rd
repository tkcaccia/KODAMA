\name{loads}
\alias{loads}
\title{Variable Ranking}
\description{
This function can be used to extract the variable ranking when KODAMA is performed with the PLS-DA classifier.
}
\usage{
loads(model,method=c("loadings","kruskal.test"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{output of KODAMA.}
  \item{method}{method to be used. Choices are "\code{loadings}" and "\code{kruskal.test}".}
}

\value{
The function returns a vector of values indicating the "importance" of each variable. If "\code{method="loadings"} the average of the loading of the first component of PLS models based on the cross-validated accuracy maximized vector is computed. If "\code{method="kruskal.test"} the average of minus logarithm of p-value of Kruskal-Wallis Rank Sum test is computed. 
}
\author{Stefano Cacciatore}

\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/doi/10.1073/pnas.1220873111}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/33/4/621/2667156}{Link}
}

\seealso{\code{\link{KODAMA.matrix}},\code{\link{KODAMA.visualization}}}
\examples{
\donttest{
 data(iris)
 data=iris[,-5]
 labels=iris[,5]
 kk=KODAMA.matrix(data,FUN="PLS-DA")
 loads(kk)
}
}



