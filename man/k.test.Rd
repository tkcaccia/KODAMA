\name{k.test}
\alias{k.test}
\title{K-Test of Statistical Association}
\description{
This function performs a permutation test using PLS to assess association between the KODAMA output and any additional related parameters such as clinical metadata.
}
\usage{k.test(data, labels, n = 100)}
\arguments{
  \item{data}{a matrix.}
  \item{labels}{a classification vector.}
  \item{n}{number of iterations of the permutation test.}
}

\value{
The p-value of the test.
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
 kk=KODAMA.matrix(data,FUN="KNN",f.par=2)
 kkplot=KODAMA.visualization(kk,"t-SNE")
 k1=k.test(kkplot,labels)
 print(k1)
 k2=k.test(kkplot,sample(labels))
 print(k2)
}
}
\keyword{k.test}
