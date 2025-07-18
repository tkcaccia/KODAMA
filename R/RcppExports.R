# This file was generated by Rcpp::compileAttributes
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

floyd <- function(data) {
    data[is.na(data)] <- .Machine$double.xmax    
    data[is.infinite(data) & data > 0] <- .Machine$double.xmax
    data = .Call('KODAMA_floyd', PACKAGE = 'KODAMA', data)
    data[data == .Machine$double.xmax] <- NA
    data
}


transformy <- function(y) {
    .Call('KODAMA_transformy', PACKAGE = 'KODAMA', y)
}

PLSDACV_fastpls <- function(x, cl, constrain, k) {
    .Call('KODAMA_PLSDACV_fastpls', PACKAGE = 'KODAMA', x, cl, constrain, k)
}

PLSDACV_simpls <- function(x, cl, constrain, k) {
    .Call('KODAMA_PLSDACV_simpls', PACKAGE = 'KODAMA', x, cl, constrain, k)
}

corecpp <- function(x, xTdata, clbest, Tcycle, FUN, f.par.pls, Xconstrain, fix, shake, proj) {
    .Call('KODAMA_corecpp', PACKAGE = 'KODAMA', x, xTdata, clbest, Tcycle, FUN, f.par.pls, Xconstrain, fix, shake, proj)
}


