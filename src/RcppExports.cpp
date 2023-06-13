// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393



#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// floyd
arma::mat floyd(arma::mat data);
RcppExport SEXP KODAMA_floyd(SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    __result = Rcpp::wrap(floyd(data));
    return __result;
END_RCPP
}
// knn_kodama_c
arma::imat knn_kodama_c(arma::mat Xtrain, arma::ivec Ytrain, arma::mat Xtest, int k, int scaling);
RcppExport SEXP KODAMA_knn_kodama_c(SEXP XtrainSEXP, SEXP YtrainSEXP, SEXP XtestSEXP, SEXP kSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type Ytrain(YtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
  Rcpp::traits::input_parameter< int >::type k(kSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(knn_kodama_c(Xtrain, Ytrain, Xtest, k,scaling));
  return __result;
  END_RCPP
}
// knn_kodama_r
arma::mat knn_kodama_r(arma::mat Xtrain, arma::vec Ytrain, arma::mat Xtest, int k,int scaling);
RcppExport SEXP KODAMA_knn_kodama_r(SEXP XtrainSEXP, SEXP YtrainSEXP, SEXP XtestSEXP, SEXP kSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type Ytrain(YtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
  Rcpp::traits::input_parameter< int >::type k(kSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(knn_kodama_r(Xtrain, Ytrain, Xtest, k,scaling));
  return __result;
  END_RCPP
}
// KNNCV
arma::ivec KNNCV(arma::mat x, arma::ivec cl, arma::ivec constrain, int k);
RcppExport SEXP KODAMA_KNNCV(SEXP xSEXP, SEXP clSEXP, SEXP constrainSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type cl(clSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    __result = Rcpp::wrap(KNNCV(x, cl, constrain, k));
    return __result;
END_RCPP
}
// transformy
arma::mat transformy(arma::ivec y);
RcppExport SEXP KODAMA_transformy(SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::ivec >::type y(ySEXP);
    __result = Rcpp::wrap(transformy(y));
    return __result;
END_RCPP
}
// PLSDACV
arma::ivec PLSDACV(arma::mat x, arma::ivec cl, arma::ivec constrain, int k);
RcppExport SEXP KODAMA_PLSDACV(SEXP xSEXP, SEXP clSEXP, SEXP constrainSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type cl(clSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    __result = Rcpp::wrap(PLSDACV(x, cl, constrain, k));
    return __result;
END_RCPP
}
// pls_kodama
List pls_kodama(arma::mat Xtrain, arma::mat Ytrain, arma::mat Xtest, int ncomp,int scaling);
RcppExport SEXP KODAMA_pls_kodama(SEXP XtrainSEXP, SEXP YtrainSEXP, SEXP XtestSEXP, SEXP ncompSEXP, SEXP scalingSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Ytrain(YtrainSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
    Rcpp::traits::input_parameter< int >::type ncomp(ncompSEXP);
    Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
    __result = Rcpp::wrap(pls_kodama(Xtrain, Ytrain, Xtest, ncomp, scaling));
    return __result;
END_RCPP
}
// unic
int unic(arma::mat x);
RcppExport SEXP KODAMA_unic(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    __result = Rcpp::wrap(unic(x));
    return __result;
END_RCPP
}
// RQ
double RQ(arma::vec yData,arma::vec yPred);
RcppExport SEXP KODAMA_RQ(SEXP yDataSEXP,SEXP yPredSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type yData(yDataSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type yPred(yPredSEXP);
  __result = Rcpp::wrap(RQ(yData,yPred));
  return __result;
  END_RCPP
}
// optim_pls_cv
List optim_pls_cv(arma::mat x, arma::mat clmatrix, arma::ivec constrain, int ncomp,int scaling);
RcppExport SEXP KODAMA_optim_pls_cv(SEXP xSEXP, SEXP clmatrixSEXP, SEXP constrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type clmatrix(clmatrixSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(optim_pls_cv(x, clmatrix, constrain, ncomp,scaling));
  return __result;
  END_RCPP
}
// optim_knn_r_cv
List optim_knn_r_cv(arma::mat x, arma::vec clmatrix, arma::ivec constrain, int ncomp,int scaling);
RcppExport SEXP KODAMA_optim_knn_r_cv(SEXP xSEXP, SEXP clmatrixSEXP, SEXP constrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type clmatrix(clmatrixSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(optim_knn_r_cv(x, clmatrix, constrain, ncomp,scaling));
  return __result;
  END_RCPP
}
// optim_knn_c_cv
List optim_knn_c_cv(arma::mat x, arma::ivec clmatrix, arma::ivec constrain, int ncomp,int scaling);
RcppExport SEXP KODAMA_optim_knn_c_cv(SEXP xSEXP, SEXP clmatrixSEXP, SEXP constrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type clmatrix(clmatrixSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(optim_knn_c_cv(x, clmatrix, constrain, ncomp,scaling));
  return __result;
  END_RCPP
}
// double_pls_cv
List double_pls_cv(arma::mat x, arma::mat y, arma::ivec constrain, int type, int verbose, int compmax, int opt,int scaling);
RcppExport SEXP KODAMA_double_pls_cv(SEXP xSEXP, SEXP ySEXP, SEXP constrainSEXP, SEXP typeSEXP, SEXP verboseSEXP, SEXP compmaxSEXP, SEXP optSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type type(typeSEXP);
  Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
  Rcpp::traits::input_parameter< int >::type compmax(compmaxSEXP);
  Rcpp::traits::input_parameter< int >::type opt(optSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(double_pls_cv(x, y, constrain, type, verbose, compmax,opt,scaling));
  return __result;
  END_RCPP
}
// double_knn_cv
List double_knn_cv(arma::mat x, arma::vec yy, arma::ivec constrain, int type, int verbose, int compmax,int opt, int scaling);
RcppExport SEXP KODAMA_double_knn_cv(SEXP xSEXP, SEXP yySEXP, SEXP constrainSEXP, SEXP typeSEXP, SEXP verboseSEXP, SEXP compmaxSEXP, SEXP optSEXP, SEXP scalingSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type yy(yySEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type type(typeSEXP);
  Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
  Rcpp::traits::input_parameter< int >::type compmax(compmaxSEXP);
  Rcpp::traits::input_parameter< int >::type opt(optSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  __result = Rcpp::wrap(double_knn_cv(x, yy, constrain, type, verbose, compmax,opt,scaling));
  return __result;
  END_RCPP
}
// corecpp
List corecpp(arma::mat x, arma::mat xTdata, arma::ivec clbest, const int Tcycle, int FUN, int fpar, arma::ivec constrain, NumericVector fix, bool shake, int proj, arma::mat xNeighbors);
RcppExport SEXP KODAMA_corecpp(SEXP xSEXP, SEXP xTdataSEXP, SEXP clbestSEXP, SEXP TcycleSEXP, SEXP FUNSEXP, SEXP fparSEXP, SEXP constrainSEXP, SEXP fixSEXP, SEXP shakeSEXP, SEXP projSEXP, SEXP xNeighborsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type xTdata(xTdataSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type clbest(clbestSEXP);
    Rcpp::traits::input_parameter< const int >::type Tcycle(TcycleSEXP);
    Rcpp::traits::input_parameter< int >::type FUN(FUNSEXP);
    Rcpp::traits::input_parameter< int >::type fpar(fparSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type fix(fixSEXP);
    Rcpp::traits::input_parameter< bool >::type shake(shakeSEXP);
    Rcpp::traits::input_parameter< int >::type proj(projSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type xNeighbors(xNeighborsSEXP);
    __result = Rcpp::wrap(corecpp(x, xTdata, clbest, Tcycle, FUN, fpar, constrain, fix, shake, proj,xNeighbors));
    return __result;
END_RCPP
}
// knn_Armadillo
List knn_Armadillo(arma::mat Xtrain,arma::mat Xtest,int k);
RcppExport SEXP KODAMA_knn_Armadillo(SEXP XtrainSEXP, SEXP XtestSEXP, SEXP kSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
  Rcpp::traits::input_parameter< int >::type k(kSEXP);
  __result = Rcpp::wrap(knn_Armadillo(Xtrain, Xtest, k));
  return __result;
  END_RCPP
}