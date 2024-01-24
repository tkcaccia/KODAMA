#define R_NO_REMAP

#include <map>
#include <vector>
#include <iostream>
#include <fstream>

//#include <math.h>           // math routines
#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>


#include "ANN/ANN.h"        // ANN library header
#include "NN.h"             // ANN library header
#include <R.h>              // R header
#include "RcppArmadillo.h"  // RcppArmadillo library header


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


#ifdef _OPENMP
#include <omp.h>
#endif

// RAND_MAX (used in 'uniform_negative') is a constant defined in <cstdlib> ( see: http://www.cplusplus.com/reference/cstdlib/rand/ )
#include <cstdlib>







using namespace std;
using namespace Rcpp;



// [[Rcpp::export]]
arma::mat satlins(arma::mat x) {

  arma::uvec idx_greater = arma::find(x >= 1.0);

  arma::uvec idx_less = arma::find(x <= -1.0);

  if (idx_greater.n_elem > 0) {

    x(idx_greater).fill(1.0);
  }

  else {

    if (idx_less.n_elem > 0) {

      x(idx_less).fill(-1.0);
    }
  }

  return x;
}



// 'tribas' activation function
//

// [[Rcpp::export]]
arma::mat tribas(arma::mat x) {

  arma::uvec idx_condition = arma::find(x >= -1.0 && x <= 1.0);

  arma::uvec idx_NOT_condition = arma::find(x < -1.0 && x > 1.0);

  if (idx_condition.n_elem > 0) {

    for (unsigned int i = 0; i < idx_condition.n_elem; i++) {

      x(idx_condition(i)) = 1.0 - std::abs(x(idx_condition(i)));
    }
  }

  if (idx_NOT_condition.n_elem > 0) {

    x(idx_NOT_condition).fill(0.0);
  }

  return x;
}



// 'hardlim' activation function
//

// [[Rcpp::export]]
arma::mat hardlim(arma::mat x) {

  arma::uvec idx_greater = arma::find(x >= 0.0);

  arma::uvec idx_less = arma::find(x < 0.0);

  if (idx_greater.n_elem > 0) {

    x(idx_greater).fill(1.0);
  }

  if (idx_less.n_elem > 0) {

    x(idx_less).fill(0.0);
  }

  return x;
}



// 'hardlims' activation function
//

// [[Rcpp::export]]
arma::mat hardlims(arma::mat x) {

  arma::uvec idx_greater = arma::find(x >= 0.0);

  arma::uvec idx_less = arma::find(x < 0.0);

  if (idx_greater.n_elem > 0) {

    x(idx_greater).fill(1.0);
  }

  if (idx_less.n_elem > 0) {

    x(idx_less).fill(-1.0);
  }

  return x;
}



// 'relu' activation function
//[ I renamed 'poslin' to 'relu' because it's easier to remember I also added alpha for leaky-relu ]
//

// [[Rcpp::export]]
arma::mat relu(arma::mat x, double alpha = 0.0) {

  arma::uvec idx = arma::find(x < 0.0);

  if (idx.n_elem > 0) {

    if (alpha == 0.0) {

      x(idx).fill(0.0);
    }

    else {

      x(idx) *= alpha;
    }
  }

  return x;
}



// use base R's set.seed() in Rcpp for RNG
// http://thecoatlessprofessor.com/programming/set_rs_seed_in_rcpp_sequential_case
//

// [[Rcpp::export]]
void set_seed(int seed) {

  Rcpp::Environment base_env("package:base");

  Rcpp::Function set_seed_r = base_env["set.seed"];

  set_seed_r(seed);
}



// uniform negative [-1, 1] (see my commented out scipt below)
// see issue: https://github.com/mlampros/elmNNRcpp/issues/2
// slightly modified from https://stackoverflow.com/a/34757263/8302386
//

// [[Rcpp::export]]
arma::mat uniform_negative(int n_rows, int n_cols) {

  arma::mat res_neg = arma::randu(n_rows, n_cols);
  res_neg *= RAND_MAX;
  res_neg = -1.0 + res_neg / (RAND_MAX / 2.0);
  return res_neg;
}



// //-------------------------------------------------------------------------------------------------------  No effect, because I set the seed inside the 'elm_train_rcpp' function, see issue: https://github.com/mlampros/elmNNRcpp/issues/2  [ keep as a reference for negative-uniform-distribution ]
// // random-uniform-matrix generation [ I can adjust the cols, rows, min-value, max-value ]
// // [ see the Armadillo documentation for more details : http://arma.sourceforge.net/docs.html ]
// //
//
// // [[Rcpp::export]]
// arma::mat randomMatrix(int nCols, int nRows, double minValue, double maxValue) {
//
//   std::mt19937 engine;  // Mersenne twister random number engine
//
//   std::uniform_real_distribution<double> distr(minValue, maxValue);
//
//   arma::mat A(nRows, nCols);
//
//   A.imbue( [&]() { return distr(engine); } );
//
//   return A;
// }
//
//
// // uniform value generation for a vector
// // [ see the Armadillo documentation for more details : http://arma.sourceforge.net/docs.html ]
// //
//
// // [[Rcpp::export]]
// std::vector<double> uniform_vector(int iters, double minValue, double maxValue) {
//
//   std::mt19937 engine;  // Mersenne twister random number engine
//
//   std::uniform_real_distribution<double> distr(minValue, maxValue);
//
//   std::vector<double> res;
//
//   for (int i = 0; i < iters; i++) {
//
//     res.push_back(distr(engine));
//   }
//
//   return res;
// }
// //-------------------------------------------------------------------------------------------------------



// switch function for activation functions
//

// [[Rcpp::export]]
arma::mat activation_functions(arma::mat& tempH, std::string actfun, double alpha = 0.0) {

  arma::mat H;

  if (actfun == "sig") { H = 1.0 / (1.0 + arma::exp(-1.0 * tempH)); }

  else if (actfun == "sin") { H = arma::sin(tempH); }

  else if (actfun == "radbas") { H = arma::exp(-1.0 * (arma::pow(tempH, 2.0))); }

  else if (actfun == "hardlim") { H = hardlim(tempH); }

  else if (actfun == "hardlims")  { H = hardlims(tempH); }

  else if (actfun == "satlins") { H = satlins(tempH); }

  else if (actfun == "tansig") { H = (2.0 / ( 1 + arma::exp(-2.0 * tempH)) - 1.0); }

  else if (actfun == "tribas") { H = tribas(tempH); }

  else if (actfun == "relu") { H = relu(tempH, alpha); }

  else if (actfun == "purelin") { H = tempH; }

  else {

    std::string message = "ERROR: " + actfun + " is not a valid activation function.";

    Rcpp::stop(message);
  }

  return H;
}




// extreme learning machines train-function     [ init_weights can be either "normal_gaussian" ([0,1]), "uniform_positive" ([0,1]) or "uniform_negative" ([-1,1]) ]
//

// [[Rcpp::export]]
Rcpp::List elm_train_rcpp(arma::mat& x, 
                          arma::mat y, 
                          int nhid, 
                          std::string actfun, 
                          std::string init_weights = "normal_gaussian", 
                          bool bias = false,
                          double moorep_pseudoinv_tol = 0.01, 
                          double alpha = 0.0, 
                          int seed = 1, 
                          bool verbose = false) {


  if (nhid < 1) Rcpp::Rcout << "ERROR: number of hidden neurons must be >= 1" << std::endl;

  int transpose_ROWS = x.n_rows;

  int transpose_COLS = x.n_cols;

  arma::mat inpweight;

  arma::colvec biashid;

  set_seed(seed);

  if (verbose) Rcpp::Rcout << "Input weights will be initialized ..." << std::endl;

  if (init_weights == "normal_gaussian") {

    inpweight = arma::randn<arma::mat>(nhid, transpose_COLS);

    if (bias) {

      biashid = arma::randn<arma::colvec>(nhid);
    }
  }

  else if (init_weights == "uniform_positive") {

    inpweight = arma::randu<arma::mat>(nhid, transpose_COLS);

    if (bias) {

      biashid = arma::randu<arma::colvec>(nhid);
    }
  }

  else if (init_weights == "uniform_negative") {

    inpweight = uniform_negative(nhid, transpose_COLS);

    if (bias) {

      biashid = arma::conv_to< arma::colvec >::from(uniform_negative(nhid, 1));
    }
  }

  else {

    Rcpp::stop("Invalid type for the 'init_weights' parameter");
  }

  if (verbose) Rcpp::Rcout << "Dot product of input weights and data starts ..." << std::endl;

  x = inpweight * x.t();

  arma::mat biasMatrix;

  if (bias) {

    if (verbose) Rcpp::Rcout << "Bias will be added to the dot product ..." << std::endl;

    biasMatrix.set_size(biashid.size(), transpose_ROWS);

    for (int i = 0; i < transpose_ROWS; i++) {

      biasMatrix.col(i) = biashid;
    }

    x += biasMatrix;
  }

  if (verbose) Rcpp::Rcout << "'" + actfun + "' activation function will be utilized ..." << std::endl;

  x = activation_functions(x, actfun, alpha);

  x = x.t();

  if (verbose) Rcpp::Rcout << "The computation of the Moore-Pseudo-inverse starts ..." << std::endl;

  arma::mat outweight = arma::pinv(x, moorep_pseudoinv_tol, "dc" );

  if (outweight.n_cols != y.n_rows) { Rcpp::stop("The number of rows of the input data and the size of the response variable do not match!"); }

  outweight = outweight * y;

  arma::mat Y = x * outweight;

  arma::mat residuals = y - Y;

  if (verbose) Rcpp::Rcout << "The computation is finished!" << std::endl;

  return Rcpp::List::create(Rcpp::Named("inpweight") = inpweight, Rcpp::Named("biashid") = biashid, Rcpp::Named("outweight") = outweight, Rcpp::Named("actfun") = actfun,

                            Rcpp::Named("nhid") = nhid, Rcpp::Named("predictions") = Y, Rcpp::Named("fitted_values") = Y, Rcpp::Named("residuals") = residuals);
}



// normalize predictions to [0,1] range
//

// [[Rcpp::export]]
arma::mat norm_preds(arma::mat x) {

  x = arma::abs(x);

  arma::colvec sum_x = sum(x, 1);

  for (unsigned int i = 0; i < sum_x.n_elem; i++) {

    x.row(i) /= sum_x(i);
  }

  return x;
}




// prediction function for elmNNRcpp       [ in case of classification it returns probabilities ]
//

// [[Rcpp::export]]
arma::mat elm_predict_rcpp(Rcpp::List object, arma::mat &newdata, bool normalize = false) {

  arma::mat inpweight = Rcpp::as<arma::mat>(object[0]);

  arma::colvec biashid = Rcpp::as<arma::colvec>(object[1]);

  arma::mat outweight = Rcpp::as<arma::mat>(object[2]);

  std::string actfun = Rcpp::as<std::string>(object[3]);

  int transpose_COLS = newdata.n_rows;

  // arma::mat TV_P = newdata.t();

  arma::mat tmpHTest = inpweight * newdata.t();

  if (!biashid.empty()) {

    arma::mat biasMatrixTE(biashid.n_elem, transpose_COLS);      // transpose_COLS == TV_P.n_cols

    for (int i = 0; i < transpose_COLS; i++) {          // transpose_COLS == TV_P.n_cols

      biasMatrixTE.col(i) = biashid;
    }

    tmpHTest += biasMatrixTE;
  }

  arma::mat HTest = activation_functions(tmpHTest, actfun);

  arma::mat predictions = HTest.t() * outweight;

  if (normalize) {

    if (predictions.n_cols > 1) {

      predictions = norm_preds(predictions);                   // normalize each row to [0,1] as it's possible that the output probabilities include negative values
    }
  }

  return predictions;
}



// build one-hot labels from a numeric vector
// it expects the labels to be in the range between [0,n]
//

// [[Rcpp::export]]
arma::mat onehot_labels_rcpp(arma::rowvec x) {

  arma::rowvec unq = arma::unique(x);

  if (arma::min(unq) != 0) {

    Rcpp::stop("The minimum value for the unique labels (response variable) should be 0!");
  }

  arma::mat onehot_res(x.n_elem, unq.n_elem, arma::fill::zeros);

  for (unsigned int i = 0; i < x.n_elem; i++) {

    onehot_res(i, x(i)) = 1;
  }

  return onehot_res;
}





// This function performs a random selection of the elements of a vector "yy".
// The number of elements to select is defined by the variable "size".

IntegerVector samplewithoutreplace(IntegerVector yy,int size){
  IntegerVector xx(size);
  int rest=yy.size();
  int it;
  for(int ii=0;ii<size;ii++){
    it=unif_rand()*rest;
    xx[ii]=yy[it];
    yy.erase(it);
    rest--;
  }
  return xx;
}



arma::uvec which(LogicalVector x) {
  int a=std::accumulate(x.begin(),x.end(), 0.0);

  arma::uvec w(a);
  int counter=0;
  for(int i = 0; i < x.size(); i++) {
    if(x[i] == 1){
      w[counter]=i;
      counter++;
    }
  }
    return w;
}


arma::mat variance(arma::mat x) {
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::mat out(1,ncol);
  
  for (int j = 0; j < ncol; j++) {
    double mean = 0;
    double M2 = 0;
    int n=0;
    double delta, xx;
    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(0,j) = sqrt(M2/(n-1));
  }
  return out;
}


List scalecpp(arma::mat Xtrain,arma::mat Xtest,int type){
  arma::mat mX=mean(Xtrain,0);
  Xtrain.each_row()-=mX;
  Xtest.each_row()-=mX;
  arma::mat vX=variance(Xtrain); 
  if(type==2){
    Xtrain.each_row()/=vX;
    Xtest.each_row()/=vX;  
  }
  return List::create(
    Named("Xtrain") = Xtrain,
    Named("Xtest")   = Xtest,
    Named("mean")       = mX,
    Named("sd")       = vX
  ) ;
}



double accuracy(arma::ivec cl,arma::ivec cvpred){
  double acc=0;
  for(unsigned i=0;i<cl.size();i++){
    if(cl[i]==cvpred[i])
      acc++;
  }
  acc=acc/cl.size();
  return acc;
  
}




arma::imat knn_kodama(arma::mat Xtrain,arma::ivec Ytrain,arma::mat Xtest,int k) {
  arma::ivec cla=unique(Ytrain);
  int maxlabel=max(Ytrain);
  double* data = Xtrain.memptr();
  int *label=Ytrain.memptr();
  double* query = Xtest.memptr();
  int D=Xtrain.n_cols;
  int ND=Xtrain.n_rows;
  int NQ=Xtest.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::imat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  for(int j=0;j<NQ;j++){
    int *lab= new int[k];
    arma::ivec scale(maxlabel);
    scale.zeros();
    for(int i=0;i<k;i++){
      
      lab[i]=label[nn_index[j*k+i]-1];
      
      scale(lab[i]-1)=scale(lab[i]-1)+1;
      
      
      int most_common=-1;
      int value=0;
      for(int h=0;h<maxlabel;h++){
        if(scale(h)>value){
          value=scale(h);
          
          most_common=h;
        }
      }
      Ytest(j,i)=most_common+1;
    }
    delete [] lab;
    
    
  }
  
  delete [] nn_index;
  delete [] distances;
  return Ytest;
}




// [[Rcpp::export]]
List knn_Armadillo(arma::mat Xtrain,arma::mat Xtest,int k) {
  double* data = Xtrain.memptr();
  double* query = Xtest.memptr();
  int D=Xtrain.n_cols;
  int ND=Xtrain.n_rows;
  int NQ=Xtest.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::imat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  
  arma::mat distancesArmadillo(NQ,k);
  arma::mat nn_indexArmadillo(NQ,k);
  for(int j=0;j<NQ;j++){
    for(int i=0;i<k;i++){
      nn_indexArmadillo(j,i)=nn_index[j*k+i];
      distancesArmadillo(j,i)=distances[j*k+i];
    }
  }
  
  delete [] nn_index;
  delete [] distances;
  return List::create(
    Named("nn_index")   = nn_indexArmadillo,
    Named("distances")   = distancesArmadillo
  );
}






// [[Rcpp::export]]
arma::mat floyd(arma::mat data){
  int n=data.n_cols;
  int i,j,k;
  double temp;
  arma::mat A=data;
  for (i=0; i<n; i++)
    A(i,i) = 0;           
  for (k=0; k<n; k++){
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
        temp=A(i,k)+A(k,j);
        if (temp < A(i,j))
        {
          A(i,j) = temp;
          
        }
      }
    }
  }
      return A;
}



// [[Rcpp::export]]
arma::imat knn_kodama_c(arma::mat Xtrain,arma::ivec Ytrain,arma::mat Xtest,int k,int scaling) {
  List temp0=scalecpp(Xtrain,Xtest,scaling);
  arma::mat Xtrain1=temp0[0];
  arma::mat Xtest1=temp0[1];
  arma::ivec cla=unique(Ytrain);
  int maxlabel=max(Ytrain);
  double* data = Xtrain1.memptr();
  int *label=Ytrain.memptr();
  double* query = Xtest1.memptr();
  int D=Xtrain1.n_cols;
  int ND=Xtrain1.n_rows;
  int NQ=Xtest1.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::imat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  for(int j=0;j<NQ;j++){
    int *lab= new int[k];
    arma::ivec scale(maxlabel);
    scale.zeros();
    for(int i=0;i<k;i++){
      
      lab[i]=label[nn_index[j*k+i]-1];
      
      scale(lab[i]-1)=scale(lab[i]-1)+1;
      
      
      int most_common=-1;
      int value=0;
      for(int h=0;h<maxlabel;h++){
        if(scale(h)>value){
          value=scale(h);
          
          most_common=h;
        }
      }
      Ytest(j,i)=most_common+1;
    }
    delete [] lab;
  }
  
  delete [] nn_index;
  delete [] distances;
  return Ytest;
}


// [[Rcpp::export]]
arma::mat knn_kodama_r(arma::mat Xtrain,arma::vec Ytrain,arma::mat Xtest,int k,int scaling) {
  List temp0=scalecpp(Xtrain,Xtest,scaling);
  arma::mat Xtrain1=temp0[0];
  arma::mat Xtest1=temp0[1];
  
  double* data = Xtrain1.memptr();
  double *label=Ytrain.memptr();
  double* query = Xtest1.memptr();
  
  int D=Xtrain.n_cols;
  int ND=Xtrain1.n_rows;
  int NQ=Xtest1.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::mat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  for(int j=0;j<NQ;j++){
    double *lab= new double[k];
    double media=0;
    for(int i=0;i<k;i++){
      lab[i]=label[nn_index[j*k+i]-1];
      media=media+lab[i];
      Ytest(j,i)=media/(i+1);
    }
    delete [] lab;
  }
  delete [] nn_index;
  delete [] distances;
  return Ytest;
}



// [[Rcpp::export]]
arma::ivec KNNCV(arma::mat x,arma::ivec cl,arma::ivec constrain,int k) {

  arma::ivec Ytest(x.n_rows);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
  IntegerVector v=samplewithoutreplace(frame,xsa_t);

  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;


  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::ivec Ytrain;
    
    w1=find(fold==i);
    w9=find(fold!=i);
    temp=unique(cl(w1));
    if(temp.size()>1){
      Xtrain=x.rows(w9);
      Xtest=x.rows(w1);
      Ytrain=cl.elem(w9);
      arma::imat temp69=knn_kodama(Xtrain,Ytrain,Xtest,k);
      Ytest.elem(w1)=temp69.col(k-1);
    }else{
      Ytest.elem(w1)=cl.elem(w1);
      
    }
  }  
  return Ytest;
}

arma::mat pred_pls_pos(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp,arma::mat POS) {

  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  // arma::mat mm=a*b;
  
  //X=Xtrain
  arma::mat X=Xtrain;

  

  
  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX=mean(Xtrain,0);
  X.each_row()-=mX;
  Xtest.each_row()-=mX;
  
  arma::mat Y=Ytrain;
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,ncomp);
  VV.zeros();
  
  //  UU<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat UU(n,ncomp);
  UU.zeros();
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,ncomp);
  B.zeros();
  
  // Ypred <- matrix(0,ncol=m,nrow=n)
  arma::cube Ypred(w,m,ncomp);
  Ypred.zeros();  
  
  
  arma::mat qq;
  arma::mat pp;
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat uu;
  arma::mat vv;

  
  

  // for(a in 1:ncomp){
    for (int a=0; a<ncomp; a++) {
      //qq<-svd(S)$v[,1]
      //rr <- S%*%qq    


      
      svd_econ(svd_U,svd_s,svd_V,S,"left");
      
      rr=svd_U.col( 0 );

      

      
      // tt<-scale(X%*%rr,scale=FALSE)
      tt=X*rr; 
      arma::mat mtt=mean(tt,0);
      tt.each_row()-=mtt;
      //tnorm<-sqrt(sum(tt*tt))
      double tnorm=sqrt(sum(sum(tt%tt)));
      
      //tt<-tt/tnorm
      tt/=tnorm;
      
      //rr<-rr/tnorm
      rr/=tnorm;
      
      // pp <- crossprod(X,tt)
      pp=trans(X)*tt;
      
      // qq <- crossprod(Y,tt)
      qq=trans(Y)*tt;
      
      
      //uu <- Y%*%qq
      uu=Y*qq;
      
      //vv<-pp
      vv=pp;
      
      if(a>0){
        //vv<-vv-VV%*%crossprod(VV,pp)
        vv-=VV*(trans(VV)*pp);
        
        //uu<-uu-TT%*%crossprod(TT,uu)
        uu-=TT*(trans(TT)*uu);
      }
      
      //vv <- vv/sqrt(sum(vv*vv))
      vv/=sqrt(sum(sum(vv%vv)));
      
      //S <- S-vv%*%crossprod(vv,S)
      S-=vv*(trans(vv)*S);
      
      //RR[,a]=rr
      RR.col(a)=rr;
      TT.col(a)=tt;
      PP.col(a)=pp;
      QQ.col(a)=qq;
      VV.col(a)=vv;
      UU.col(a)=uu;
      B.slice(a)=RR*trans(QQ);
   
   
      Ypred.slice(a)=Xtest*B.slice(a);
      
 

    } 
    for (int a=0; a<ncomp; a++) {
      arma::mat temp1=Ypred.slice(a);
      temp1.each_row()+=mY;
      Ypred.slice(a)=temp1;
    }  


    arma::mat sli=Ypred.slice(ncomp-1);
    
    

    //int w = Xtest.n_rows;
    int k=POS.n_cols;
    arma::umat Mtest(w,m);
    Mtest.zeros();
    for(int j=0;j<w;j++){
      
      // m is the number of column of Ytrain
      for(int i=0;i<k;i++){
        
        
        arma::umat temp=Ytrain.row(POS(j,i)-1)==1;


        Mtest.row(j)= temp || Mtest.row(j)==1;
      }
         
        
      
    }
    

    sli=sli % Mtest;
    
    
    return sli;
    
    
  }
   



arma::mat pred_pls(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp) {
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  // arma::mat mm=a*b;
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  
  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX=mean(Xtrain,0);
  X.each_row()-=mX;
  Xtest.each_row()-=mX;

  arma::mat Y=Ytrain;
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,ncomp);
  VV.zeros();
  
  //  UU<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat UU(n,ncomp);
  UU.zeros();
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,ncomp);
  B.zeros();
  
  // Ypred <- matrix(0,ncol=m,nrow=n)
  arma::cube Ypred(w,m,ncomp);
  Ypred.zeros();  
  
  
  arma::mat qq;
  arma::mat pp;
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat uu;
  arma::mat vv;

  // for(a in 1:ncomp){
  for (int a=0; a<ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq    

    svd_econ(svd_U,svd_s,svd_V,S,"left");

    rr=svd_U.col( 0 );
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm; 
    //rr<-rr/tnorm
    rr/=tnorm;
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    //uu <- Y%*%qq
    uu=Y*qq;
    
    //vv<-pp
    vv=pp;
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);

      //uu<-uu-TT%*%crossprod(TT,uu)
      uu-=TT*(trans(TT)*uu);
    }
    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S); 
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    UU.col(a)=uu;
    B.slice(a)=RR*trans(QQ);
    Ypred.slice(a)=Xtest*B.slice(a);
  } 
  for (int a=0; a<ncomp; a++) {
    arma::mat temp1=Ypred.slice(a);
    temp1.each_row()+=mY;
    Ypred.slice(a)=temp1;
  }  

  arma::mat sli=Ypred.slice(ncomp-1);
  
  return sli;
  
  
}





// [[Rcpp::export]]
arma::mat transformy(arma::ivec y){
  int n=y.size();
  int nc=max(y);
  arma::mat yy(n,nc);
  yy.zeros();
  for(int i=0;i<nc;i++){
    for(int j=0;j<n;j++){
      yy(j,i)=((i+1)==y(j));
    }
  }
  
  return yy;
}






// [[Rcpp::export]]
arma::ivec PLSDACV(arma::mat x,arma::ivec cl,arma::ivec constrain,int k) {
  
  
  arma::mat clmatrix=transformy(cl);
  
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  
  int xsa_t = max(constrain);
  
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    
    w1=find(fold==i);
    w9=find(fold!=i);
    temp=unique(cl(w9));    //  I changed this temp=unique(cl(w1));
    if(temp.size()>1){
      Xtrain=x.rows(w9);
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);

      

      Ytest.rows(w1)=pred_pls(Xtrain,Ytrain,Xtest,k);

    }else{
      Ytest.rows(w1)=clmatrix.rows(w1);

    }
  }  
  
  int mm2=constrain.size();
  arma::ivec pp(mm2);
  
  //min_val is modified to avoid a warning
  double min_val=0;
  min_val++;
  arma::uvec ww;
  for (int i=0; i<mm2; i++) {
    ww=i;
    arma::mat v22=Ytest.rows(ww);
    arma::uword index;                                                                                                                                                                                                                                                                                                                
    min_val = v22.max(index);
    pp(i)=index+1;
  }
  return pp;
}




// [[Rcpp::export]]
arma::ivec KNNPLSDACV(arma::mat x,arma::ivec cl,arma::ivec constrain,int k,arma::mat pos,int knn) {

  arma::mat clmatrix=transformy(cl);
  
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  
  int xsa_t = max(constrain);
  
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  
  
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest,POStrain,POStest;
    arma::mat Ytrain;
    
    w1=find(fold==i);
    w9=find(fold!=i);
    temp=unique(cl(w9));
    if(temp.size()>1){
      Xtrain=x.rows(w9);
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);

  
      
      List resELM =  elm_train_rcpp(Xtrain, Ytrain, k, "relu");
     
      Ytest.rows(w1)=elm_predict_rcpp(resELM,Xtest);

      
      
    }else{

      Ytest.rows(w1)=clmatrix.rows(w1);
      

    }
  }  

  Ytest.elem( find(Ytest==0) ) -= 999.0;
  
 
  int mm2=constrain.size();
  arma::ivec pp(mm2);
  
  //min_val is modified to avoid a warning
  double min_val=0;
  min_val++;
  arma::uvec ww;
  for (int i=0; i<mm2; i++) {
    ww=i;
    arma::mat v22=Ytest.rows(ww);
    arma::uword index;                                                                                                                                                                                                                                                                                                                
    min_val = v22.max(index);
    pp(i)=index+1;
  }
  return pp;
}





// [[Rcpp::export]]
List pls_kodama(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp,int scaling) {
  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  
  arma::mat mX=mean(Xtrain,0);
  Xtrain.each_row()-=mX;
  Xtest.each_row()-=mX;
  arma::mat vX=variance(Xtrain); 
  if(scaling==2){
    Xtrain.each_row()/=vX;
    Xtest.each_row()/=vX;  
  }
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  // arma::mat mm=a*b;
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  //Y=Ytrain
  arma::mat Y=Ytrain;
  
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,ncomp);
  VV.zeros();
  
  //  UU<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat UU(n,ncomp);
  UU.zeros();
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,ncomp);
  B.zeros();
  
  // Ypred <- matrix(0,ncol=m,nrow=n)
  arma::cube Ypred(w,m,ncomp);
  Ypred.zeros();  
  
  
  arma::mat qq;
  arma::mat pp;
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat uu;
  arma::mat vv;

  // for(a in 1:ncomp){
  for (int a=0; a<ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq
    svd_econ(svd_U,svd_s,svd_V,S,"left");
    rr=svd_U.col( 0 );
    
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm;
    
    //rr<-rr/tnorm
    rr/=tnorm;
    
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    
    //uu <- Y%*%qq
    uu=Y*qq;
    
    //vv<-pp
    vv=pp;
    
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);
      
      
      //uu<-uu-TT%*%crossprod(TT,uu)
      uu-=TT*(trans(TT)*uu);
    }

    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S);
    
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    UU.col(a)=uu;
    
    
    
    B.slice(a)=RR*trans(QQ);
    
    
    Ypred.slice(a)=Xtest*B.slice(a);
    
  } 
  for (int a=0; a<ncomp; a++) {
    arma::mat temp1=Ypred.slice(a);
    temp1.each_row()+=mY;
    Ypred.slice(a)=temp1;
  }  
  
  arma::mat Rnorm1(ncomp,ncomp);
  Rnorm1.zeros();
  Rnorm1.diag()=sqrt(sum(RR%RR)); 
  
  arma::mat Rnorm2(ncomp,ncomp);
  Rnorm2.zeros();
  Rnorm2.diag()=1/sqrt(sum(RR%RR)); 
  
  
  arma::mat wM=Rnorm1;
  arma::mat wMi=Rnorm2;
  
  RR=RR*wMi;
  TT=TT*wMi;
  QQ=QQ*wM;
  PP=PP*wM;
  
  arma::mat Ztest = Xtest*RR;
  
  return List::create(
    Named("B") = B,
    Named("Ypred")   = Ypred,
    Named("P")       = PP,
    Named("Q")       = QQ,
    Named("T")       = TT,
    Named("R")       = RR,
    Named("Xtest")   = Ztest
  );
}

// [[Rcpp::export]]
int unic(arma::mat x){
  int x_size=x.size();
  for(int i=0;i<x_size;i++){
    if(x(i)!=x(0))
      return 2;
  }
    return 1;
}


// [[Rcpp::export]]
double RQ(arma::vec yData,arma::vec yPred){


  double my=mean(yData);
  double TSS=0,PRESS=0;
  for(unsigned int j=0;j<yData.n_elem;j++){
    double b1=yPred(j);
    double c1=yData(j);
    double d1=yData(j)-my;
    double arg_TR=(c1-b1);
    PRESS+=arg_TR*arg_TR;
    TSS+=d1*d1;  
    
  }

  double R2Y=1-PRESS/TSS;
  return R2Y;
}






// [[Rcpp::export]]
List optim_pls_cv(arma::mat x,arma::mat clmatrix,arma::ivec constrain,int ncomp, int scaling) {
  
  int nsamples=x.n_rows;
  int nvar=x.n_cols;
  ncomp=min(ncomp,nvar);
  ncomp=min(ncomp,nsamples);
  int ncolY=clmatrix.n_cols;
  arma::cube Ypred(nsamples,ncolY,ncomp); 
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    if(unic(clmatrix.rows(w9))==2){
      Xtrain=x.rows(w9);
      
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);

      List pls=pls_kodama(Xtrain,Ytrain,Xtest,ncomp,scaling);
      arma::cube temp1=pls[1];
      for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=temp1(ii,kk,jj);  
    }else{
      for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=clmatrix(w1[0],kk);  
    }
  }  

  List pls2=pls_kodama(x,clmatrix,x,ncomp,scaling);
  
  arma::cube BB  =pls2[0];
  arma::cube Yfit=pls2[1];
  arma::mat  PP  =pls2[2];
  arma::mat  QQ  =pls2[3];
  arma::mat  TT  =pls2[4];
  arma::mat  RR  =pls2[5];
  
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  for(int i=0;i<ncomp;i++){
    arma::mat Ypred_i=Ypred.slice(i);
    arma::mat Yfit_i=Yfit.slice(i);
    
    
    //    arma::mat mYpred=Ypred_i;
    arma::mat mYpred=clmatrix;
    //    arma::mat my_i=mean(Ypred_i,0);
    arma::mat my_i=mean(clmatrix,0);
    mYpred.each_row()-=my_i;
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        
        double a1=Ypred_i(j,k);
        double b1=clmatrix(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }

  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;

  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("B")          = BB,
    Named("P")          = PP,
    Named("Q")          = QQ,
    Named("T")          = TT,
    Named("R")          = RR
  );

}







// [[Rcpp::export]]
List optim_knn_r_cv(arma::mat x,arma::vec clmatrix,arma::ivec constrain,int ncomp,int scaling) {
  int nsamples=x.n_rows;
  ncomp=min(ncomp,nsamples);
  arma::mat Ypred(nsamples,ncomp); 
  arma::vec Ytest(clmatrix.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::vec Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Ytrain=clmatrix.elem(w9);
    int a=1;
    
    int Ytrain_size=Ytrain.size();
    for(int i=0;i<Ytrain_size;i++){
      if(Ytrain(i)!=Ytrain(0))
        a=2;
    }
      if(a==2){ 
        Xtrain=x.rows(w9);
        Xtest=x.rows(w1);

        arma::mat knn=knn_kodama_r(Xtrain,Ytrain,Xtest,ncomp,scaling);
        for(int ii=0;ii<w1_size;ii++)  
          for(int jj=0;jj<ncomp;jj++)  
            Ypred(w1[ii],jj)=knn(ii,jj);  
      }else{
        for(int ii=0;ii<w1_size;ii++) for(int jj=0;jj<ncomp;jj++) Ypred(w1[ii],jj)=clmatrix(w1[0]);  
        
      }
  }  
  
  arma::mat Yfit=knn_kodama_r(x,clmatrix,x,ncomp,scaling);
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  for(int i=0;i<ncomp;i++){
    arma::vec Ypred_i=Ypred.col(i);
    arma::vec Yfit_i=Yfit.col(i);
    
    arma::vec mYpred=clmatrix;
    double my_i=mean(clmatrix);
    for(int j=0;j<nsamples;j++){
      mYpred(j)=mYpred(j)-my_i;
    }
    
    
    double PRESSQ=0,TSS=0,PRESSR=0;
    
    for(int j=0;j<nsamples;j++){
      double a1=Ypred_i(j);
      double a2=clmatrix(j);
      double b1=Yfit_i(j);
      double d1=mYpred(j);
      double arg_TQ=(a1-a2);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(b1-a2);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;  
      
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }
  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;
  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y
    
  ) ;
}


// [[Rcpp::export]]
List optim_knn_c_cv(arma::mat x,arma::ivec clmatrix,arma::ivec constrain,int ncomp,int scaling) {
  int nsamples=x.n_rows;
  ncomp=min(ncomp,nsamples);
  arma::imat Ypred(nsamples,ncomp); 
  arma::ivec Ytest(clmatrix.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::ivec Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Ytrain=clmatrix.elem(w9);
    int a=1;
    
    int Ytrain_size=Ytrain.size();
    for(int i=0;i<Ytrain_size;i++){
      if(Ytrain(i)!=Ytrain(0))
        a=2;
    }
      if(a==2){ 
        Xtrain=x.rows(w9);
        Xtest=x.rows(w1);

        arma::imat knn=knn_kodama_c(Xtrain,Ytrain,Xtest,ncomp,scaling);
        for(int ii=0;ii<w1_size;ii++)  
          for(int jj=0;jj<ncomp;jj++)  
            Ypred(w1[ii],jj)=knn(ii,jj);  
      }else{
        for(int ii=0;ii<w1_size;ii++) for(int jj=0;jj<ncomp;jj++) Ypred(w1[ii],jj)=clmatrix(w1[0]);  
        
      }
  }  

  arma::imat Yfit=knn_kodama_c(x,clmatrix,x,ncomp,scaling);
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  arma::mat clmatrix2=transformy(clmatrix);
  int ncolY=clmatrix2.n_cols;
  for(int i=0;i<ncomp;i++){
    arma::mat Ypred_i=transformy(Ypred.col(i));
    arma::mat Yfit_i=transformy(Yfit.col(i));
    arma::mat mYpred=clmatrix2;
    
    arma::mat my_i=mean(clmatrix2,0);
    mYpred.each_row()-=my_i;
    
    
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        double a1=Ypred_i(j,k);
        double b1=clmatrix2(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }
  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;
  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y
    
  ) ;
}




// [[Rcpp::export]]
List double_pls_cv(arma::mat x,arma::mat y,arma::ivec constrain,int type,int verbose,int compmax, int opt, int scaling) {
  if(verbose==2) Rcpp::Rcout<<".";
  
  arma::mat clmatrix;
  arma::ivec cl;
  arma::vec cl_sup;
  if(type==1){
    cl=as<arma::ivec>(wrap(y));
    clmatrix=transformy(cl);
  }
  if(type==2){
    clmatrix=y;
  }
  int best_comp=compmax;
  int nsamples=x.n_rows;
  int nvar=x.n_cols;
  int ncomp=min(nsamples,nvar);
  ncomp=min(ncomp,compmax);
  
  int ncolY=clmatrix.n_cols;
  arma::mat Ypred(nsamples,ncolY); 
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  double mean_nc=0;
  int n_nc=0;
  for (int i=0; i<10; i++) {
    Rcpp::checkUserInterrupt();
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    if(unic(y.rows(w9))==2){
      Xtrain=x.rows(w9);
      arma::ivec constrain_train=constrain(w9);
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);

      if(opt==1){
        List optim=optim_pls_cv(Xtrain,Ytrain,constrain_train,ncomp,scaling);
        best_comp=optim[0];
      }else{
        best_comp=ncomp;
      }
      
      mean_nc+=best_comp;
      if(verbose==1) Rcpp::Rcout<<"Number of component selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
      
      List pls=pls_kodama(Xtrain,Ytrain,Xtest,best_comp,scaling);
      arma::cube temp1=pls[1];
      
      
      
      for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=temp1(ii,kk,best_comp-1);  
      
      n_nc++;
    }else{
      if(verbose==1) Rcpp::Rcout<<"Number of component selected (loop #"<<i+1<<"): "<<"NA\n";
      
      for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=clmatrix(w1[0],kk);  
    }
    
  }  
  int b_comp;
  if(opt==1){
    List optimALL=optim_pls_cv(x,clmatrix,constrain,ncomp,scaling);
    b_comp=optimALL[0];
  }else{
    b_comp=ncomp;
  }
  
  
  
  if(verbose==1) Rcpp::Rcout<<"Number of component selected for R2y calculation: "<<b_comp<<"\n";
  
  List pls2=pls_kodama(x,clmatrix,x,b_comp,scaling);

  
  
  arma::cube BB    = pls2[0];
  arma::cube temp2 = pls2[1];
  arma::mat  PP    = pls2[2];
  arma::mat  QQ    = pls2[3];
  arma::mat  TT    = pls2[4];
  arma::mat  RR    = pls2[5];
  
  arma::mat Yfit(nsamples,ncolY); 
  for(int ii=0;ii<nsamples;ii++)  for(int kk=0;kk<ncolY;kk++)  Yfit(ii,kk)=temp2(ii,kk,b_comp-1);  
  
  arma::vec res_pred(nsamples),res_fit(nsamples);
  double Q2Y;
  double R2Y;
  
  
  
  //  arma::mat mYpred=Ypred;
  
  arma::mat mYpred=clmatrix;
  //  arma::mat my_i=mean(Ypred,0);
  arma::mat my_i=mean(clmatrix,0);
  mYpred.each_row()-=my_i;
  double PRESSQ=0,TSS=0,PRESSR=0;
  for(int k=0;k<ncolY;k++){
    for(int j=0;j<nsamples;j++){
      double a1=Ypred(j,k);
      double b1=clmatrix(j,k);
      double c1=Yfit(j,k);
      double d1=mYpred(j,k);
      
      double arg_TQ=(a1-b1);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(c1-b1);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;
    }
  }
  Q2Y=1-PRESSQ/TSS;
  R2Y=1-PRESSR/TSS;
  
  
  
  if(type==1){
    
    
    //min_val is modified to avoid a warning
    double min_val=0;
    min_val++;
    arma::uvec ww;
    for (int i=0; i<nsamples; i++) {
      ww=i;
      arma::mat v22=Ypred.rows(ww);
      arma::mat v33=Yfit.rows(ww);
      arma::uword index;                                                                                                                                                                                                                                                                                                                
      min_val = v22.max(index);
      res_pred(i)=index+1;
      min_val = v33.max(index);
      res_fit(i)=index+1;
    }
  }
  if(type==2){
    for (int j=0; j<nsamples; j++) {
      res_pred(j)=Ypred(j,0);
      res_fit(j)=Yfit(j,0);
    }
  }
  
  if(b_comp==1){
    
    List pls3=pls_kodama(x,clmatrix,x,2,scaling);
    
    arma::cube BB2    = pls3[0];
    arma::mat  PP2    = pls3[2];
    arma::mat  QQ2    = pls3[3];
    arma::mat  TT2    = pls3[4];
    arma::mat  RR2    = pls3[5];
    
    BB=BB2;
    PP=PP2;
    QQ=QQ2;
    TT=TT2;
    RR=RR2;
  }
  
  return List::create(
    Named("Yfit")  = res_fit,
    Named("Ypred") = res_pred,
    Named("Q2Y")   = Q2Y,
    Named("R2Y")   = R2Y,
    Named("B")     = BB,
    Named("P")     = PP,
    Named("Q")     = QQ,
    Named("T")     = TT,
    Named("R")     = RR,
    Named("bcomp") = b_comp
  ) ;
  
  
  
}


////////////////////////////////////////////////////////////////////////////////////


// [[Rcpp::export]]
List double_knn_cv(arma::mat x,arma::vec yy,arma::ivec constrain,int type,int verbose,int compmax, int opt, int scaling) {
  
  if(verbose==2) Rcpp::Rcout<<".";
  
  int best_comp=compmax;
  int nsamples=x.n_rows;
  int ncomp=min(nsamples,compmax);
  arma::vec Ypred(nsamples); 
  arma::vec Ytest(yy.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);

  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  double mean_nc=0;
  int n_nc=0;
  
  
  for (int i=0; i<10; i++) {
    
    
    
  
    Rcpp::checkUserInterrupt();
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    Ytrain=yy.elem(w9);
    
    int gg=1;
    int y_size=yy.size();
    for(int j=0;j<y_size;j++)
      if(yy(j)!=yy(0))
        gg=2;
    
      
    if(gg==2){
      Xtrain=x.rows(w9);
      arma::ivec constrain_train=constrain(w9);
      Xtest=x.rows(w1);
        

      List optim;
      arma::mat knn;
      if(type==1){
        arma::ivec iYtrain=as<arma::ivec>(wrap(Ytrain));
        if(opt==1){
          optim=optim_knn_c_cv(Xtrain,iYtrain,constrain_train,ncomp,scaling);
          best_comp=optim[0];
        }else{
          best_comp=ncomp;
        }
        mean_nc+=best_comp;
        if(verbose==1) Rcpp::Rcout<<"Number of k selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
        arma::imat iknn=knn_kodama_c(Xtrain,iYtrain,Xtest,best_comp,scaling);
        knn=as<arma::mat>(wrap(iknn));
      }
      if(type==2){
        optim=optim_knn_r_cv(Xtrain,Ytrain,constrain_train,ncomp,scaling);
        best_comp=optim[0];
        mean_nc+=best_comp;
        if(verbose==1) Rcpp::Rcout<<"Number of k selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
        knn=knn_kodama_r(Xtrain,Ytrain,Xtest,best_comp,scaling);
      }
      for(int ii=0;ii<w1_size;ii++)  Ypred(w1[ii])=knn(ii,best_comp-1);  
      n_nc++;
    }else{
      if(verbose==1) Rcpp::Rcout<<"Number of k selected (loop #"<<i+1<<"): "<<"NA\n";
      for(int ii=0;ii<w1_size;ii++) Ypred(w1[ii])=yy(w1[0]);  
    }
  }
  

  int b_comp;
  if(opt==1){
    List optimALL;
    if(type==2)
      optimALL=optim_knn_r_cv(x,yy,constrain,ncomp,scaling);
    if(type==1){
      arma::ivec iyy=as<arma::ivec>(wrap(yy));
      optimALL=optim_knn_c_cv(x,iyy,constrain,ncomp,scaling);
    }
    b_comp=optimALL[0];
  }else{
    b_comp=ncomp;
  }
  

  
  if(verbose==1) Rcpp::Rcout<<"Number of k selected for R2y calculation: "<<b_comp<<"\n";
  arma::vec Yfit(nsamples); 
  double Q2Y;
  double R2Y;
  arma::mat knn2;
  
  if(type==1){
    arma::ivec iY=as<arma::ivec>(wrap(yy));
    arma::imat iknn2=knn_kodama_c(x,iY,x,b_comp,scaling);
    knn2=as<arma::mat>(wrap(iknn2));
    for(int ii=0;ii<nsamples;ii++) Yfit(ii)=knn2(ii,b_comp-1);  
    arma::mat ymatrix=transformy(iY);
    int ncolY=ymatrix.n_cols;
    arma::mat Ypred_i=transformy(as<arma::ivec>(wrap(Ypred)));
    arma::mat Yfit_i=transformy(as<arma::ivec>(wrap(Yfit)));
    arma::mat mYpred=ymatrix;
    arma::mat my_i=mean(ymatrix,0);
    mYpred.each_row()-=my_i;
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        double a1=Ypred_i(j,k);
        double b1=ymatrix(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y=1-PRESSQ/TSS;
    R2Y=1-PRESSR/TSS;
  }
  
  if(type==2){
    knn2=knn_kodama_r(x,yy,x,b_comp,scaling);
    for(int ii=0;ii<nsamples;ii++) Yfit(ii)=knn2(ii,b_comp-1);  
    arma::vec mYpred=yy;
    double my_i=mean(yy);
    for(int j=0;j<nsamples;j++){
      mYpred(j)=mYpred(j)-my_i;
    }
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int j=0;j<nsamples;j++){
      double a1=Ypred(j);
      double a2=yy(j);
      double b1=Yfit(j);
      double d1=mYpred(j);
      double arg_TQ=(a1-a2);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(b1-a2);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;  
    }
    Q2Y=1-PRESSQ/TSS;
    R2Y=1-PRESSR/TSS;
  }
  

  
  return List::create(
    Named("Yfit")  = Yfit,
    Named("Ypred") = Ypred,
    Named("Q2Y")   = Q2Y,
    Named("R2Y")   = R2Y,
    Named("bk") = b_comp
  ) ;
}






// [[Rcpp::export]]
List corecpp(arma::mat x,
             arma::mat xTdata,
             arma::ivec clbest,
             const int Tcycle,
             int FUN,
             int fparknn, 
             int fparpls, 
             arma::ivec constrain,
             NumericVector fix,
             bool shake,
             int proj,
             arma::mat posxy,
             arma::mat posxyTdata) {
  
  arma::ivec cvpred=clbest;
  arma::ivec cvpredbest;

  if(FUN==1){
    cvpredbest=PLSDACV(x,clbest,constrain,fparpls);   

  }
  if(FUN==2){
     cvpredbest=KNNPLSDACV(x,clbest,constrain,fparpls,x,fparknn);
  }
  if(FUN==3){
    cvpredbest=KNNCV(x,clbest,constrain,fparknn); 
  }

  double accbest;
  if (shake == FALSE) {
    accbest = accuracy(clbest,cvpredbest);
  }
  else {
    accbest = 0;
  }
  
  bool success = FALSE;
  int j = 0;
  
  //Inizialization of the vector to store the values of accuracy

  double *vect_acc= new double[Tcycle];
  for(int ii=0;ii<Tcycle;ii++){
    vect_acc[ii]=-1;
  }
  
    
  arma::ivec sup1= unique(constrain);
  int nconc=sup1.size();
  
  NumericVector constrain2=as<NumericVector>(wrap(constrain));
  NumericVector fix2=as<NumericVector>(wrap(fix));
  
  while (j < Tcycle && !success) {
    Rcpp::checkUserInterrupt();
    j++;
    arma::ivec cl = clbest;
    
    IntegerVector sup2=seq_len(nconc);

    int nn_temp=(unif_rand()*nconc)+1;
  
 //   int xknn=xNeighbors.n_cols;
  //  int clbest_elem=clbest.n_elem;
 //   for(unsigned jj=0;jj<clbest_elem;jj++){
//
//      int check=0;
//      for(int jj_knn=0; jj_knn<xknn; jj_knn++){
//        
//        unsigned wjj_knn_sele=xNeighbors(jj,jj_knn)-1;

        
 //       if(cvpredbest[jj]==cl[wjj_knn_sele]){
 //         check=1;
 //       }
 //     }
 //     if(check==0){
  //      cvpredbest[jj]=cl[jj];
 //     }
      

 //   }
    
  
  //////////////////////////////////////////////////
  
  //////////////////////////////////////////////////
  
  
    IntegerVector ss=samplewithoutreplace(sup2,nn_temp);

    for (int* k = ss.begin(); k != ss.end(); ++k) {
      LogicalVector sele=((constrain2 == *k) & (fix2!=1));
      double flag = std::accumulate(sele.begin(),sele.end(), 0.0);
      if (flag != 0) {
        arma::uvec whi=which(sele);
        arma::ivec uni=unique(cvpredbest.elem(whi));

        IntegerVector ss_uni=as<IntegerVector>(wrap(uni));

        int nn_uni=ss_uni.size();
        int nn_t=unif_rand()*nn_uni;
        IntegerVector soso(1);
        soso(0)=ss_uni(nn_t);   //Cambiato da soso[0]=ss_uni[nn_t]
        
        
        IntegerVector soso2=rep(soso,whi.size());

        cl.elem(whi)=as<arma::ivec>(soso2);
      }
    }
    
    
    
    
    
    

    if(FUN==1){
      cvpred=PLSDACV(x,cl,constrain,fparpls);  
    }
    if(FUN==2){
      cvpred=KNNPLSDACV(x,cl,constrain,fparpls,x,fparknn);  
    }
    if(FUN==3){
      cvpred=KNNCV(x,cl,constrain,fparknn);
    }
    double accTOT= accuracy(cl,cvpred);
    if (accTOT > accbest) {
      cvpredbest = cvpred;
      clbest = cl;
      accbest = accTOT;
    }
    
    vect_acc[j-1] = accbest;  //Cambiato da vect_acc[j] = accbest;
    if (accTOT == 1) 
      success = TRUE;
  }
  

  arma::vec vect_acc2(Tcycle);
  vect_acc2.fill(0);
  for(int ii=0;ii<Tcycle;ii++){
    vect_acc2(ii)=vect_acc[ii];
  }
  
  delete [] vect_acc;
  
  
  if(proj==2){
    arma::mat projmat;
    int mm2=xTdata.n_rows;
    arma::ivec pp(mm2); 

    if(FUN==1){
      arma::mat lcm=transformy(clbest);
      if(x.n_rows==posxy.n_rows){
        
         List res=knn_Armadillo(posxy,posxyTdata,10);
         arma::mat POS_knn=res[0];
         projmat=pred_pls_pos(x,lcm,xTdata,fparpls,POS_knn);  
      }else{
        projmat=pred_pls(x,lcm,xTdata,fparpls);
      }
      //min_val is modified to avoid a warning
      double min_val=0;
      min_val++;
      arma::uvec ww;
      for (int i=0; i<mm2; i++) {
        ww=i;
        arma::mat v22=projmat.rows(ww);
        arma::uword index;                                                                                                                                                                                                                                                                                                                
        min_val = v22.max(index);
        pp(i)=index+1;
      }
    }
    if(FUN==2){
      arma::mat lcm=transformy(clbest);
      List resELM =  elm_train_rcpp(x, lcm, fparpls, "relu");
     
      projmat=elm_predict_rcpp(resELM,xTdata);
      if(x.n_rows==posxy.n_rows){
         List res=knn_Armadillo(posxy,posxyTdata,10);
         arma::mat POS_knn=res[0];
         int k=POS_knn.n_cols;
         int w = xTdata.n_rows;
         int m = lcm.n_cols;

         arma::umat Mtest(w,m);
         Mtest.zeros();
         for(int j=0;j<w;j++){
          
           for(int i=0;i<k;i++){
           
              arma::umat temp=lcm.row(POS_knn(j,i)-1)==1;
             
              Mtest.row(j)= temp || Mtest.row(j)==1;
           }
         
        
      
        }
         projmat=projmat % Mtest;
     
      }
      
      //min_val is modified to avoid a warning
      double min_val=0;
      min_val++;
      arma::uvec ww;
      for (int i=0; i<mm2; i++) {
        ww=i;
        arma::mat v22=projmat.rows(ww);
        arma::uword index;                                                                                                                                                                                                                                                                                                                
        min_val = v22.max(index);
        pp(i)=index+1;
      }
    }
   
    if(FUN==3){
      arma::imat temp70=knn_kodama(x,clbest,xTdata,fparknn);
      pp=temp70.col(fparknn-1);
    }
    return List::create(Named("clbest") = clbest,
                        Named("accbest") = accbest,
                        Named("vect_acc") = vect_acc2,
                        Named("vect_proj") = pp
    );
    
  }else{
    return List::create(Named("clbest") = clbest,
                        Named("accbest") = accbest,
                        Named("vect_acc") = vect_acc2
    );
  }
  
}









