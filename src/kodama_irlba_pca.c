#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

#include "irlba.h"

SEXP KODAMA_irlba_pca_dense(SEXP xSEXP, SEXP nvSEXP, SEXP maxitSEXP, SEXP tolSEXP)
{
  int nprot = 0;
  if (!isMatrix(xSEXP)) {
    error("x must be a matrix");
  }

  SEXP dim = getAttrib(xSEXP, R_DimSymbol);
  int m = INTEGER(dim)[0];
  int n = INTEGER(dim)[1];
  if (m < 4 || n < 4) {
    error("irlba backend requires at least 4 rows and 4 columns");
  }

  int nu = asInteger(nvSEXP);
  if (nu < 1) {
    error("nv must be positive");
  }
  int minmn = m < n ? m : n;
  if (nu >= minmn) {
    error("nv must be smaller than min(nrow(x), ncol(x))");
  }

  int maxit = asInteger(maxitSEXP);
  if (maxit < 1) {
    maxit = 1000;
  }
  double tol = asReal(tolSEXP);
  if (!R_FINITE(tol) || tol <= 0) {
    tol = 1e-5;
  }

  SEXP xReal = xSEXP;
  if (TYPEOF(xSEXP) != REALSXP) {
    xReal = PROTECT(coerceVector(xSEXP, REALSXP));
    nprot++;
    setAttrib(xReal, R_DimSymbol, dim);
  }

  int work = 2 * nu + 5;
  if (work >= minmn) {
    work = minmn - 1;
  }
  if (work <= nu) {
    work = nu + 1;
  }
  if (work < 4) {
    work = 4;
  }

  int lwork = 5 * work * work + 10 * work;
  if (lwork < 1) {
    lwork = 1;
  }

  double *A = REAL(xReal);
  double *s = (double *) R_alloc((size_t) nu, sizeof(double));
  double *U = (double *) R_alloc((size_t) m * (size_t) work, sizeof(double));
  double *V = (double *) R_alloc((size_t) n * (size_t) work, sizeof(double));

  double *V1 = (double *) R_alloc((size_t) n * (size_t) work, sizeof(double));
  double *U1 = (double *) R_alloc((size_t) m * (size_t) work, sizeof(double));
  double *W = (double *) R_alloc((size_t) m * (size_t) work, sizeof(double));
  double *F = (double *) R_alloc((size_t) n, sizeof(double));
  double *B = (double *) R_alloc((size_t) work * (size_t) work, sizeof(double));
  double *BU = (double *) R_alloc((size_t) work * (size_t) work, sizeof(double));
  double *BV = (double *) R_alloc((size_t) work * (size_t) work, sizeof(double));
  double *BS = (double *) R_alloc((size_t) work, sizeof(double));
  double *BW = (double *) R_alloc((size_t) lwork, sizeof(double));
  double *res = (double *) R_alloc((size_t) work, sizeof(double));
  double *T = (double *) R_alloc((size_t) lwork, sizeof(double));
  double *svratio = (double *) R_alloc((size_t) work, sizeof(double));

  for (int i = 0; i < n * work; ++i) {
    V[i] = 0.0;
  }
  GetRNGstate();
  for (int i = 0; i < n; ++i) {
    V[i] = norm_rand();
  }
  PutRNGstate();

  int iter = 0;
  int mprod = 0;
  int status = irlb(
    A, NULL, 0, m, n, nu, work, maxit, 0, tol,
    NULL, NULL, NULL,
    s, U, V, &iter, &mprod, 1e-12,
    lwork, V1, U1, W, F, B, BU, BV, BS, BW, res, T,
    1e-3, svratio
  );

  if (status < 0) {
    UNPROTECT(nprot);
    error("internal IRLBA solver failed with status %d", status);
  }

  SEXP uSEXP = PROTECT(allocMatrix(REALSXP, m, nu)); nprot++;
  SEXP dSEXP = PROTECT(allocVector(REALSXP, nu)); nprot++;
  SEXP vSEXP = PROTECT(allocMatrix(REALSXP, n, nu)); nprot++;
  for (int j = 0; j < nu; ++j) {
    for (int i = 0; i < m; ++i) {
      REAL(uSEXP)[(size_t) j * (size_t) m + i] = U[(size_t) j * (size_t) m + i];
    }
    REAL(dSEXP)[j] = s[j];
    for (int i = 0; i < n; ++i) {
      REAL(vSEXP)[(size_t) j * (size_t) n + i] = V[(size_t) j * (size_t) n + i];
    }
  }

  SEXP out = PROTECT(allocVector(VECSXP, 5)); nprot++;
  SET_VECTOR_ELT(out, 0, uSEXP);
  SET_VECTOR_ELT(out, 1, dSEXP);
  SET_VECTOR_ELT(out, 2, vSEXP);
  SET_VECTOR_ELT(out, 3, ScalarInteger(iter));
  SET_VECTOR_ELT(out, 4, ScalarInteger(mprod));

  SEXP out_names = PROTECT(allocVector(STRSXP, 5)); nprot++;
  SET_STRING_ELT(out_names, 0, mkChar("u"));
  SET_STRING_ELT(out_names, 1, mkChar("d"));
  SET_STRING_ELT(out_names, 2, mkChar("v"));
  SET_STRING_ELT(out_names, 3, mkChar("iter"));
  SET_STRING_ELT(out_names, 4, mkChar("mprod"));
  setAttrib(out, R_NamesSymbol, out_names);

  UNPROTECT(nprot);
  return out;
}
