
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>

#define USE_FC_LEN_T
#include <Rconfig.h>
#include <R_ext/BLAS.h>

#ifndef FCONE
# define FCONE
#endif
#include <R.h>
#define USE_RINTERNALS
#include <Rinternals.h>
#include <Rdefines.h>

#include <R_ext/Lapack.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>
#include <R_ext/Parse.h>

#include <Matrix.h>
#include <Matrix_stubs.c>



#include "irlba.h"




/* helper function for calling rnorm below */
SEXP
RNORM (int n)
{
  char buf[4096];
  SEXP cmdSexp, cmdexpr, ans = R_NilValue;
  ParseStatus status;
  cmdSexp = PROTECT (allocVector (STRSXP, 1));
  snprintf (buf, 4095, "rnorm(%d)", n);
  SET_STRING_ELT (cmdSexp, 0, mkChar (buf));
  cmdexpr = PROTECT (R_ParseVector (cmdSexp, -1, &status, R_NilValue));
  if (status != PARSE_OK)
    {
      UNPROTECT (2);
      error ("invalid call");
    }
  for (int i = 0; i < length (cmdexpr); i++)
    {
      ans = PROTECT (eval (VECTOR_ELT (cmdexpr, i), R_GlobalEnv));
      UNPROTECT (1);
    }
  UNPROTECT (2);
  return ans;
}


void
orthog (double *X, double *Y, double *T, int xm, int xn, int yn)
{
  double a = 1, b = 1;
  int inc = 1;
  memset (T, 0, xn * yn * sizeof (double));
  // T = t(X) * Y
  F77_CALL (dgemv) ("t", &xm, &xn, &a, X, &xm, Y, &inc, &b, T, &inc FCONE);
  // Y = Y - X * T
  a = -1.0;
  b = 1.0;
  F77_CALL (dgemv) ("n", &xm, &xn, &a, X, &xm, T, &inc, &b, Y, &inc FCONE);
}


/*
 * Convergence tests
 * Input parameters
 * Bsz            number of rows of the bidiagonal matrix B (scalar)
 * tol            convergence tolerance (scalar)
 * svtol          max change in each singular value tolerance (scalar)
 * n              requested number of singular values
 * Smax           largest singular value of B
 * svratio        vector of abs(current - previous) / current singular value ratios
 * residuals      vector of residual values
 * k              number of estimated signular values (scalar)
 * S              check for invariant subspace when S == 0
 *
 * Output
 * converged      0 = FALSE, 1 = TRUE (all converged)
 * k              adjusted subspace size.
 */
void
convtests (int Bsz, int n, double tol, double svtol, double Smax,
           double *svratio, double *residuals, int *k, int *converged, double S)
{
  int j, Len_res = 0;
  for (j = 0; j < Bsz; j++)
    {
      if ((fabs (residuals[j]) < tol * Smax) && (svratio[j] < svtol))
        Len_res++;
    }
  if (Len_res >= n || S == 0)
    {
      *converged = 1;
      return;
    }
  if (*k < n + Len_res)
    *k = n + Len_res;
  if (*k > Bsz - 3)
    *k = Bsz - 3;
  if (*k < 1)
    *k = 1;
  *converged = 0;
  return;
}





/* irlb: main computation function.
 * returns:
 *  0 on success,
 * -1 invalid dimensions,
 * -2 not converged
 * -3 out of memory
 * -4 starting vector near the null space of A
 *
 * all data must be allocated by caller, required sizes listed below
 */
int
irlb (double *A,                // Input data matrix (double case)
      void *AS,                 // input data matrix (sparse case)
      int mult,                 // 0 -> use double *A, 1 -> use AS
      int m,                    // data matrix number of rows, must be > 3.
      int n,                    // data matrix number of columns, must be > 3.
      int nu,                   // dimension of solution
      int work,                 // working dimension, must be > 3.
      int maxit,                // maximum number of main iterations
      int restart,              // 0->no, n>0 -> restarted algorithm of dimension n
      double tol,               // convergence tolerance
      double *scale,            // optional scale (NULL for no scale) size n * 2
      double *shift,            // optional shift (NULL for no shift)
      double *center,           // optional center (NULL for no center)
      // output values
      double *s,                // output singular values at least length nu
      double *U,                // output left singular vectors  m x work
      double *V,                // output right singular vectors n x work
      int *ITER,                // ouput number of Lanczos iterations
      int *MPROD,               // output number of matrix vector products
      double eps,               // tolerance for invariant subspace detection
      // working intermediate storage, sizes shown
      int lwork, double *V1,    // n x work
      double *U1,               // m x work
      double *W,                // m x work  input when restart > 0
      double *F,                // n
      double *B,                // work x work  input when restart > 0
      double *BU,               // work x work
      double *BV,               // work x work
      double *BS,               // work
      double *BW,               // lwork
      double *res,              // work
      double *T,                // lwork
      double svtol,             // svtol limit
      double *svratio)          // convtest extra storage vector of length work
{
  double d, S, R, alpha, beta, R_F, SS;
  double *x;
  int jj, kk;
  int converged;
  int info, j, k = restart;
  int inc = 1;
  int mprod = 0;
  int iter = 0;
  double Smax = 0;
  SEXP FOO;

/* Check for valid input dimensions */
  if (work < 4 || n < 4 || m < 4)
    return -1;

  if (restart == 0)
    memset (B, 0, work * work * sizeof (double));
  memset(svratio, 0, work * sizeof(double));

/* Main iteration */
  while (iter < maxit)
    {
      j = 0;
/*  Normalize starting vector */
      if (iter == 0 && restart == 0)
        {
          d = F77_CALL (dnrm2) (&n, V, &inc);
          if (d < eps)
            return -1;
          d = 1 / d;
          F77_CALL (dscal) (&n, &d, V, &inc);
        }
      else
        j = k;

/* optionally apply scale */
      x = V + j * n;
      if (scale)
        {
          x = scale + n;
          memcpy (scale + n, V + j * n, n * sizeof (double));
          for (kk = 0; kk < n; ++kk)
            x[kk] = x[kk] / scale[kk];
        }

      switch (mult)
        {
        case 1:
          dsdmult ('n', m, n, AS, x, W + j * m);
          break;
        default:
          alpha = 1;
          beta = 0;
          F77_CALL (dgemv) ("n", &m, &n, &alpha, (double *) A, &m, x,
                            &inc, &beta, W + j * m, &inc FCONE);
        }
      mprod++;
      R_CheckUserInterrupt ();
/* optionally apply shift in square cases m = n */
      if (shift)
        {
          jj = j * m;
          for (kk = 0; kk < m; ++kk)
            W[jj + kk] = W[jj + kk] + shift[0] * x[kk];
        }
/* optionally apply centering */
      if (center)
        {
          jj = j * m;
          beta = F77_CALL (ddot) (&n, x, &inc, center, &inc);
          for (kk = 0; kk < m; ++kk)
            W[jj + kk] = W[jj + kk] - beta;
        }
      if (iter > 0)
        orthog (W, W + j * m, T, m, j, 1);
      S = F77_CALL (dnrm2) (&m, W + j * m, &inc);
      if (S < eps && j == 0)
        return -4;
      SS = 1.0 / S;
      F77_CALL (dscal) (&m, &SS, W + j * m, &inc);

/* The Lanczos process */
      while (j < work)
        {
          switch (mult)
            {
            case 1:
              dsdmult ('t', m, n, AS, W + j * m, F);
              break;
            default:
              alpha = 1.0;
              beta = 0.0;
              F77_CALL (dgemv) ("t", &m, &n, &alpha, (double *) A, &m,
                                W + j * m, &inc, &beta, F, &inc FCONE);
            }
          mprod++;
          R_CheckUserInterrupt ();
/* optionally apply shift, scale, center */
          if (shift)
            {
              // Note, not a bug because shift only applies to square matrices
              for (kk = 0; kk < m; ++kk)
                F[kk] = F[kk] + shift[0] * W[j * m + kk];
            }
          if (scale)
            {
              for (kk = 0; kk < n; ++kk)
                F[kk] = F[kk] / scale[kk];
            }
          if (center)
            {
              beta = 0;
              for (kk = 0; kk < m; ++kk) beta += W[j *m + kk];
              if (scale)
                for (kk = 0; kk < n; ++kk)
                  F[kk] = F[kk] - beta * center[kk] / scale[kk];
              else
                for (kk = 0; kk < n; ++kk)
                  F[kk] = F[kk] - beta * center[kk];
            }
          SS = -S;
          F77_CALL (daxpy) (&n, &SS, V + j * n, &inc, F, &inc);
          orthog (V, F, T, n, j + 1, 1);

          if (j + 1 < work)
            {
              R_F = F77_CALL (dnrm2) (&n, F, &inc);
              R = 1.0 / R_F;
              if (R_F < eps)        // near invariant subspace
                {
                  FOO = RNORM (n);
                  for (kk = 0; kk < n; ++kk)
                    F[kk] = REAL (FOO)[kk];
                  orthog (V, F, T, n, j + 1, 1);
                  R_F = F77_CALL (dnrm2) (&n, F, &inc);
                  R = 1.0 / R_F;
                  R_F = 0;
                }
              memmove (V + (j + 1) * n, F, n * sizeof (double));
              F77_CALL (dscal) (&n, &R, V + (j + 1) * n, &inc);
              B[j * work + j] = S;
              B[(j + 1) * work + j] = R_F;
/* optionally apply scale */
              x = V + (j + 1) * n;
              if (scale)
                {
                  x = scale + n;
                  memcpy (x, V + (j + 1) * n, n * sizeof (double));
                  for (kk = 0; kk < n; ++kk)
                    x[kk] = x[kk] / scale[kk];
                }
              switch (mult)
                {
                case 1:
                  dsdmult ('n', m, n, AS, x, W + (j + 1) * m);
                  break;
                default:
                  alpha = 1.0;
                  beta = 0.0;
                  F77_CALL (dgemv) ("n", &m, &n, &alpha, (double *) A, &m,
                                    x, &inc, &beta, W + (j + 1) * m, &inc FCONE);
                }
              mprod++;
              R_CheckUserInterrupt ();
/* optionally apply shift */
              if (shift)
                {
                  jj = j + 1;
                  for (kk = 0; kk < m; ++kk)
                    W[jj * m + kk] = W[jj * m + kk] + shift[0] * x[kk];
                }
/* optionally apply centering */
              if (center)
                {
                  jj = (j + 1) * m;
                  beta = F77_CALL (ddot) (&n, x, &inc, center, &inc);
                  for (kk = 0; kk < m; ++kk)
                    W[jj + kk] = W[jj + kk] - beta;
                }
/* One step of classical Gram-Schmidt */
              R = -R_F;
              F77_CALL (daxpy) (&m, &R, W + j * m, &inc, W + (j + 1) * m,
                                &inc);
/* full re-orthogonalization of W_{j+1} */
              orthog (W, W + (j + 1) * m, T, m, j + 1, 1);
              S = F77_CALL (dnrm2) (&m, W + (j + 1) * m, &inc);
              SS = 1.0 / S;
              if (S < eps)
                {
                  FOO = RNORM (m);
                  jj = (j + 1) * m;
                  for (kk = 0; kk < m; ++kk)
                    W[jj + kk] = REAL (FOO)[kk];
                  orthog (W, W + (j + 1) * m, T, m, j + 1, 1);
                  S = F77_CALL (dnrm2) (&m, W + (j + 1) * m, &inc);
                  SS = 1.0 / S;
                  F77_CALL (dscal) (&m, &SS, W + (j + 1) * m, &inc);
                  S = 0;
                }
              else
                F77_CALL (dscal) (&m, &SS, W + (j + 1) * m, &inc);
            }
          else
            {
              B[j * work + j] = S;
            }
          j++;
        }

      memmove (BU, B, work * work * sizeof (double));   // Make a working copy of B
      int *BI = (int *) T;
      F77_CALL (dgesdd) ("O", &work, &work, BU, &work, BS, BU, &work, BV,
                         &work, BW, &lwork, BI, &info FCONE);
      R_F = F77_CALL (dnrm2) (&n, F, &inc);
      R = 1.0 / R_F;
      F77_CALL (dscal) (&n, &R, F, &inc);
/* Force termination after encountering linear dependence */
      if (R_F < eps)
        R_F = 0;

      Smax = 0;
      for (jj = 0; jj < j; ++jj)
        {
          if (BS[jj] > Smax)
            Smax = BS[jj];
          svratio[jj] = fabs (svratio[jj] - BS[jj]) / BS[jj];
        }
      for (kk = 0; kk < j; ++kk)
        res[kk] = R_F * BU[kk * work + (j - 1)];
/* Update k to be the number of converged singular values. */
      convtests (j, nu, tol, svtol, Smax, svratio, res, &k, &converged, S);

      if (converged == 1)
        {
          iter++;
          break;
        }
      for (jj = 0; jj < j; ++jj)
        svratio[jj] = BS[jj];

      alpha = 1;
      beta = 0;
      F77_CALL (dgemm) ("n", "t", &n, &k, &j, &alpha, V, &n, BV, &work, &beta,
                        V1, &n FCONE FCONE);
      memmove (V, V1, n * k * sizeof (double));
      memmove (V + n * k, F, n * sizeof (double));

      memset (B, 0, work * work * sizeof (double));
      for (jj = 0; jj < k; ++jj)
        {
          B[jj * work + jj] = BS[jj];
          B[k * work + jj] = res[jj];
        }

/*   Update the left approximate singular vectors */
      alpha = 1;
      beta = 0;
      F77_CALL (dgemm) ("n", "n", &m, &k, &j, &alpha, W, &m, BU, &work, &beta,
                        U1, &m FCONE FCONE);
      memmove (W, U1, m * k * sizeof (double));
      iter++;
    }

/* Results */
  memmove (s, BS, nu * sizeof (double));        /* Singular values */
  alpha = 1;
  beta = 0;
  F77_CALL (dgemm) ("n", "n", &m, &nu, &work, &alpha, W, &m, BU, &work, &beta,
                    U, &m FCONE FCONE);
  F77_CALL (dgemm) ("n", "t", &n, &nu, &work, &alpha, V, &n, BV, &work, &beta,
                    V1, &n FCONE FCONE);
  memmove (V, V1, n * nu * sizeof (double));

  *ITER = iter;
  *MPROD = mprod;
  return (converged == 1 ? 0 : -2);
}


cholmod_common chol_c;
/* Need our own CHOLMOD error handler */
void attribute_hidden
irlba_R_cholmod_error (int status, const char *file, int line,
                       const char *message)
{
  if (status < 0)
    error ("Cholmod error '%s' at file:%s, line %d", message, file, line);
  else
    warning ("Cholmod warning '%s' at file:%s, line %d", message, file, line);
}


typedef int (*cholmod_sdmult_func)
(
    /* ---- input ---- */
    cholmod_sparse *A,  /* sparse matrix to multiply */
    int transpose,      /* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,   /* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,   /* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
);

void
dsdmult (char transpose, int m, int n, void * a, double *b, double *c)
{
  cholmod_sdmult_func sdmult;
  sdmult = (cholmod_sdmult_func) R_GetCCallable ("Matrix", "cholmod_sdmult");
  int t = transpose == 't' ? 1 : 0;
  CHM_SP cha = (CHM_SP) a;

  cholmod_dense chb;
  chb.nrow = transpose == 't' ? m : n;
  chb.d = chb.nrow;
  chb.ncol = 1;
  chb.nzmax = chb.nrow;
  chb.xtype = cha->xtype;
  chb.dtype = 0;
  chb.x = (void *) b;
  chb.z = (void *) NULL;

  cholmod_dense chc;
  chc.nrow = transpose == 't' ? n : m;
  chc.d = chc.nrow;
  chc.ncol = 1;
  chc.nzmax = chc.nrow;
  chc.xtype = cha->xtype;
  chc.dtype = 0;
  chc.x = (void *) c;
  chc.z = (void *) NULL;

  double one[] = { 1, 0 }, zero[] = { 0, 0};
  sdmult (cha, t, one, zero, &chb, &chc, &chol_c);
}



