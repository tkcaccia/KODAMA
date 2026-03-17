# Benchmark script for KODAMA.matrix backends.
# Not run during tests; execute manually when profiling.

set.seed(1)
n <- 2000
p <- 30
x <- matrix(rnorm(n * p), nrow = n, ncol = p)

cat("Running sequential backend...\n")
t_seq <- system.time(
  kk_seq <- KODAMA::KODAMA.matrix(
    x,
    M = 10,
    Tcycle = 5,
    ncomp = 5,
    landmarks = 500,
    splitting = 50,
    n.cores = 1,
    seed = 1
  )
)
print(t_seq)

cat("Running parallel backend...\n")
t_par <- system.time(
  kk_par <- KODAMA::KODAMA.matrix(
    x,
    M = 10,
    Tcycle = 5,
    ncomp = 5,
    landmarks = 500,
    splitting = 50,
    n.cores = 2,
    seed = 1
  )
)
print(t_par)

cat("Result agreement check (res matrix):\n")
print(all.equal(kk_seq$res, kk_par$res, tolerance = 1e-8))
