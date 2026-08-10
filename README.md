# kodama-r

Thin R wrapper for the standalone `kodama-cpp` C++/CUDA/Metal library.

The wrapper and numerical core are maintained as separate repositories:

- [`tkcaccia/kodama-r`](https://github.com/tkcaccia/kodama-r)
- [`tkcaccia/kodama-cpp`](https://github.com/tkcaccia/kodama-cpp)

## Quick Install

Clone the repositories as siblings, build the core, and install the wrapper:

```sh
git clone https://github.com/tkcaccia/kodama-cpp.git
git clone https://github.com/tkcaccia/kodama-r.git

cmake -S kodama-cpp -B kodama-cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF
cmake --build kodama-cpp/build -j

Rscript -e 'install.packages("Rcpp", repos = "https://cloud.r-project.org")'

KODAMA_CPP_ROOT="$PWD/kodama-cpp" \
KODAMA_CPP_BUILD_DIR="$PWD/kodama-cpp/build" \
R CMD INSTALL kodama-r
```

Then verify the installation with:

```r
library(kodamaR)
KODAMA.diagnostics()
help(package = "kodamaR")
```

The sections below explain CPU, CUDA, Metal, and development-check workflows.

`kodama-r` does not reimplement the KODAMA mathematics in R. It converts R
matrices and vectors to the C++ ABI, calls the compiled `kodama-cpp` library,
and returns R-friendly lists, matrices, and S3 objects. The numerical kernels
remain in `kodama-cpp`, so the same core can be reused by R and Python wrappers.

## What Is Linked

At install time the package compiles only the small Rcpp bridge in `src/`.
The bridge links against a previously built `libkodama_cpp` library:

```text
R session -> kodamaR R functions -> Rcpp bridge -> libkodama_cpp
```

The wrapper exports:

- `KNNCV()` and `PLSLDACV()` for cross-validated classifier kernels.
- `CoreKNN()` and `CorePLSLDA()` for label-optimization kernels.
- `KODAMA.matrix()` for complete KODAMA matrix construction.
- `KODAMA.matrix.graph()` for bare neighbor-index/distance input.
- `KODAMA.pca()` / `kodama_pca()` for backend-native float32 PCA.
- `KODAMA.visualization()` for UMAP/openTSNE embeddings from KODAMA graphs.
- `KODAMA.graph()`, `KODAMA.makeSNNGraph()`, `makeSNNGraph()`, and
  `KODAMA.clustering()` for graph construction and CPU random-walk clustering.

## Prerequisites

Install the C++ dependencies required by `kodama-cpp` before installing the R
wrapper:

- CMake and a C++17 compiler.
- R, `Rcpp`, and `testthat`.
- The CUDA Toolkit when installing a CUDA-enabled `kodama-cpp` build.
- macOS and Xcode command-line tools for the Apple Metal backend.

On macOS with Homebrew, the CPU development environment is typically:

```sh
brew install cmake libomp
Rscript -e 'install.packages(c("Rcpp", "testthat"), repos = "https://cloud.r-project.org")'
```

On Linux/CUDA machines, use the same CUDA environment used to build the core.
The important rule is that `R CMD INSTALL` and later R sessions must see the
same CUDA Toolkit libraries used by `kodama-cpp`.

## Detailed installation

From a checkout where `kodama-cpp` and `kodama-r` are siblings:

```sh
cmake -S ../kodama-cpp -B ../kodama-cpp/build -DKODAMA_ENABLE_CUDA=OFF
cmake --build ../kodama-cpp/build -j
```

From this monorepo-style development checkout:

```sh
cmake -S ../.. -B ../../build -DKODAMA_ENABLE_CUDA=OFF
cmake --build ../../build -j
```

For a CUDA build, use a separate build directory:

```sh
cmake -S ../kodama-cpp -B ../kodama-cpp/build-cuda -DKODAMA_ENABLE_CUDA=ON
cmake --build ../kodama-cpp/build-cuda -j
```

For a native Metal build on macOS:

```sh
cmake -S ../kodama-cpp -B ../kodama-cpp/build-metal \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=ON
cmake --build ../kodama-cpp/build-metal -j
```

## Install The R Wrapper

The wrapper needs two paths:

- `KODAMA_CPP_ROOT`: directory containing `include/kodama/kodama.hpp`.
- `KODAMA_CPP_BUILD_DIR`: directory containing `libkodama_cpp.a`,
  `libkodama_cpp.so`, or `libkodama_cpp.dylib`.

CPU install from a sibling checkout:

```sh
cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build && pwd)" \
R CMD INSTALL .
```

CPU install from this development checkout:

```sh
cd split-repos/kodama-r
KODAMA_CPP_ROOT="$(cd ../.. && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../../build && pwd)" \
R CMD INSTALL .
```

During configuration, the wrapper records the selected core-library path and
checksums of the core archive and public headers. If any of them changes, both
Rcpp bridge objects are rebuilt and the package is relinked automatically; an
unchanged core retains normal incremental-install behavior. This prevents a
development reinstall from silently loading a wrapper linked against an older
CPU, CUDA, or Metal build. `R CMD INSTALL --preclean` remains useful for a
fully clean diagnostic build, but is not required merely because the selected
core was rebuilt or changed.

CUDA install uses the CUDA-enabled build directory and the runtime library
environment. On a Linux machine with a conda/micromamba CUDA environment:

```sh
export ENV_DIR=/path/to/cuda-runtime-env
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-cuda && pwd)" \
R CMD INSTALL .
```

If extra CUDA libraries are needed by your local static link, provide them
through `KODAMA_R_CUDA_LIBS`:

```sh
export KODAMA_R_CUDA_LIBS="-lcudart -lcublas -lcusolver -lcusparse"
```

The package `configure` script writes `src/Makevars` during installation. It
uses only library directories that exist on the current machine, which keeps
local checks quiet while still allowing CUDA/conda paths on GPU hosts. On
macOS it links the Metal, Metal Performance Shaders, and Foundation frameworks.

Metal installation uses the Metal-enabled core build:

```sh
cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-metal && pwd)" \
R CMD INSTALL .
```

## Verify The Installation

Start R and check the linked runtime:

```r
library(kodamaR)
KODAMA.diagnostics()
```

Run a small CPU smoke test:

```r
set.seed(1)
x <- matrix(rnorm(120 * 8), 120, 8)
lab <- rep(1:3, length.out = nrow(x))

cv <- KNNCV(x, lab, folds = 3, k = 5, backend = "cpu")
cv$accuracy

pc <- KODAMA.pca(x, ncomp = 3, backend = "cpu")
dim(pc$scores)

kk <- KODAMA.matrix(
  x,
  classifier = "knn",
  backend = "cpu",
  M = 2,
  Tcycle = 2,
  landmarks = 80,
  progress = FALSE
)

KODAMA.timing(kk)
kk$landmark_seconds
head(kk$best_labels)

prepared <- KODAMA.graph(
  x, k = 30, backend = "cpu", storage = "handle"
)
stopifnot(is.null(prepared$data))
kk_from_graph <- KODAMA.matrix(
  graph = prepared,
  M = 2,
  Tcycle = 2,
  progress = FALSE
)
pls_from_graph_and_x <- KODAMA.matrix(
  data = x,
  graph = prepared,
  classifier = "pls_lda",
  M = 2,
  Tcycle = 2,
  progress = FALSE
)
```

With `storage = "handle"`, the prepared object owns one native float32 graph
without creating R index and distance matrices. It can be passed directly to
`KODAMA.matrix()`, `KODAMA.visualization()`, and `KODAMA.clustering()`; call
`KODAMA.graph.materialize(prepared)` only when R matrices are required.
`storage = "handle"` is the default; request `storage = "matrix"` explicitly
when R arrays are required. Both forms carry
backend-matched UMAP/openTSNE PCA starts and never retain the raw matrix.
External handles are process-local; materialize the graph before saving it for
use in another R session.
`KODAMA.matrix(..., return.graph = FALSE)` returns labels only and omits the
otherwise-unused final graph-distance correction.
`KODAMA.matrix()` reserves `data` for
the raw matrix and `graph` for a prepared graph or bare `indices`/`distances`
list. Either can be supplied alone, or both can be supplied together.

For CUDA verification, switch `backend = "cuda"` after confirming that the R
session can load the same CUDA Toolkit libraries used by the C++ build.
For native KNN, PLS-LDA, Core, matrix, graph, and PCA verification on macOS,
install against the Metal build and use `backend = "metal"`.
For CUDA or Metal `KODAMA.matrix()`, set `n.cores = 0` to let the core select
independent `M` lanes from device capacity and backend-specific workspace
memory. Positive values remain explicit. The returned scheduler fields record
the selected lane count and estimated memory per lane.

## Preprocessing and example manifolds

The wrapper includes the dependency-free preprocessing and synthetic-data
utilities from KODAMA:

```r
x_normalized <- normalization(x, method = "pqn")$newXtrain
x_scaled <- scaling(x_normalized, method = "autoscaling", backend = "cpu",
                    n.cores = 4)$newXtrain
x_scaled <- scaling(x_normalized, method = "autoscaling")$newXtrain

dini <- dinisurface(1000)
helix <- helicoid(1000)
spiral_clusters <- spirals(c(100, 100, 100), c(0.1, 0.1, 0.1))
roll <- swissroll(1000)
```

`normalization()` supports `pqn`, `sum`, `median`, `sqrt`, and `none`.
`scaling()` supports `none`, `centering`, `autoscaling`, `rangescaling`, and
`paretoscaling`. Both are thin calls to the standalone float32 C++ core and
accept `backend = "cpu"`, `"cuda"`, `"metal"`, or `"auto"`; no R numerical
implementation is used.
`scaling()` supports `none`, `centering`, `autoscaling`, `rangescaling`, and
`paretoscaling`. Their signatures and returned fields match the KODAMA R
package.

## Preprocessing and example manifolds

The wrapper includes the dependency-free preprocessing and synthetic-data
utilities from KODAMA:

```r
x_normalized <- normalization(x, method = "pqn")$newXtrain
x_scaled <- scaling(x_normalized, method = "autoscaling", backend = "cpu",
                    n.cores = 4)$newXtrain

dini <- dinisurface(1000)
helix <- helicoid(1000)
spiral_clusters <- spirals(c(100, 100, 100), c(0.1, 0.1, 0.1))
roll <- swissroll(1000)
```

`normalization()` supports `pqn`, `sum`, `median`, `sqrt`, and `none`.
`scaling()` supports `none`, `centering`, `autoscaling`, `rangescaling`, and
`paretoscaling`. Both are thin calls to the standalone float32 C++ core and
accept `backend = "cpu"`, `"cuda"`, `"metal"`, or `"auto"`; no R numerical
implementation is used, and their signatures and returned fields match the
KODAMA R package.

## Run `R CMD check`

The CRAN-style source-tarball procedure is documented once in
[`inst/INSTALL.md`](inst/INSTALL.md#cran-style-local-check). It compiles the
Rcpp bridge, links `libkodama_cpp`, loads the package, runs the testthat suite,
and builds the manual.

## Recommended Workflow

```r
library(kodamaR)

kk <- KODAMA.matrix(
  x,
  classifier = "knn",
  backend = "cuda",
  M = 100,
  Tcycle = 20,
  knn.k = 30
)

KODAMA.timing(kk)
labels <- kk$best_labels
um <- KODAMA.visualization(kk, "UMAP", k = 30, backend = "cuda")
clu <- KODAMA.clustering(um, n.iterations = 10, random.walk.steps = 4)
```

`KODAMA.matrix()` returns a `kodama_matrix` object. The raw C++ fields are still
available (`res`, `acc`, `knn`, `timing`), and convenience fields include
`best_labels`, `best_run`, `class_counts`, and `parameters`.

The native matrix call builds the full-data graph once before all `M` searches
and stores it in `knn`. It also performs one backend-native float32 PCA by
default and derives both UMAP and openTSNE starts from the same scores.
`KODAMA.visualization()` reuses these artifacts without another graph or PCA
calculation. The observable fields are `graph_builds` and
`timing$visual_init_seconds`. Only `knn` is serialized; the former duplicate
`base_knn` payload has been removed. `knn_is_kodama_corrected` and
`graph_storage_bytes` expose graph state and retained native capacity. An
accelerator call returns labels and scores without materializing resident graph
matrices by default. Set `return.graph = TRUE` when the result will be passed
to `KODAMA.visualization()` or the graph matrices are otherwise required.
Explicit `init` still has precedence,
followed by a stored start whose backend matches the requested CPU, CUDA, or Metal
visualization backend, then a start computed from `raw.data`. For example,
after changing backends use:

```r
um_cpu <- KODAMA.visualization(
  kk,
  "UMAP",
  raw.data = x,
  k = 30,
  backend = "cpu"
)
```

If none is available, UMAP uses graph-spectral initialization and openTSNE
uses deterministic random initialization. The result attributes
`initialization` and `initialization_backend` make the chosen path observable.

## Troubleshooting

If installation cannot find the C++ headers, set `KODAMA_CPP_ROOT` explicitly.

If installation cannot find `libkodama_cpp`, build `kodama-cpp` first and set
`KODAMA_CPP_BUILD_DIR` to the CMake build directory.

If loading fails with missing OpenMP or CUDA symbols, start R from
the same shell where `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `CONDA_PREFIX`, and
related runtime variables are set.

If the CUDA backend is unavailable, confirm that `kodama-cpp` was configured
with `-DKODAMA_ENABLE_CUDA=ON` and that the R wrapper was installed against that
CUDA build directory.

If the Metal backend is unavailable, confirm that `kodama-cpp` was configured
with `-DKODAMA_ENABLE_METAL=ON` and that the wrapper was installed against the
same Metal build. UMAP and FFT-grid openTSNE both expose native Metal paths.
