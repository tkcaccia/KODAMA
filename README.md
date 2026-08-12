# KODAMA

R interface to the standalone `kodama-cpp` C++/CUDA/Metal library.

The wrapper and numerical core are maintained as separate repositories:

- [`tkcaccia/KODAMA`](https://github.com/tkcaccia/KODAMA)
- [`tkcaccia/kodama-cpp`](https://github.com/tkcaccia/kodama-cpp)

## Quick Install

The source package contains the portable MIT-licensed CPU core and does not
require a separately installed C++ library:

```sh
git clone https://github.com/tkcaccia/KODAMA.git
Rscript -e 'install.packages("Rcpp", repos = "https://cloud.r-project.org")'
R CMD INSTALL KODAMA
```

Then verify the installation with:

```r
library(KODAMA)
KODAMA.diagnostics()
help(package = "KODAMA")
```

The sections below explain optional CUDA, Metal, and development workflows.

`KODAMA` does not reimplement the mathematics in R. It converts R matrices and
vectors to the C++ API and returns R-friendly lists, matrices, and S3 objects.
The vendored CPU sources are synchronized from `kodama-cpp`; the same public
core is shared by the independently maintained R and Python wrappers.

## What Is Linked

The normal source installation compiles the Rcpp bridge and portable CPU core:

```text
R session -> KODAMA R functions -> Rcpp bridge -> vendored kodama-cpp CPU core
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
- `RunKODAMAgraph()`, `RunKODAMAmatrix()`, and
  `RunKODAMAvisualization()` for `SingleCellExperiment`, `SpatialExperiment`,
  Seurat, and Giotto containers.
- `SpatialFeatureSelection()` / `RunSpatialFeatureSelection()` for
  `SpatialExperiment`, Seurat, and Giotto containers. The selector is
  multicore CPU-only.

## Single-Cell Object Workflow

The three object generics mirror the matrix pipeline. Graph construction is
performed once and retained as an opaque native handle; the matrix step reuses
that graph, and only the visualization step creates a KODAMA reduced dimension:

```r
object <- RunKODAMAgraph(
  object,
  reduction = "PCA", # use "pca" for Seurat and Giotto
  dims = 50,
  backend = "cpu"
)
object <- RunKODAMAmatrix(
  object,
  reduction = "PCA",
  dims = 50,
  classifier = "pls_lda",
  backend = "cpu",
  M = 100,
  Tcycle = 100
)
object <- RunKODAMAvisualization(
  object,
  reduction = "PCA",
  dims = 50,
  method = "UMAP",
  backend = "cpu"
)
```

For `SpatialExperiment`, spatial coordinates are used by default and
`colData(object)$sample_id` keeps slides independent. Set
`use.spatial = FALSE` for a data-only analysis or change `sample.column` when
slide identifiers use another column. Seurat spatial methods obtain coordinates
from the image objects and use image names as slide identifiers. Giotto methods
use its selected spatial-location object. Lists of Seurat objects are processed
element by element.

Bioconductor containers retain graph and matrix state under
`metadata(object)$KODAMA$KODAMA`. Seurat stores the same state in
`Misc(object, "KODAMA")$KODAMA`; Giotto stores it in the `misc` field of its
KODAMA dimensional-reduction object. The final coordinates are available from
`reducedDim(object, "KODAMA")`, `Embeddings(object, "KODAMA")`, or the
corresponding Giotto dimensional reduction.

## Prerequisites

The portable package requires R, `Rcpp`, and a C++17 compiler. CMake is needed
only for optional accelerator builds against a separate `kodama-cpp` checkout.

On macOS with Homebrew, the CPU development environment is typically:

```sh
brew install cmake libomp
Rscript -e 'install.packages(c("Rcpp", "testthat"), repos = "https://cloud.r-project.org")'
```

On Linux/CUDA machines, use the same CUDA environment used to build the core.
The important rule is that `R CMD INSTALL` and later R sessions must see the
same CUDA Toolkit libraries used by `kodama-cpp`.

## Optional accelerator installation

From a checkout where `kodama-cpp` and `KODAMA` are siblings:

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

## Link an external core

The portable default does not use environment variables. To opt into a CUDA or
Metal build, provide two paths:

- `KODAMA_CPP_ROOT`: directory containing `include/kodama/kodama.hpp`.
- `KODAMA_CPP_BUILD_DIR`: directory containing `libkodama_cpp.a`,
  `libkodama_cpp.so`, or `libkodama_cpp.dylib`.

CPU install from a sibling checkout:

```sh
cd KODAMA
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build && pwd)" \
R CMD INSTALL .
```

CPU install from this development checkout:

```sh
cd split-repos/KODAMA
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

cd KODAMA
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
cd KODAMA
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-metal && pwd)" \
R CMD INSTALL .
```

## Verify The Installation

Start R and check the linked runtime:

```r
library(KODAMA)
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
Multiple slides can be kept distinct during spatial graph construction with
the original KODAMA `samples` contract:

```r
prepared <- KODAMA.graph(x, spatial = xy, samples = slide_id)
kk <- KODAMA.matrix(
  data = x, graph = prepared, spatial = xy, samples = slide_id,
  classifier = "knn"
)
```

The first spatial coordinate is offset slide by slide before spatial neighbors
and constraints are built. Spatial landmark strata and optimization constraint
IDs are also keyed by slide, so no constraint group can cross a slide boundary.
A one-level `samples` vector is a no-op.

Repeated population collection coordinates can be handled with
`spatial.mode = "population"`. This opt-in mode regularizes identical
latitude/longitude groups independently inside every `M` run using the
classic harmonic attraction/repulsion and within/between-location
equalization. The former `ancestry = TRUE` spelling has been removed; use
`spatial.mode = "population"`. Always report a
genetics-only call without `spatial` alongside the geographic analysis.

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

Spatially variable features can be ranked independently across slides without
SPARK-X or a GPL runtime dependency:

```r
svg <- spatial_feature_selection(
  expression, coordinates, samples = slide_id,
  n.cores = 4
)
head(svg$features)

# Containers provide their own expression values, coordinates, and slide IDs.
svg_spe <- SpatialFeatureSelection(
  spe, assay.type = "logcounts", sample.column = "sample_id", n.cores = 4
)
svg_seurat <- SpatialFeatureSelection(
  seurat_object, assay = "RNA", layer = "data", n.cores = 4
)
svg_giotto <- SpatialFeatureSelection(
  giotto_object, values = "normalized", sample.column = "slide_id",
  n.cores = 4
)
```

The fixed low-rank projection statistic uses a multicore CPU implementation.
Heavy arithmetic is float32; probability tails and BH adjustment are double
precision. CUDA and Metal are intentionally not exposed for this function
because they did not improve end-to-end runtime on the validation workloads.

## Run `R CMD check`

The CRAN-style source-tarball procedure is documented once in
[`inst/INSTALL.md`](inst/INSTALL.md#cran-style-local-check). It compiles the
Rcpp bridge, links `libkodama_cpp`, loads the package, runs the testthat suite,
and builds the manual.

## Recommended Workflow

```r
library(KODAMA)

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
