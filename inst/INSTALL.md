# Installing kodamaR Against kodama-cpp

`kodamaR` is a thin R wrapper. The KODAMA algorithms are compiled in
`libkodama_cpp`; the R package compiles an Rcpp bridge and links to that library
during `R CMD INSTALL`.

## Required Build Inputs

Set these environment variables when installing:

- `KODAMA_CPP_ROOT`: directory containing `include/kodama/kodama.hpp`.
- `KODAMA_CPP_BUILD_DIR`: CMake build directory containing `libkodama_cpp`.

Example CPU build and install:

```sh
cmake -S kodama-cpp -B kodama-cpp/build -DKODAMA_ENABLE_CUDA=OFF
cmake --build kodama-cpp/build -j

cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build && pwd)" \
R CMD INSTALL .
```

Example CUDA install:

```sh
export ENV_DIR=/path/to/cuda-runtime-env
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cmake -S kodama-cpp -B kodama-cpp/build-cuda -DKODAMA_ENABLE_CUDA=ON
cmake --build kodama-cpp/build-cuda -j

cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-cuda && pwd)" \
R CMD INSTALL .
```

If the static CUDA link needs extra libraries, provide them with
`KODAMA_R_CUDA_LIBS`, for example:

```sh
export KODAMA_R_CUDA_LIBS="-lcudart -lcublas -lcusolver -lcusparse"
```

## Runtime Verification

```r
library(kodamaR)
KODAMA.diagnostics()
```

Then run a small CPU smoke test:

```r
set.seed(1)
x <- matrix(rnorm(120 * 8), 120, 8)
lab <- rep(1:3, length.out = nrow(x))
KNNCV(x, lab, folds = 3, k = 5, backend = "cpu")$accuracy
```

For CUDA, repeat with `backend = "cuda"` after confirming the same CUDA Toolkit
runtime paths are visible to the R session.

## Apple Metal Install

On macOS, build the core with Metal enabled and install the wrapper against
that build:

```sh
cmake -S kodama-cpp -B kodama-cpp/build-metal \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=ON
cmake --build kodama-cpp/build-metal -j

cd kodama-r
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-metal && pwd)" \
R CMD INSTALL .
```

Native KNN, PLS-LDA, Core, matrix, graph, and PCA entry points accept
`backend = "metal"`. UMAP and openTSNE currently remain CPU/CUDA operations.

## CRAN-Style Local Check

Build a source package and check the tarball:

```sh
cd split-repos
KODAMA_CPP_ROOT="$(cd .. && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../build && pwd)" \
R CMD build kodama-r

KODAMA_CPP_ROOT="$(cd .. && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../build && pwd)" \
R CMD check --as-cran kodamaR_0.1.0.tar.gz
```
