# Installing KODAMA

The source package vendors the portable MIT-licensed CPU implementation from
`kodama-cpp`. A normal installation requires no external KODAMA system library:

```sh
R CMD INSTALL KODAMA
```

## Optional external accelerator build

CUDA and Metal developers may build `kodama-cpp` separately and set:

- `KODAMA_CPP_ROOT`: directory containing `include/kodama/kodama.hpp`.
- `KODAMA_CPP_BUILD_DIR`: CMake build directory containing `libkodama_cpp`.

Example CPU build and install:

```sh
cmake -S kodama-cpp -B kodama-cpp/build -DKODAMA_ENABLE_CUDA=OFF
cmake --build kodama-cpp/build -j

cd KODAMA
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

cd KODAMA
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
library(KODAMA)
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

cd KODAMA
KODAMA_CPP_ROOT="$(cd ../kodama-cpp && pwd)" \
KODAMA_CPP_BUILD_DIR="$(cd ../kodama-cpp/build-metal && pwd)" \
R CMD INSTALL .
```

Native KNN, PLS-LDA, Core, matrix, graph, PCA, UMAP, and FFT-grid openTSNE
entry points accept `backend = "metal"`.

## CRAN/Bioconductor-style local check

Build a source package and check the tarball:

```sh
R CMD build KODAMA
R CMD check --as-cran KODAMA_0.99.0.tar.gz
Rscript -e 'BiocCheck::BiocCheck("KODAMA_0.99.0.tar.gz")'
```
