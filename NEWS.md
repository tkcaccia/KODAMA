# KODAMA 0.99.7

* Added `RunKODAMAclustering()` for matrices, KODAMA graphs,
  `SingleCellExperiment`, `SpatialExperiment`, Seurat, Giotto, and lists of
  Seurat objects.
* Made `KODAMA.clustering()` a focused adapter to fastEmbedR's Louvain,
  Leiden, and Walktrap implementations, removing competing R-level
  clustering behavior from the KODAMA workflow.
* Cluster memberships are stored in the native observation metadata of each
  supported container, while complete clustering diagnostics are retained in
  KODAMA state where available.

# KODAMA 0.99.6

* Added `RunFastPCA()` as the primary native float32 PCA interface for numeric
  matrices, `SingleCellExperiment`, `SpatialExperiment`, Seurat, and Giotto.
  Container methods select and transpose expression assays internally and
  store PCA scores using each framework's dimensional-reduction interface.
  The lower-level `kodama_pca()` and `KODAMA.pca()` names remain available for
  compatibility.
* Added a session-wide `options(n.cores = ...)` setting, with `N_CORES` as an
  environment fallback, for PCA, graph construction, KODAMA optimization, and
  visualization. CPU-only spatial feature selection follows the same core-count
  setting. Explicit `n.cores` arguments retain priority.
* Preserved sparse assays during spatial feature selection and bounded dense
  C++ work buffers by feature batches.
* Spatial feature-selection container methods now return the updated object,
  store statistics and ranks in feature metadata, and allow
  `RunFastPCA(nfeatures = ...)` to consume the ranking directly.

# KODAMA 0.99.2

* Standardized session-wide backend selection across KODAMA, fastPLS,
  fastEmbedR, and faissR through `options(backend = ...)` and `BACKEND`.
  Legacy KODAMA-specific selectors remain compatibility fallbacks.

# KODAMA 0.99.1

* Established `tkcaccia/KODAMA` as the canonical repository for the new
  C++-backed R package and preserved the classic implementation in
  `tkcaccia/KODAMAlegacy`, aligning the repository and package names required
  for Bioconductor submission.
* Replaced the legacy R implementation with bindings to the standalone
  float32 `kodama-cpp` library.
* Added reusable `KODAMA.graph()`, KNN and PLS-LDA optimization, CPU, CUDA,
  and Metal execution, and graph-input workflows.
* Added adapters for `SingleCellExperiment`, `SpatialExperiment`, Seurat, and
  Giotto objects.
* Retained the historical `MetRef`, `USA`, and `lymphoma` datasets.
* Renamed the former implementation repository to `KODAMAlegacy`; this
  package now owns the canonical `KODAMA` name.
