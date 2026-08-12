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
