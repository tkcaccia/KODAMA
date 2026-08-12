# KODAMA 0.99.0

* Replaced the legacy R implementation with bindings to the standalone
  float32 `kodama-cpp` library.
* Added reusable `KODAMA.graph()`, KNN and PLS-LDA optimization, CPU, CUDA,
  and Metal execution, and graph-input workflows.
* Added adapters for `SingleCellExperiment`, `SpatialExperiment`, Seurat, and
  Giotto objects.
* Retained the historical `MetRef`, `USA`, and `lymphoma` datasets.
* Renamed the former implementation repository to `KODAMAlegacy`; this
  package now owns the canonical `KODAMA` name.
