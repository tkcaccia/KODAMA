# KODAMA 0.99.0

* Added a session-wide backend selector through `KODAMA_backend()`,
  `options(KODAMA.backend = ...)`, and `KODAMA_BACKEND`. Explicit function
  arguments retain precedence and CPU remains the default.
* Replaced the legacy R implementation with bindings to the standalone
  float32 `kodama-cpp` library.
* Added reusable `KODAMA.graph()`, KNN and PLS-LDA optimization, CPU, CUDA,
  and Metal execution, and graph-input workflows.
* Added adapters for `SingleCellExperiment`, `SpatialExperiment`, Seurat, and
  Giotto objects.
* Retained the historical `MetRef`, `USA`, and `lymphoma` datasets.
* Renamed the former implementation repository to `KODAMAlegacy`; this
  package now owns the canonical `KODAMA` name.
