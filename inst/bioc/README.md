# Bioconductor release preparation

The package metadata, vignette layout, optional S4 dependencies, vendored CPU
core, and testthat suite follow Bioconductor conventions. Before submission:

1. Synchronize `src/` and `inst/include/kodama/` from the release-tagged
   `kodama-cpp` commit and record that commit in release provenance. The source
   tarball already builds without GitHub `Remotes` or a system KODAMA library.
2. Run `R CMD build .`, `R CMD check --as-cran`, and `BiocCheck::BiocCheck()` on
   Linux, macOS, and Windows CPU builds.
3. Keep CUDA and Metal optional. A CPU-only build must install and pass all
   examples and tests without accelerator libraries.
4. Add a public support site and verify all optional object adapters against
   current Bioconductor release versions.
5. Resolve the package-name transition from the existing CRAN `KODAMA` release
   before Bioconductor submission; Bioconductor cannot host a distinct package
   that conflicts with an active CRAN package of the same name.
