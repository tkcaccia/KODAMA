config.tsne.default <- list(
  dims = 2,
  perplexity = 30,
  theta = 0.5,
  max_iter = 1000,
  verbose = getOption("verbose", FALSE),
  Y_init = NULL,
  momentum = 0.5,
  final_momentum = 0.8,
  eta = 200,
  exaggeration_factor = 12,
  num_threads = 1,
  define.n.cores = FALSE
)
class(config.tsne.default) <- "tsne.config"


config.umap.default <- list(
           n_neighbors= 15,
          n_components= 2,
                metric= "euclidean",
              n_epochs= 200,
                 input= "data",
                  init= "spectral",
              min_dist= 0.1,
      set_op_mix_ratio= 1,
    local_connectivity= 1,
             bandwidth= 1,
                 alpha= 1,
                 gamma= 1,
  negative_sample_rate= 5,
                     a= NA,
                     b= NA,
                spread= 1,
          random_state= NA,
       transform_state= NA,
                   knn= NA,
           knn_repeats= 1,
               verbose= FALSE,
       umap_learn_args= NA,
            n_threads = NULL,
        n_sgd_threads = 0,
      define.n.cores = FALSE
)
class(config.umap.default) <- "umap.config"



MDS.defaults <- list(
  dims = 2
)
class(MDS.defaults) <- "MDS.config"

print.tsne.config <- function(x, ...) {
  if (!is(x, "tsne.config")) {
    stop("x is not a tsne configuration object")
  }
  
  # produce a string of form "  z:  " of total length width
  padspaces <- function(z, width=24) {
    padleft <- max(0, width-nchar(z)-2)
    paste(c(rep(" ", padleft), z, ": "), collapse="")
  }
  
  message("t-SNE configuration parameters")
  primitives <- c("numeric", "integer", "character", "logical")
  vapply(names(x), function(z) {
    zval <- x[[z]]
    if (sum(class(zval) %in% primitives)) {
      message(padspaces(z), paste(zval, collapse=" "))
    } else {
      message(padspaces(z), "[", paste(class(zval), collapse=","), "]")
    }
    z
  }, character(1))
  
  invisible(x)
}


print.MDS.config <- function(x, ...) {
  if (!is(x, "MDS.config")) {
    stop("x is not a MDS configuration object")
  }
  
  # produce a string of form "  z:  " of total length width
  padspaces <- function(z, width=24) {
    padleft <- max(0, width-nchar(z)-2)
    paste(c(rep(" ", padleft), z, ": "), collapse="")
  }
  
  message("MDS configuration parameters")
  primitives <- c("numeric", "integer", "character", "logical")
  vapply(names(x), function(z) {
    zval <- x[[z]]
    if (sum(class(zval) %in% primitives)) {
      message(padspaces(z), paste(zval, collapse=" "))
    } else {
      message(padspaces(z), "[", paste(class(zval), collapse=","), "]")
    }
    z
  }, character(1))
  
  invisible(x)
}

kabsch <- function(pm, qm) {
  pm_dims <- dim(pm)
  if (!all(dim(qm) == pm_dims)) {
    stop(call. = TRUE, "Point sets must have the same dimensions")
  }
  # The rotation matrix will have (ncol - 1) leading ones in the diagonal
  diag_ones <- rep(1, pm_dims[2] - 1)
  
  # center the points
  pm <- scale(pm, center = TRUE, scale = FALSE)
  qm <- scale(qm, center = TRUE, scale = FALSE)
  
  am <- crossprod(pm, qm)
  
  svd_res <- svd(am)
  # use the sign of the determinant to ensure a right-hand coordinate system
  d <- determinant(tcrossprod(svd_res$v, svd_res$u))$sign
  dm <- diag(c(diag_ones, d))
  
  # rotation matrix
  um <- svd_res$v %*% tcrossprod(dm, svd_res$u)
  
  # Rotate and then translate to the original centroid location of qm
  sweep(t(tcrossprod(um, pm)), 2, -attr(qm, "scaled:center"))
}



.kodama_irlba_pca <- function(x, nv = 50L, maxit = 1000L, tol = 1e-5) {
  x = as.matrix(x)
  if (!is.numeric(x)) {
    stop("x must be a numeric matrix")
  }
  if (sum(!is.finite(x)) > 0) {
    stop("x contains non-finite values")
  }
  nr = nrow(x)
  nc = ncol(x)
  k = as.integer(min(max(1L, as.integer(nv)), nr, nc))

  # irlb() in vendored C requires m,n >= 4 and nv < min(m, n).
  if (nr < 4L || nc < 4L || k >= min(nr, nc)) {
    sv = svd(x, nu = k, nv = k)
    return(list(u = sv$u, d = sv$d[seq_len(k)], v = sv$v))
  }

  out = try(
    .Call(
      "KODAMA_irlba_pca_dense",
      x,
      as.integer(k),
      as.integer(maxit),
      as.numeric(tol),
      PACKAGE = "KODAMA"
    ),
    silent = TRUE
  )
  if (inherits(out, "try-error")) {
    sv = svd(x, nu = k, nv = k)
    return(list(u = sv$u, d = sv$d[seq_len(k)], v = sv$v))
  }
  out
}

pca = function(x, nv = min(50L, ncol(x)), ...) {
  pca_results = .kodama_irlba_pca(x, nv = nv)
  scores = sweep(pca_results$u, 2, pca_results$d, "*")
  sdev = pca_results$d / sqrt(max(1, nrow(scores) - 1))

  var_expl = pca_results$d^2 / sum(pca_results$d^2)
  ss = sprintf("%.1f", var_expl * 100)
  txt = paste("PC", seq_along(ss), " (", ss, "%)", sep = "")

  res = list(
    sdev = sdev,
    rotation = pca_results$v,
    center = FALSE,
    scale = FALSE,
    x = scores,
    txt = txt
  )
  class(res) = "prcomp"
  colnames(res$x) = txt
  colnames(res$rotation) = txt
  res
}


quality_control = function(data_row,data_col,spatial_row=NULL,data=NULL,f.par.pls){
  if (!is.null(spatial_row)){
    if (spatial_row!=data_row) 
      stop("The number of spatial coordinates and number of entries do not match.")    

  } 

  if (f.par.pls > data_col) {
    message("The number of components selected for PLS-DA is too high and it will be automatically reduced to ", data_col)
    f.par.pls = data_col
  }
  if (f.par.pls > data_row) {
    message("The number of components selected for PLS-DA is too high and it will be automatically reduced to ", data_row)
    f.par.pls = data_row
  }

  
  return(list(f.par.pls=f.par.pls))
}


                              
#' Knowledge Discovery by Accuracy Maximization
#'
#' Run KODAMA on a data matrix and return the optimized labels together with the
#' neighborhood structure used for downstream visualization.
#'
#' @param data Numeric matrix, rows are samples and columns are variables.
#' @param spatial Optional numeric matrix with spatial coordinates.
#' @param samples Optional sample identity vector used to horizontally separate
#'   multiple spatial samples before clustering.
#' @param M Number of independent KODAMA runs.
#' @param Tcycle Number of optimization cycles for each run.
#' @param ncomp Number of PLS components.
#' @param W Optional starting labels for semi-supervised initialization.
#' @param metrics Distance metric used by `Rnanoflann::nn`.
#' @param constrain Optional constraint vector. Samples with identical values are
#'   forced to keep a common label in each run.
#' @param fix Optional logical vector marking entries in `W` that must remain fixed.
#' @param landmarks Number of landmark clusters used in each run.
#' @param splitting Number of clusters used for initialization when `W` is `NULL`.
#' @param spatial.resolution Fraction of landmarks used to define spatial clusters.
#' @param n.cores Number of worker processes. On Unix, `mclapply` is used so
#'   read-only matrices are shared via copy-on-write. On Windows, PSOCK workers
#'   are used and data are copied to workers.
#' @param ancestry Logical; if `TRUE`, use ancestry-aware spatial processing.
#' @param seed Random seed.
#'
#' @return A list with:
#' \itemize{
#'   \item `acc`: final cross-validated accuracy for each run.
#'   \item `v`: accuracy trace matrix (`M x Tcycle`).
#'   \item `res`: label matrix (`M x nrow(data)`).
#'   \item `knn_Rnanoflann`: nearest-neighbor index and distance structure.
#'   \item `data`: input data matrix.
#'   \item `res_constrain`: effective constraints used in each run.
#'   \item `n.cores`: number of cores used in `KODAMA.matrix`.
#' }
#' @export
KODAMA.matrix =
function (data,
          spatial = NULL,
          samples = NULL,
          M = 100, Tcycle = 20,
          ncomp = min(c(50, ncol(data))),
          W = NULL, metrics = "euclidean",
          constrain = NULL, fix = NULL, landmarks = 10000,
          splitting = ifelse(nrow(data) < 40000, 100, 300),
          spatial.resolution = 0.3,
          n.cores = 1,
          ancestry = FALSE,
          seed = 1234,
          ...)
{
  dots = list(...)
  if ("FUN" %in% names(dots)) {
    message("`FUN` is deprecated and ignored. PLS backend is selected automatically from `ncomp` and class count.")
  }
  data = as.matrix(data)
  if (!is.numeric(data)) {
    stop("data must be a numeric matrix")
  }
  if (sum(is.na(data)) > 0) {
    stop("Missing values are present")
  }
  nsample = nrow(data)
  nvariable = ncol(data)
  if (nsample < 2L || nvariable < 1L) {
    stop("data must contain at least 2 rows and 1 column")
  }

  if (!is.null(spatial)) {
    spatial = as.matrix(spatial)
    if (!is.numeric(spatial)) {
      stop("spatial must be a numeric matrix")
    }
    if (nrow(spatial) != nsample) {
      stop("The number of spatial coordinates and number of entries do not match.")
    }
  }

  n.cores = max(1L, as.integer(n.cores))
  set.seed(seed)
  f.par.pls = ncomp
  neighbors = max(1L, floor(min(c(landmarks, nsample * 0.75 - 1), 500)))
  nn_call = function(x, y, k, method) {
    if (n.cores > 1L) {
      out = try(
        Rnanoflann::nn(x, y, k, method = method, parallel = TRUE, cores = n.cores),
        silent = TRUE
      )
      if (!inherits(out, "try-error")) {
        return(out)
      }
    }
    Rnanoflann::nn(x, y, k, method = method)
  }

  writeLines("Calculating Feature Network...")
  knn_Rnanoflann = nn_call(data, data, neighbors + 1, metrics)
  dist_name = if (!is.null(knn_Rnanoflann$distances)) "distances" else "distance"
  if (is.null(knn_Rnanoflann[[dist_name]])) {
    stop("Rnanoflann::nn did not return distances")
  }
  knn_Rnanoflann$distances = knn_Rnanoflann[[dist_name]][, -1, drop = FALSE]
  knn_Rnanoflann$indices = knn_Rnanoflann$indices[, -1, drop = FALSE]

  spatial_flag = !is.null(spatial)
  spatial_jitter = NULL
  if (spatial_flag) {
    writeLines("\nCalculating Spatial Network..")
    knn_spatial = nn_call(spatial, spatial, neighbors, "euclidean")
    idx_far = min(20L, ncol(knn_spatial$indices))
    spatial_jitter = colMeans(abs(
      spatial[knn_spatial$indices[, 1], , drop = FALSE] -
      spatial[knn_spatial$indices[, idx_far], , drop = FALSE]
    )) * 3

    if (!is.null(samples)) {
      samples_names = names(table(samples))
      if (length(samples_names) > 1L) {
        ma = 0
        for (j in seq_along(samples_names)) {
          sel = samples_names[j] == samples
          spatial[sel, 1] = spatial[sel, 1] + ma
          ran = range(spatial[sel, 1])
          ma = ran[2] + stats::dist(ran)[1] * 0.5
        }
      }
    }
  }

  if (is.null(fix)) {
    fix = rep(FALSE, nsample)
  }
  if (length(fix) != nsample) {
    stop("fix must have length nrow(data)")
  }
  fix = as.logical(fix)

  if (is.null(constrain)) {
    constrain = seq_len(nsample)
  }
  if (length(constrain) != nsample) {
    stop("constrain must have length nrow(data)")
  }
  is.na.constrain = is.na(constrain)
  if (any(is.na.constrain)) {
    constrain = as.numeric(as.factor(constrain))
    constrain[is.na.constrain] = max(constrain, na.rm = TRUE) +
      seq_len(sum(is.na.constrain))
  } else {
    constrain = as.numeric(as.factor(constrain))
  }

  if (!is.null(W) && length(W) != nsample) {
    stop("W must have length nrow(data)")
  }

  if (nsample <= landmarks) {
    landmarks = ceiling(nsample * 0.75)
  }
  landmarks = max(2L, as.integer(landmarks))
  splitting = max(2L, as.integer(splitting))
  nspatialclusters = max(1L, round(landmarks * spatial.resolution))

  QC = quality_control(
    data_row = nsample,
    data_col = nvariable,
    spatial_row = if (spatial_flag) nrow(spatial) else NULL,
    data = data,
    f.par.pls = f.par.pls
  )
  f.par.pls = QC$f.par.pls

  one_iteration = function(k) {
    set.seed(seed + k)

    landpoints = integer(landmarks)
    clust_landmarks = as.numeric(stats::kmeans(data, landmarks)$cluster)
    for (ii in seq_len(landmarks)) {
      ww = which(clust_landmarks == ii)
      landpoints[ii] = ww[sample.int(length(ww), 1L)]
    }

    Tdata = data[-landpoints, , drop = FALSE]
    Xdata = data[landpoints, , drop = FALSE]
    Tfix = fix[-landpoints]
    Xfix = fix[landpoints]

    if (spatial_flag) {
      if (ancestry) {
        ethnicity = as.integer(factor(apply(spatial, 1, function(x) paste(x, collapse = "@"))))
        res_move = move_clusters_harmonic_repulsive(
          spatial,
          ethnicity, k = 3, weight = "inv_dist2", lambda = 2,
          p_repulse = 1, r0 = 10, repel_set = "all",
          eta = 0.01, tol = 1e-04, verbose = FALSE
        )
        eq = equalize_within_between(
          res_move$xy,
          ethnicity,
          within_target = "median", between_target_ratio = 2
        )
        spatialclusters = as.numeric(stats::kmeans(eq$xy, nspatialclusters)$cluster)
      } else {
        delta = matrix(0, nrow = nsample, ncol = ncol(spatial))
        for (jj in seq_len(ncol(spatial))) {
          delta[, jj] = stats::runif(nsample, -spatial_jitter[jj], spatial_jitter[jj])
        }
        spatialclusters = as.numeric(stats::kmeans(spatial + delta, nspatialclusters)$cluster)
      }

      ta_const = table(spatialclusters)
      ta_const = ta_const[ta_const > 1]
      sel_cluster_1 = spatialclusters %in% as.numeric(names(ta_const))
      if (sum(!sel_cluster_1) > 0) {
        spatialclusters[!sel_cluster_1] =
          spatialclusters[sel_cluster_1][
            Rnanoflann::nn(
              spatial[sel_cluster_1, , drop = FALSE],
              spatial[!sel_cluster_1, , drop = FALSE],
              1
            )$indices
          ]
      }
      constrain_clean = numeric(nsample)
      for (ic in seq_len(max(constrain))) {
        sel_ic = ic == constrain
        constrain_clean[sel_ic] = as.numeric(names(which.max(table(spatialclusters[sel_ic]))))
      }
    } else {
      constrain_clean = constrain
    }

    Xconstrain = as.numeric(as.factor(constrain_clean[landpoints]))
    if (!is.null(W)) {
      SV_startingvector = W[landpoints]
      unw = unique(SV_startingvector)
      unw = unw[!is.na(unw)]
      ghg = is.na(SV_startingvector)
      SV_startingvector[ghg] = as.numeric(as.factor(SV_startingvector[ghg])) + length(unw)
      XW = numeric(length(Xconstrain))
      for (ic in seq_len(max(Xconstrain))) {
        XW[ic == Xconstrain] =
          as.numeric(names(which.max(table(SV_startingvector[ic == Xconstrain]))))
      }
    } else if (landmarks < 200) {
      XW = Xconstrain
    } else {
      clust_init = as.numeric(stats::kmeans(Xdata, splitting)$cluster)
      XW = numeric(length(Xconstrain))
      for (ic in seq_len(max(Xconstrain))) {
        XW[ic == Xconstrain] =
          as.numeric(names(which.max(table(clust_init[ic == Xconstrain]))))
      }
    }

    clbest = XW
    old_warn = getOption("warn")
    options(warn = -1)
    on.exit(options(warn = old_warn), add = TRUE)
    yatta = structure(0, class = "try-error")
    while (!is.null(attr(yatta, "class"))) {
      yatta = try(
        core_cpp(Xdata, Tdata, clbest, Tcycle, f.par.pls = f.par.pls, Xconstrain, Xfix),
        silent = FALSE
      )
    }

    res_k = rep(NA_real_, nsample)
    vect_acc_k = rep(NA_real_, Tcycle)
    acc_k = NA_real_
    if (is.list(yatta)) {
      clbest = as.vector(yatta$clbest)
      acc_k = yatta$accbest
      vect_acc_k = as.vector(yatta$vect_acc)
      vect_acc_k[vect_acc_k == -1] = NA

      vect_proj = as.vector(yatta$vect_proj)
      if (!is.null(W)) {
        vect_proj[Tfix] = W[-landpoints][Tfix]
      }

      res_k[landpoints] = clbest
      res_k[-landpoints] = vect_proj
      res_k_temp = numeric(nsample)
      for (ic in seq_len(max(constrain_clean))) {
        res_k_temp[ic == constrain_clean] =
          as.numeric(names(which.max(table(res_k[ic == constrain_clean]))))
      }
      res_k = res_k_temp
    }

    list(
      res_k = res_k,
      constrain_k = constrain_clean,
      vect_acc_k = vect_acc_k,
      acc_k = acc_k
    )
  }

  run_parallel_chunks = function(indices, fun, title) {
    writeLines(title)
    pb = txtProgressBar(min = 0, max = length(indices), style = 3)
    on.exit(close(pb), add = TRUE)

    chunk_size = max(1L, min(16L, ceiling(length(indices) / max(1L, n.cores * 4L))))
    chunks = split(indices, ceiling(seq_along(indices) / chunk_size))
    out = vector("list", length(indices))
    done = 0L

    if (n.cores <= 1L) {
      for (chunk in chunks) {
        chunk_res = lapply(chunk, fun)
        out[chunk] = chunk_res
        done = done + length(chunk)
        setTxtProgressBar(pb, done)
      }
      return(out)
    }

    if (.Platform$OS.type != "windows") {
      for (chunk in chunks) {
        chunk_res = parallel::mclapply(
          chunk,
          fun,
          mc.cores = n.cores,
          mc.preschedule = FALSE
        )
        out[chunk] = chunk_res
        done = done + length(chunk)
        setTxtProgressBar(pb, done)
      }
      return(out)
    }

    cl = parallel::makeCluster(n.cores, type = "PSOCK")
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterSetRNGStream(cl, seed)
    parallel::clusterExport(cl, varlist = c(
      "data", "nsample", "fix", "spatial", "spatial_flag", "ancestry",
      "spatial_jitter", "nspatialclusters", "constrain", "W", "landmarks",
      "splitting", "seed", "Tcycle", "f.par.pls", "one_iteration",
      "neighbors"
    ), envir = environment())
    for (chunk in chunks) {
      chunk_res = parallel::parLapplyLB(cl, chunk, fun)
      out[chunk] = chunk_res
      done = done + length(chunk)
      setTxtProgressBar(pb, done)
    }
    out
  }

  idx_M = seq_len(M)
  iter_res = run_parallel_chunks(idx_M, one_iteration, "Running KODAMA optimization...")
  res = matrix(NA_real_, nrow = M, ncol = nsample)
  res_constrain = matrix(NA_real_, nrow = M, ncol = nsample)
  vect_acc = matrix(NA_real_, nrow = M, ncol = Tcycle)
  accu = rep(NA_real_, M)
  for (k in idx_M) {
    res[k, ] = iter_res[[k]]$res_k
    res_constrain[k, ] = iter_res[[k]]$constrain_k
    vect_acc[k, ] = iter_res[[k]]$vect_acc_k
    accu[k] = iter_res[[k]]$acc_k
  }

  one_dissimilarity_row = function(k) {
    knn_indices = knn_Rnanoflann$indices[k, ]
    knn_distances = knn_Rnanoflann$distances[k, ]
    for (j_tsne in seq_len(neighbors)) {
      kod_tsne = mean(res[, k] == res[, knn_indices[j_tsne]], na.rm = TRUE)
      knn_distances[j_tsne] = (1 + knn_distances[j_tsne]) / (kod_tsne^2)
    }
    oo_tsne = order(knn_distances)
    list(
      knn_indices = knn_indices[oo_tsne],
      knn_distances = knn_distances[oo_tsne]
    )
  }

  row_idx = seq_len(nsample)
  dis_res = run_parallel_chunks(
    row_idx,
    one_dissimilarity_row,
    "Calculation of dissimilarity matrix..."
  )
  for (k in row_idx) {
    knn_Rnanoflann$indices[k, ] = dis_res[[k]]$knn_indices
    knn_Rnanoflann$distances[k, ] = dis_res[[k]]$knn_distances
  }

  knn_Rnanoflann$neighbors = neighbors
  return(list(
    acc = accu,
    v = vect_acc,
    res = res,
    knn_Rnanoflann = knn_Rnanoflann,
    data = data,
    res_constrain = res_constrain,
    n.cores = n.cores
  ))
}

                            



                              
KODAMA.visualization=function(kk,method=c("UMAP","t-SNE"),config=NULL){
  
  mat=c("UMAP","t-SNE","MDS")[pmatch(method,c(c("UMAP","t-SNE")))[1]]
  kk_ncores = if (!is.null(kk$n.cores)) as.integer(kk$n.cores) else 1L
  kk_ncores = max(1L, kk_ncores)

  if(mat=="t-SNE"){ 
    if(is.null(config)){
      config = config.tsne.default
    }
    if (is.null(config$define.n.cores)) {
      config$define.n.cores = FALSE
    }
    if (!isTRUE(config$define.n.cores)) {
      config$num_threads = kk_ncores
    }
    if(config$perplexity>(floor(nrow(kk$data)/3)-1)){
      stop("Perplexity is too large for the number of samples")
    }

    ntsne=min(c(round(config$perplexity)*3,nrow(kk$data)-1,ncol(kk$knn_Rnanoflann$indices)))

    if(is.null(config$stop_lying_iter)){
      config$stop_lying_iter = ifelse(is.null(config$Y_init), 250L, 0L)
    }

    if(is.null(config$mom_switch_iter)){
      config$mom_switch_iter = ifelse(is.null(config$Y_init), 250L, 0L)
    }
    res_tsne=Rtsne_neighbors(kk$knn_Rnanoflann$indices[,1:ntsne],kk$knn_Rnanoflann$distances[,1:ntsne],
                             dims = config$dims,
                             perplexity = config$perplexity,
                             theta = config$theta,
                             max_iter = config$max_iter,
                             verbose = config$verbose,
                             Y_init = config$Y_init,
                             stop_lying_iter = config$stop_lying_iter,
                             mom_switch_iter = config$mom_switch_iter,
                             momentum = config$momentum,
                             final_momentum = config$final_momentum,
                             eta = config$eta,
                             exaggeration_factor = config$exaggeration_factor,
                             num_threads = config$num_threads)
  dimensions=res_tsne$Y
  #res_tsne=within(res_tsne, rm(Y))
  res_tsne=res_tsne[names(res_tsne)!="Y"]
    colnames(dimensions)[1:config$dims] = paste ("Dimension", 1:config$dims)
    rownames(dimensions)=rownames(kk$data)
    
  }

  if(mat=="UMAP"){ 
    if(is.null(config)){
      config = config.umap.default
    }
    if (is.null(config$define.n.cores)) {
      config$define.n.cores = FALSE
    }
    if (!isTRUE(config$define.n.cores)) {
      config$n_threads = kk_ncores
      config$n_sgd_threads = kk_ncores
    }
    numap=min(c(round(config$n_neighbors)*3,nrow(kk$data)-1,ncol(kk$knn_Rnanoflann$indices)))

    
    u=umap.knn(kk$knn_Rnanoflann$indices[,1:numap],kk$knn_Rnanoflann$distances[,1:numap])
    u$distances[u$distances==Inf]=max(u$distances[u$distances!=Inf])
    config$knn=u
   
    dimensions = umap(kk$data,knn=u,config=config,n_sgd_threads=config$n_sgd_threads,n_threads=config$n_threads)$layout
    colnames(dimensions)[1:config$n_components] = paste ("Dimension", 1:config$n_components)
    rownames(dimensions)=rownames(kk$data)
  }
  dimensions 
}




mcplot = function (model){
  A=model$v
  A[,1]=0
  plot(A[1,],type="l",xlim=c(1,ncol(model$v)),ylim=c(0,1),xlab="Numer of interatation",ylab="Accuracy")
  for(i in 1:nrow(A))
      points(A[i,],type="l")
}


                              

core_cpp <- function(x, 
                     xTdata=NULL,
                     clbest, 
                     Tcycle=20, 
                     f.par.pls = 5,
                     constrain=NULL, 
                     fix=NULL,
                     ...) {

  dots = list(...)
  if ("FUN" %in% names(dots)) {
    message("`FUN` is deprecated and ignored. PLS backend is selected automatically from `f.par.pls` and class count.")
  }
  QC=quality_control(data_row = nrow(x),
                     data_col = ncol(x),
                     f.par.pls = f.par.pls)

  f.par.pls=QC$f.par.pls
  n_class = length(unique(clbest[!is.na(clbest)]))
  if (n_class < 1L) {
    stop("clbest must contain at least one non-missing class label")
  }
  matchFUN = if (f.par.pls < n_class) 1L else 2L
  
  if (is.null(constrain)) 
    constrain = 1:length(clbest)
  
  if (is.null(fix)) 
    fix = rep(FALSE, length(clbest))
  if(is.null(xTdata)){
    xTdata=matrix(1,ncol=1,nrow=1)
    proj=1
  }else{
    proj=2
  }
# shake=FALSE
  out=corecpp(x, xTdata,clbest, Tcycle, matchFUN, f.par.pls, constrain, fix, FALSE,proj)
  return(out)
}










 
move_clusters_harmonic_repulsive <- function(
    xy, label,
    k = 5,                               # number of attractive neighbors (k-NN)
    weight = c("uniform","inv_dist","inv_dist2","gaussian"),
    p_attract = 2,                       # power for inv_dist (if chosen)
    sigma = NULL,                        # bandwidth for gaussian
    # Repulsion settings:
    lambda = 1.0,                        # strength of repulsion
    p_repulse = 2,                       # r^{-p} barrier (>=2 recommended)
    r0 = 1.0,                            # distance scale for barrier
    repel_set = c("all","outside_knn","outside_kplus1"), # which neighbors repel
    # Optim & convergence:
    eta = 0.7,                           # gradient step size for centroids
    max_iter = 200, tol = 1e-3, verbose = FALSE,
    grad_clip = 5                        # clip very large gradients (stability)
){
  if (!is.matrix(xy)) xy <- as.matrix(xy)
  stopifnot(ncol(xy) >= 2, length(label) == nrow(xy))
  if (!is.factor(label)) label <- factor(label)
  labs <- levels(label); G <- length(labs); if (G < 2) stop("Need at least 2 clusters.")
  
  weight <- match.arg(weight)
  repel_set <- match.arg(repel_set)
  
  # helpers
  get_centroids <- function(xy, label, labs) {
    do.call(cbind, lapply(labs, function(l)
      colMeans(xy[label == l, , drop = FALSE])))
  }
  build_w <- function(d, scheme, p, sigma){
    if (scheme == "uniform") {
      w <- rep(1, length(d))
    } else if (scheme == "inv_dist") {
      eps <- .Machine$double.eps
      w <- 1 / pmax(d, eps)^p
    } else if (scheme == "inv_dist2") {
      eps <- .Machine$double.eps
      w <- 1 / pmax(d, eps)^2
    } else { # gaussian
      if (is.null(sigma) || !is.finite(sigma) || sigma <= 0) {
        sigma <- stats::median(d[is.finite(d) & d > 0]); 
        if (!is.finite(sigma) || sigma <= 0) sigma <- mean(d[d > 0])
        if (!is.finite(sigma) || sigma <= 0) sigma <- 1
      }
      w <- exp(-(d^2) / (2*sigma^2))
    }
    sw <- sum(w); if (sw <= 0) { w[] <- 1; sw <- length(w) }
    w / sw
  }
  
  xy_cur <- xy
  it_done <- 0
  for (it in seq_len(max_iter)) {
    C <- get_centroids(xy_cur, label, labs)     # 2 x G
    D <- as.matrix(dist(t(C))); diag(D) <- Inf  # G x G
    
    shifts <- matrix(0, nrow = 2, ncol = G)
    
    for (g in seq_len(G)) {
      c0 <- C[, g]
      
      # --- Attractive gradient to k-NN (harmonic) ---
      ord <- order(D[g, ])
      m <- min(k, G - 1)
      nbr_idx <- if (m > 0) ord[seq_len(m)] else integer(0)
      if (length(nbr_idx)) {
        d_attr <- D[g, nbr_idx]
        w <- build_w(d_attr, weight, p_attract, sigma)
        muN <- C[, nbr_idx, drop = FALSE] %*% w
        # grad of 1/2 * sum_j wj ||c-muj||^2 = (sum wj) * (c - weighted_mean)
        grad_attr <- as.vector(c0 - muN) * sum(w)
      } else {
        grad_attr <- c(0, 0)
      }
      
      # --- Repulsive gradient from other centroids ---
      # E_rep = lambda * sum_l (r0 / r_l)^p  with r_l = ||c - mu_l||
      # grad = -lambda * p * r0^p * sum_l (c - mu_l) / r_l^{p+2}
      repel_idx <- setdiff(seq_len(G), g)
      if (repel_set == "outside_knn" && length(nbr_idx)) {
        repel_idx <- setdiff(repel_idx, nbr_idx)
      } else if (repel_set == "outside_kplus1" && length(ord) >= (k+1)) {
        repel_idx <- ord[(k+1):(G-1)]
      }
      grad_rep <- c(0, 0)
      if (length(repel_idx)) {
        V <- sweep(C[, repel_idx, drop = FALSE], 1, c0, "-")   # mu_l - c0
        V <- -V                                                # (c0 - mu_l)
        r2 <- colSums(V^2)
        eps <- 1e-12
        r2 <- pmax(r2, eps)
        scale <- (lambda * p_repulse * (r0^p_repulse)) / (r2^((p_repulse + 2)/2))
        grad_rep <- as.vector(V %*% scale)  # sum_l (c0 - mu_l)/r^{p+2} * const
        grad_rep <- -grad_rep               # negative gradient (since we built V as c0-mu_l)
      }
      
      # total gradient of E_g
      grad <- grad_attr + grad_rep
      
      # clip for stability
      gnorm <- sqrt(sum(grad^2))
      if (gnorm > grad_clip) grad <- grad * (grad_clip / gnorm)
      
      # gradient descent step on centroid
      c_new <- c0 - eta * grad
      shifts[, g] <- c_new - c0
    }
    
    max_shift <- max(sqrt(colSums(shifts^2)))
    if (verbose) message(sprintf("iter %d: max centroid shift = %.6f", it, max_shift))
    
    # translate each cluster by its centroid shift
    for (g in seq_len(G)) {
      idx <- label == labs[g]
      if (any(idx)) {
        xy_cur[idx, ] <- sweep(xy_cur[idx, , drop = FALSE], 2, shifts[, g], "+")
      }
    }
    
    it_done <- it
    if (max_shift < tol) break
  }
  
  list(
    xy = xy_cur,
    centers = get_centroids(xy_cur, label, labs),
    iterations = it_done
  )
}

equalize_within_between <- function(xy, cluster,
                                    within_target = c("median", "mean", "value"),
                                    within_value = NULL,
                                    between_target_ratio = 1,
                                    eps = 1e-8) {
  x=xy[,1]+rnorm(nrow(xy),sd = 0.001)
  y=xy[,2]+rnorm(nrow(xy),sd = 0.001)
  
  cluster <- as.factor(cluster)
  within_target <- match.arg(within_target)
  
  ## Per-row cluster centroids
  mu_x <- ave(x, cluster, FUN = function(z) mean(z, na.rm = TRUE))
  mu_y <- ave(y, cluster, FUN = function(z) mean(z, na.rm = TRUE))
  
  ## Within distances and per-cluster median radius
  r_i <- sqrt((x - mu_x)^2 + (y - mu_y)^2)
  r_med_by_cl <- tapply(r_i, cluster, function(z) stats::median(z, na.rm = TRUE))
  
  ## Target within radius r_w
  r_w <- switch(within_target,
                median = stats::median(r_med_by_cl, na.rm = TRUE),
                mean   = mean(r_med_by_cl, na.rm = TRUE),
                value  = { if (is.null(within_value)) stop("Provide `within_value` for within_target='value'.")
                  as.numeric(within_value) })
  
  ## Per-cluster within scaling (guard zeros)
  beta_by_cl <- r_w / pmax(r_med_by_cl, eps)
  beta_row   <- beta_by_cl[ match(cluster, names(beta_by_cl)) ]
  
  ## Unique centroids (one per cluster)
  mu_tab_x <- tapply(x, cluster, mean)
  mu_tab_y <- tapply(y, cluster, mean)
  K <- length(mu_tab_x)
  
  ## Median nearest-centroid distance
  if (K >= 2) {
    D <- as.matrix(dist(cbind(mu_tab_x, mu_tab_y)))
    diag(D) <- Inf
    d_nn <- apply(D, 1, min)          # nearest centroid for each cluster
    d_nn_med <- stats::median(d_nn)
  } else d_nn_med <- NA_real_
  
  ## Global centroid of centroids (keep overall center fixed)
  g_x <- mean(mu_tab_x)
  g_y <- mean(mu_tab_y)
  
  ## Between scaling (alpha): target nearest-centroid distance ratio.
  alpha <- if (is.finite(d_nn_med) && d_nn_med > eps) {
    (between_target_ratio * r_w) / d_nn_med
  } else 1
  
  ## Apply transforms:
  ## 1) move each cluster centroid toward/away from global center by alpha
  mu_x_scaled <- g_x + alpha * (mu_x - g_x)
  mu_y_scaled <- g_y + alpha * (mu_y - g_y)
  ## 2) rescale within-cluster deviations by beta
  x_new <- mu_x_scaled + beta_row * (x - mu_x)
  y_new <- mu_y_scaled + beta_row * (y - mu_y)
  
  xy_new=cbind(x_new,y_new)
  colnames(xy_new)=colnames(xy)
  rownames(xy_new)=rownames(xy)
  list(xy = xy_new,
       r_w = r_w, d_nn_med = d_nn_med,
       alpha = alpha, beta_by_cl = beta_by_cl,
       centers_before = cbind(mu_tab_x, mu_tab_y))
}

##########################################################################
