
#create simulated 200 numbers
vertex = function(vertex=c(0,10),sd=0.2,dims=2, noisy_dimension = 8 ,size_cluster = 50,center=TRUE,scale=TRUE){
  out=as.matrix(vertex)
  cluster_number=2^dims
  if(dims>1){
    for(i in 2:dims){
      nr=nrow(out)
      out=cbind(out,NA)
      out=rbind(out,out)
      
      out[1:nr,i]=vertex[1]
      out[1:nr+nr,i]=vertex[2]
    }
  }
  else{
    out=cbind(out,0)
    dims=2
  }
  out=as.numeric(out)
  if(length(size_cluster)==1) {
    v=matrix(rep(out,each=size_cluster),ncol=dims)
  }
  if(length(size_cluster)==cluster_number) {
    v=matrix(rep(out,rep(size_cluster,dims)),ncol=dims)
  }
  if(length(size_cluster)!=1 & length(size_cluster)==1) {
    stop("The length of size_cluster should be equal to 1 or to the total numer of clusters.")
  }
    
  ma=v+rnorm(length(v),sd = sd)
  if(noisy_dimension>0){
    ma=cbind(ma,matrix(rnorm(nrow(ma)*noisy_dimension),ncol=noisy_dimension))
  }
  ma=scale(ma,center = center,scale = scale)
  ma
}


# This function creates a data set based upon data points distribuited on a Ulisse Dini's surface.
dinisurface = function (N = 1000) 
{
  u = sort(runif(N) * 4 * pi)
  v = runif(N)
  a = 1
  b = 0.2
  x = a * cos(u) * sin(v)
  y = a * sin(u) * sin(v)
  z = a * (cos(v) + log(tan(v/2))) + b * u
  data = cbind(x, y, z)
  return(data)
}



# This function creates a data set based upon data points distribuited on a Helicoid surface.
helicoid = function (N = 1000) 
{
  a = 1
  p = sample((seq(1, -1, length.out = N)))
  t = seq(-pi, pi, length.out = N)
  x = p * cos(a * t)
  y = p * sin(a * t)
  z = t
  data = cbind(x, y, z)
  return(data)
}


# Produces a data set of spiral clusters.

spirals = function (n = c(100, 100, 100), sd = c(0, 0, 0)) 
{
  clusters = length(n)
  x = NULL
  y = NULL
  for (i in 1:clusters) {
    t = seq(1/(4 * pi), 1, length.out = n[i])^0.5 * 2 * pi
    a = rnorm(n[i], sd = sd[i])
    x = c(x, cos(t + (2 * pi * i)/clusters) * (t + a))
    y = c(y, sin(t + (2 * pi * i)/clusters) * (t + a))
  }
  cbind(x, y)
}



# Computes the Swiss Roll data set of a given number of data points.

swissroll = function (N = 1000) 
{
  n <- 3
  m <- 2
  tt <- sort((3 * pi/2) * (1 + 2 * runif(N)))
  height <- 21 * runif(N)
  x <- tt * cos(tt)
  y <- height
  z <- tt * sin(tt)
  data = cbind(x, y, z)
  return(data)
}











