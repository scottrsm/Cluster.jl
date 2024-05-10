var documenterSearchIndex = {"docs":
[{"location":"#Cluster.jl-Documentation","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"CurrentModule = Cluster","category":"page"},{"location":"#Overview","page":"Cluster.jl Documentation","title":"Overview","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"This module contains functions to determine natural clusters using unsupervised learning. A point of interest in this module over other libraries is the rich set of metrics  that one can use with K-means clustering.  Additionally, some of the metrics may be weighted which can be used to help alleviate   K-means attraction to spherical clusters.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"This module also has the functionality to determine the cluster size of a data set.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"Cluster functions (from lowest to highest level):","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"kmeans_cluster: Given a cluster number and metric, determine clusters. \nThis clustering algorithm can use any of the metrics:    L2 (L_2 – default), LP (L_p), LI (L_infty), KL (Kullback-Leibler), CD (Cosine Distance), and JD (Jaccard Distance).\nIn the case of L2 and CD, the metrics allow for weighted distances represented as a positive definite matrix.\nfind_best_info_for_ks: Given a metric and a range of cluster numbers, determine clusters and gather fitness data for each cluster.\nThis function uses the functions:\nkmeans_cluster.\nfind_best_cluster: Given a metric, find the best cluster number and determine the clusters. \nThis function uses the functions:\nkmeans_cluster\nfind_best_info_for_ks\nfind_cluster_map: Takes clusters and a discrete attribute of the data set and maps cluster numbers to attribute values.\nconfusion_matrix: Computes the confusion matrix – comparing actual against predicted values.","category":"page"},{"location":"#Metric-Definitions:","page":"Cluster.jl Documentation","title":"Metric Definitions:","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"Given N vectors, bf x bf y :","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"L2: The standard L_2 norm: rm L2(bf x bf y) = sqrtsum_i=1^N (x_i - y_i)^2\nWith a symmetric, positive semi-definite weight matrix W   this becomes: rm L2(bf x bf y hboxM=W) = sqrtbf x boldsymbol cdot (M bf y)\nLP: The standard L_p norm: rm LP(bf x bf y p) = left(sum_i=1^N x_i - y_i^p)right)^frac1p\nNote: To use this metric with find_best_cluster for a given value of p,    you will need to pass the closure, (x,y; kwargs...) -> LP(x,y, p; kwargs...),   to the keyword parameter dmetric.\nLI: The standard L_infty norm: rm LI(bf x bf y) = mathoprm max_i in 1Nlimits x_i - y_i \nKL: A symmetrized Kullback-Leibler divergence: rm KL(bf x bf y) = sum_i=1^N x_i log(x_iy_i) + y_i log(y_ix_i)\nCD: The \"cosine\" distance: rm CD(bf x bf y) = 1 - bf x boldsymbol cdot bf y  (bf x  bf y)\nWith a symmetric strictly positive definite weight matrix W this becomes:     rm CD(bf x bf y hboxM=W) = 1 - bf x boldsymbol cdot left( M bf yright)  (bf x  bf y) \nHere:  bf z  = sqrtbf z boldsymbol cdot left( M bf zright)\nJD: The Jaccard distance.","category":"page"},{"location":"#What-is-the-best-cluster-number?","page":"Cluster.jl Documentation","title":"What is the best cluster number?","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"The function find_best_cluster attempts to find the best cluster number. To do this, it monitors the total variation as one increases the cluster number. The total variation goes  down (generally) as we find (potentially locally) optimal solutions for each cluster number. If we pick a cluster number using only the total variation, we will miss the \"natural cluster\" number.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"To avoid this, we adjust the total variation by a function that depends on the dimension of the space we are working in as well as the cluster number. The reasoning follows:","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"The idea is to look at the natural rate at which the total variation decreases with cluster number when  there are no clusters. In this way we can adjust the total variation to take into account  this \"ambient\" decay.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"To do this, we start by assuming that the data is uniformly distributed in our domain  (with respect to the metric used) when given the data: k clusters; m points; the domain in n dimensions. We assume that the k clusters have the same number of points and fill a sphere  of radius, R. This means that R^n approx k  r_k^n.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"Solving for r_k we have r_k=Rleft(frac1kright)^frac1n. The total variation of k clusters is then roughly: k  r_kleft(fracmkright).  This becomes: fracm Rk^frac1n. Thus, even in the absence of any true clusters, the total variation decays like k^frac1n.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"The function find_best_cluster compares the total variation of cluster numbers in a range. It chooses the cluster number, k, with the largest relative rate  of decrease (with respect to cluster size) in adjusted total variation. The adjusted variation modifies the total variation for each k by the multiplicative factor  k^frac1n.  The variation is further adjusted by the  fraction of unused cluster centroids. Finally, before computing the relative rate of variation decrease, the  series is further adjusted to be monotonically non-increasing.","category":"page"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"NOTE: This analysis may not be as useful if the \"natural\" clusters (or a substantial subset)  lie in some lower dimensional hyperplane in the ambient space.","category":"page"},{"location":"#Cluster-Functions","page":"Cluster.jl Documentation","title":"Cluster Functions","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"kmeans_cluster(::Matrix{T}, ::Int = 3; ::F = L2, ::Float64 = 1.0e-3,::Union{Nothing, AbstractMatrix{T}} = nothing,::Int = 1000, ::Int=0) where {T <: Real, F <: Function}","category":"page"},{"location":"#Cluster.kmeans_cluster-Union{Tuple{Matrix{T}}, Tuple{F}, Tuple{T}, Tuple{Matrix{T}, Int64}} where {T<:Real, F<:Function}","page":"Cluster.jl Documentation","title":"Cluster.kmeans_cluster","text":"kmeans_cluster(X, k=3[; dmetric, threshold, W, N, seed])\n\nGroups a set of points into k clusters based on the distance metric, dmetric.\n\nType Constraints\n\nT <: Real\nF <: Function\n\nArguments\n\nX::Matrix{T}   : (n,m) Matrix representing m points of dimension n.\nk::Int=3       : The number of clusters to form.\n\nKeyword Arguments\n\ndmetric::F=L2  : The distance metric to use.\nthreshold::Float=1.0e-2  : The relative error improvement threshold (using total variation)\nW::Union{Nothing, AbstractMatrix{T}}=nothing : Optional (nxn) weight matrix for metric.\nN::Int=1000    : The maximum number of iterations to try.\nseed::Int=0    : If value > 0, create a random number generator to use for initial clustering.\n\nInput Contract\n\nW = rm nothing  left( (rm typeof(W) = rm MatrixT)  W in boldsymbol S_++^n right)\n1 le k le m\nN > 0\nthreshold > 0.0\ndmetric <: Function\n\nReturn\n\nA Tuple:\n\nDict{Int, Int}  : Mapping of points (n-vectors) indices to centroid indices.\nMatrix{T}       : (nxk) Matrix representing k centroids of n-vectors.\nFloat64         : The total variation between points and their centroids (using dmetric).\nVector{Int}     : Unused centroids (by index).\nInt             : The number of iterations to use for the algorithm to complete.\nBool            : Did algorithm converge.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"find_best_info_for_ks(::Matrix{T}, ::UnitRange{Int}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int=1000, ::Int=300, ::Int=1) where{T <: Real, F <: Function}","category":"page"},{"location":"#Cluster.find_best_info_for_ks-Union{Tuple{F}, Tuple{T}, Tuple{Matrix{T}, UnitRange{Int64}}} where {T<:Real, F<:Function}","page":"Cluster.jl Documentation","title":"Cluster.find_best_info_for_ks","text":"find_best_info_for_ks(X, kRng[; dmetric=L2, threshold=1.0e-3, W, N=1000, num_trials=100, seed=1])\n\nGroups a set of m points (n-vectors) as an (nxm) matrix, X, into k clusters where k is in the range, kRng. The groupings are determined based on the distance metric, dmetric.\n\nType Constraints\n\nT <: Real\nF <: Function\n\nArguments\n\nX::Matrix{T}           : (n,m) Matrix representing m points of dimension n.\nkRng::UnitRange{Int}   : The number of clusters to form.\n\nKeyword Arguments\n\ndmetric::F=L2          : The distance metric to use.\nthreshold::Float=1.0e-2: The relative error improvement threshold (using total variation)\nW::Union{Nothing, AbstractMatrix{T}}=nothing : Optional Weight matrix for metric.\nN::Int=1000            : The maximum number of kmeans_clustering iterations to try for each cluster number.\nnum_trials::Int=300    : The number of times to run kmeans_clustering for a given cluster number. \nseed::Int=1            : The random seed to use. (Used by kmeans_cluster to do initial clustering.)\n\nInput Contract\n\nW = rm nothing  left( (rm typeof(W) = rm MatrixT)  W in boldsymbol S_++^n right)\nN  0\n i in rm kRng i ge 1\nthreshold > 0.0\n\nReturn\n\nA Tuple with entries:\n\nOrderedDict{Int, Float}         : 1:k -> The Total Variation for each cluster number.\nOrderedDict{Int, Vector{Int}}   : 1:k -> Mapping of index of points (n-vectors in X) to centroid indices.\nOrderedDict{Int, Matrix{T}}     : 1:k -> (nxk) Matrix representing k n-vector centroids.\nOrderedDict{Int, Vector{In64}}  : 1:k -> Vector of unused centroids by index.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"find_best_cluster(::Matrix{T}, ::UnitRange{Int}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int=1000, ::Int=300, ::Int=1, ::Bool=false) where{T <: Real, F <: Function}","category":"page"},{"location":"#Cluster.find_best_cluster-Union{Tuple{F}, Tuple{T}, Tuple{Matrix{T}, UnitRange{Int64}}} where {T<:Real, F<:Function}","page":"Cluster.jl Documentation","title":"Cluster.find_best_cluster","text":"find_best_cluster(X, kRng[; dmetric=L2, threshold=1.0e-3, W, N=1000, num_trials=100, seed=1, verbose=false])\n\nGroups a set of points into the \"best\" number of clusters based on the distance metric, dmetric. It does this by examining the total variation between the points and the centroids for groups of k where k is in the range, kRng. \n\nNOTE: If the value k was determined to be the best cluster number but some of the centroids were not used, then the value of k will be set to the number of centroids that are used and the centroids that were not used will be removed. In this case it may be that the returned value of k is less that any value in the cluster range, kRng.\n\nType Constraints\n\nT <: Real\nF <: Function\n\nArguments\n\nX::Matrix{T}           : (n,m) Matrix representing m points of dimension n.\nkRng::UnitRange{Int}   : The range of potential cluster values to try.\n\nKeyword Arguments\n\ndmetric::F=L2          : The distance metric to use.\nthreshold::Float=1.0e-2: The relative error improvement threshold (using total variation)\nW::Union{Nothing, AbstractMatrix{T}}=nothing : Optional Weight matrix for metric.\nN::Int=1000            : The maximum number of kmeans_clustering iterations to try for each cluster number.\nnum_trials::Int=300    : The number of times to run kmeans_clustering for a given cluster number. \nseed::Int=1            : The random seed to use. (Used by kmeans_cluster to do initial clustering.)\nverbose::Bool=false    : If true, print diagnostic information.\n\nInput Contract\n\nW = rm nothing  left( (rm typeof(W) = rm MatrixT)  W in boldsymbol S_++^n right)\nN > 0\n i in rm kRng i ge 1\nthreshold > 0.0\n\nReturn\n\nA Tuple:\n\nInt           : The \"best\" cluster number, k.\nDict{Int, Int}: Mapping of points (n-vectors) indices to centroid indices.\nMatrix{T}     : Cluster centroids, represented as an (n,k) matrix.\nFloat64       : The total variation between points and their centroids (using dmetric).\n\n\n\n\n\n","category":"method"},{"location":"#Metric-Functions","page":"Cluster.jl Documentation","title":"Metric Functions","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"L2(::AbstractVector{T},::AbstractVector{T}; M=::Union{Nothing, AbstractMatrix{T}} = nothing) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.L2-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.L2","text":"L2(x,y[; M=nothing])\n\nComputes the L_2 distance between two vectors. One of the features that may be different from other packages is the use of weighted metrics in some instances.\n\nType Constraints\n\nT <: Real\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\n\nKeyword Arguments\n\nC::Union{Nothing, Matrix{T} : Optional Weight matrix.\n\nInput Contract (Low level function – Input contract not checked)\n\nbf x = bf y\nM = rm nothing vee left( (rm typeof(M) = rm MatrixT) wedge M in boldsymbol S_++^bf x right)\n\nReturn\n\nL_2 (optionally weighted) distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"LP(::AbstractVector{T},::AbstractVector{T}, ::Int) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.LP-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}, Int64}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.LP","text":"LP(x,y,p)\n\nComputes the L_p distance between two vectors.\n\nType Constraints\n\nT <: Real\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\np::Int               : The power of the norm.\n\nInput Contract (Low level function – Input contract not checked)\n\nbf x = bf y\np > 0\n\nReturn\n\nL_p distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"LI(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.LI-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.LI","text":"LI(x,y)\n\nComputes the L_infty distance between two vectors.\n\nType Constraints\n\nT <: Real\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\n\nInput Contract (Low level function – Input contract not checked)\n\nbf x = bf y\n\nReturn\n\nL_infty distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"KL(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.KL-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.KL","text":"KL(x,y)\n\nComputes the Kullback-Leibler distance between two vectors.\n\nType Constraints\n\nT <: Real\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\n\nInput Contract (Low level function – Input contract not checked)\n\nLet N = bf x.\n\nbf x = bf y\nforall i in 1 N x_i ge 0\nforall i in 1 N y_i ge 0\nsum_i=1^N x_i = 1\nsum_i=1^N y_i = 1\n\nReturn\n\nKL distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"CD(::AbstractVector{T},::AbstractVector{T}; M=::Union{Nothing, AbstractMatrix{T}} = nothing) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.CD-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.CD","text":"CD(x,y[; M=nothing])\n\nComputes the \"cosine\" distance between two vectors.\n\nType Constraints\n\nT <: Real\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\n\nKeyword Arguments\n\nM::Union{Nothing, Matrix{T} : Optional Weight matrix.\n\nInput Contract (Low level function – Input contract not checked)\n\nbf x = bf y\nM = rm nothing vee left( (rm typeof(M) = rm MatrixT) wedge M in boldsymbol S_++^bf x right)\n\nReturn\n\nCosine distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"JD(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}","category":"page"},{"location":"#Cluster.Metrics.JD-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Cluster.jl Documentation","title":"Cluster.Metrics.JD","text":"JD(x,y)\n\nComputes the Jaccard metric between two vectors of a \"discrete\" type. For instance, the vectors could be integers; however, they can  also be of non-numeric type. The metric can also be used with  floating point values but, in that case, it may be more useful  to round/truncate to a particular \"block\" size.\n\nIf both x and y are vectors of zero length, a distance of 0 is returned.\n\nArguments\n\nx::AbstractVector{T} : A numeric vector.\ny::AbstractVector{T} : A numeric vector.\n\nReturn\n\nJaccard distance measure between the two vectors.\n\n\n\n\n\n","category":"method"},{"location":"#Fit-Metric-Functions","page":"Cluster.jl Documentation","title":"Fit Metric Functions","text":"","category":"section"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"raw_confusion_matrix(::AbstractVector{A},::AbstractVector{P}) where {A, P}","category":"page"},{"location":"#Cluster.Metrics.raw_confusion_matrix-Union{Tuple{P}, Tuple{A}, Tuple{AbstractVector{A}, AbstractVector{P}}} where {A, P}","page":"Cluster.jl Documentation","title":"Cluster.Metrics.raw_confusion_matrix","text":"raw_confusion_matrix(act, pred)\n\nComputes a confusion matrix of the discrete variables, act and pred. There are no row or column labels for this matrix.\n\nType Constraints:\n\nExpects types, A and P to have discrete values.\n\nArguments\n\nact ::AbstractVector{A} : A vector of the actual target values.\npred::AbstractVector{P} : A vector of the predicted target values.\n\nInput Contract:\n\n|act| = |pred|\n\nReturn\n\nTuple{Vector{A}, Vector{P}, Matrix{Int}}: A 3-tuple consisting of:\n\nVector of unique values of act.  (Sorted from lowest to highest, otherwise the order returned from the function unique.)\nVector of unique values of pred. (Sorted from lowest to highest, otherwise the order returned from the function unique.)\nA matrix of counts for all pairings of discrete values of act with pred.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"confusion_matrix(::AbstractVector{A},::AbstractVector{P}) where {A, P}","category":"page"},{"location":"#Cluster.Metrics.confusion_matrix-Union{Tuple{P}, Tuple{A}, Tuple{AbstractVector{A}, AbstractVector{P}}} where {A, P}","page":"Cluster.jl Documentation","title":"Cluster.Metrics.confusion_matrix","text":"confusion_matrix(act, pred)\n\nComputes the confusion matrix of the discrete variables, act and pred.\n\nType Constraints:\n\nExpects types, A and P to have discrete values.\n\nArguments\n\nact ::AbstractVector{A} : A vector of the actual target values.\npred::AbstractVector{P} : A prediction vector for the target.\n\nInput Contract:\n\n|act| = |pred|\n\nReturn\n\nMatrix{Any}:\n\nThe raw confusion matrix augmented by a column on the left listing  all actual values (in sorted order if sortable) and  augmented on top with a row listing  all predicted values (in sorted order if sortable).\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"find_cluster_map(::AbstractVector{V},::AbstractVector{A})  where {A, V}","category":"page"},{"location":"#Cluster.Metrics.find_cluster_map-Union{Tuple{V}, Tuple{A}, Tuple{AbstractVector{V}, AbstractVector{A}}} where {A, V}","page":"Cluster.jl Documentation","title":"Cluster.Metrics.find_cluster_map","text":"find_cluster_map(vals, attrs)\n\nThis function finds the best map between the alues vals and a target  attribute, attrs. Both, vals and attrs are assumed to have  discrete values.\n\nArguments\n\nvals::AbstractVector{V}  – The input values.\nattrs::AbstractVector{A} – The attribute values.\n\nReturn\n\n::Dict{V, A} – The map between the values and the attributes.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"predict(::Matrix{Float64},::Matrix{Float64},::Dict{Int, A}) where A","category":"page"},{"location":"#Cluster.Metrics.predict-Union{Tuple{A}, Tuple{Matrix{Float64}, Matrix{Float64}, Dict{Int64, A}}} where A","page":"Cluster.jl Documentation","title":"Cluster.Metrics.predict","text":"predict(data, cl_centers, c_num_map; metric=L2])\n\nThis function predicts the attributes from the map c_num_map based from the input data, data.\n\nArguments\n\ndata::Matrix{Float64}       – The input data that is compatible with the data used to create the cluster map, cl_centers. See the Input Contract below for details.\ncl_centers::Matrix{Float64} – The geometric centers of the clusters.\ncnummap::Dict{Int, A}     – The map from the cluster number to an attribute.\n\nOptional Arguments\n\nmetric::Function – The metric used to measure the distance between data and cluster centers.\n\nInput Contract\n\n|data[:, 1]| = |cl_centers[:, 1]|\n\nReturn\n\n::Vector{A} – The vector of attribute predictions.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Cluster.jl Documentation","title":"Cluster.jl Documentation","text":"","category":"page"}]
}
