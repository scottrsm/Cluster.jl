import Random
import RDatasets
using LinearAlgebra: Symmetric, isposdef
using Test

using Cluster

#=-------------------------------------------------------
------------ DATA PREP ----------------------------------
---------------------------------------------------------
=#

# Set random seed.
Random.seed!(1)

# Constants used to set algo parameters.
const TOL            ::Float64 = 1.0e-4
const NUM_TRIALS_T1  ::Int   = 300
const NUM_TRIALS_IRIS::Int   = 1000
const NUM_ITERATIONS ::Int   = 1000
const KM_THRESHOLD   ::Float64 = 1.0e-2

# Synthetic data for test: T1.
# There are 10 "natural" clusters.
M1 = [-1, -2] .+ rand(2, 100)
M2 = 3.0 .* [1, 2] .+ rand(2, 100)
M3 = 6.0 .* [2, 1] .+ rand(2, 100)
M4 = 9.0 .* [1, 1] .+ rand(2, 100)
M5 = 12.0 .* [-1, 1] .+ rand(2, 100)
M6 = 15.0 .* [0.5, 3.0] .+ rand(2, 100)
M7 = 18.0 .+ [-2.4, 1.0] .+ rand(2, 100)
M8 = 21.0 .+ [0.3, -0.3] .* rand(2, 100)
M9 = 24.0 .+ rand(2, 100)
M10 = 27.0 .+ rand(2, 100)

M = hcat(M1, M2, M3, M4, M5, M6, M7, M8, M9, M10)

# Data for test: IRIS.
iris = RDatasets.dataset("datasets", "iris")
MI = permutedims(Matrix(iris[:, [:SepalWidth, :SepalLength]]), (2,1))

# Centroid columns come back in an order that depends on the random start;
# compare centroid sets by sorting the columns.
sortcols(A) = sortslices(A, dims=2)

# Recompute the total variation of a clustering from its parts.
total_variation(X, cmap, xc, metric) = sum(metric(view(X, :, i), view(xc, :, cmap[i])) for i in 1:size(X, 2))

 
@testset "Cluster (Fidelity)                    " begin
    @test length(detect_ambiguities(Cluster)) == 0
end


@testset "Cluster (Test Metrics)                " begin
    C = [1. 2.; 2. 5.]

    @test L2([1., 2.], [3., -4.]     )   ≈  6.324555320336759   rtol=TOL
    @test L2([1., 2.], [3., -4.]; M=C)   ≈ 11.661903789690601   rtol=TOL
    @test LP([1., 2.], [3., -4.], 3  )   ≈  6.0731779437513245  rtol=TOL
    @test LI([1., 2.], [3., -4.]     )   ≈  6.0                 rtol=TOL

    # Cosine distance: identical directions are at distance 0, opposite at 2.
    @test CD([1., 2.], [1., 2.])           ≈ 0.0  atol=TOL
    @test CD([1., 2.], [2., 4.])           ≈ 0.0  atol=TOL
    @test CD([1., 2.], [-1., -2.])         ≈ 2.0  rtol=TOL
    @test CD([1., 0.], [0., 1.])           ≈ 1.0  rtol=TOL
    @test CD([0., 0.], [1., 2.])           == 1.0
    @test CD([1., 2.], [1., 2.]; M=C)      ≈ 0.0  atol=TOL

    # Kullback-Leibler (symmetrised): zero for identical distributions.
    @test KL([0.5, 0.5], [0.5, 0.5])       == 0.0
    @test KL([0.5, 0.5], [0.25, 0.75])     ≈ 0.2746530721670274 rtol=TOL

    # Jaccard: works for non-numeric element types too.
    @test JD([1, 2, 3], [2, 3, 4])         ≈ 0.5 rtol=TOL
    @test JD([:a, :b], [:b, :c])           ≈ 2/3 rtol=TOL
    @test JD(Int[], Int[])                 == 0.0
end


@testset "Cluster (Test kmeans_cluster)         " begin
    X = hcat([rand(2, 20) .+ 5.0 .* [i, i] for i in 1:5]...)

    # Basic invariants: every point maps to its nearest centroid and the
    # returned variation matches the mapping.
    cmap, xc, tv, unused, iters, conv = kmeans_cluster(X, 5; seed=1)
    @test length(cmap) == size(X, 2)
    @test size(xc) == (2, 5)
    @test all(1 .<= cmap .<= 5)
    @test conv
    @test tv ≈ total_variation(X, cmap, xc, L2) rtol=TOL
    @test all(cmap[i] == argmin([L2(X[:, i], xc[:, j]) for j in 1:5]) for i in 1:size(X, 2))

    # The seed selects the random start, so it must be reproducible and must matter.
    @test kmeans_cluster(X, 3; seed=7, N=1)[2] == kmeans_cluster(X, 3; seed=7, N=1)[2]
    @test kmeans_cluster(X, 3; seed=1, N=1)[2] != kmeans_cluster(X, 3; seed=99, N=1)[2]

    # Edge cases of the input contract.
    @test kmeans_cluster(rand(2, 5), 5)[3] ≈ 0.0 atol=TOL      # k == m
    @test kmeans_cluster(rand(2, 5), 1)[6]                       # k == 1 converges
    @test_throws DomainError kmeans_cluster(rand(2, 5), 6)       # k > m
    @test_throws DomainError kmeans_cluster(rand(2, 5), 0)
    @test_throws DomainError kmeans_cluster(rand(2, 5), 2; N=0)
    @test_throws DomainError kmeans_cluster(rand(2, 5), 2; threshold=0.0)

    # Perfectly clustered data (zero variation) converges instead of running N iterations.
    _, _, tv0, _, iters0, conv0 = kmeans_cluster(fill(1.0, 2, 10), 1; N=50)
    @test tv0 == 0.0 && conv0 && iters0 < 50

    # Weight matrix: accepted when symmetric positive definite, otherwise rejected.
    W = [2.0 0.5; 0.5 1.0]
    @test isposdef(Symmetric(W))
    cmapw, xcw, tvw, _, _, _ = kmeans_cluster(X, 5; W=W, check_W=true, seed=1)
    @test tvw ≈ total_variation(X, cmapw, xcw, (x, y) -> L2(x, y; M=W)) rtol=TOL
    @test_throws DomainError   kmeans_cluster(X, 2; W=[1.0 2.0; 2.0 1.0], check_W=true)  # not positive definite
    @test_throws DomainError   kmeans_cluster(X, 2; W=[1.0 2.0; 0.0 1.0], check_W=true)  # not symmetric
    @test_throws DomainError   kmeans_cluster(X, 2; W=rand(3, 3))                        # wrong shape
    @test_throws DomainError   kmeans_cluster(X, 1000; W=W, check_W=true)                # k > m still checked
    @test_throws ArgumentError kmeans_cluster(X, 2; W=W, dmetric=LI)                     # metric has no `M` keyword
end


@testset "Cluster (Test find_best_info_for_ks)  " begin
    X = hcat([rand(2, 20) .+ 5.0 .* [i, i] for i in 1:5]...)

    # Each trial uses the full iteration budget `N` and a seed derived from its
    # trial index, so a single trial reproduces the corresponding direct call.
    tv, cmap, xc, unused = find_best_info_for_ks(X, 5:5; num_trials=1, seed=3, N=1000)
    d = kmeans_cluster(X, 5; seed=4, N=1000)
    @test tv[5] == d[3]
    @test cmap[5] == d[1]
    @test xc[5] == d[2]

    # Results are independent of thread scheduling.
    a = find_best_info_for_ks(X, 1:6; num_trials=50, seed=3)
    b = find_best_info_for_ks(X, 1:6; num_trials=50, seed=3)
    @test a[1] == b[1] && a[3] == b[3]

    # The best variation is non-increasing in k for L2 given enough trials.
    tvs = collect(values(a[1]))
    @test all(tvs[i+1] <= tvs[i] + TOL for i in 1:length(tvs)-1)

    @test_throws DomainError find_best_info_for_ks(X, 3:2)
    @test_throws DomainError find_best_info_for_ks(X, 0:3)
    @test_throws DomainError find_best_info_for_ks(X, 1:1000)
    @test_throws DomainError find_best_info_for_ks(X, 1:3; num_trials=0)

    # The weight matrix is validated once, up front.
    @test_throws DomainError find_best_info_for_ks(X, 1:3; W=[1.0 2.0; 2.0 1.0], num_trials=2)
    @test length(find_best_info_for_ks(X, 1:3; W=[2.0 0.5; 0.5 1.0], num_trials=2)[1]) == 3
end


@testset "Cluster (Test find_best_cluster: T1)  " begin
    kbest, mp, xc, ds = find_best_cluster(M, 1:15                    ;
										  seed = 1                   ,
                                          num_trials = NUM_TRIALS_T1 , 
                                          N          = NUM_ITERATIONS, 
                                          threshold  = KM_THRESHOLD   )
    C = [-11.462074155872033 -0.4647786335684582 3.4934544090790887 7.993343670939024 9.479449965138636 12.511540899137428 16.067365037149926 21.145246393930183 24.49422035218324 27.447603324472404;
          12.457669306098992 -1.5175945768316743 6.503798597479442 45.478311604029514 9.509921208077595 6.524309174201606 19.536020332393772 20.839828692792324 24.483990255724194 27.488695232426544 ]

    best_var = 350.15757352935907 

    @test size(xc)   == (2, kbest)
    @test kbest      == 10
    @test sortcols(xc) ≈ C   rtol=TOL
    @test ds ≈ best_var      rtol=TOL
    @test mp isa Vector{Int}
    @test length(mp) == size(M, 2)
    @test ds ≈ total_variation(M, mp, xc, L2) rtol=TOL

    # A two-element cluster range is a valid input.
    kbest2, mp2, xc2, _ = find_best_cluster(M, 1:2; seed=1, num_trials=10)
    @test kbest2 in (1, 2) && size(xc2) == (2, kbest2) && mp2 isa Vector{Int}

    # Unused centroids are dropped and the mapping is re-indexed (still a Vector{Int}).
    Y = [zeros(2, 5) ones(2, 5)]
    kbestY, mpY, xcY, _ = find_best_cluster(Y, 5:5; seed=1, num_trials=3)
    @test kbestY <= 2 && size(xcY) == (2, kbestY)
    @test mpY isa Vector{Int} && all(1 .<= mpY .<= kbestY)
end


# Try clustering with metrics: L2 (default), L1, KL (Kullback-Liebler).
@testset "Cluster (Test find_best_cluster: IRIS)" begin

    #=------------------------
    ----- Default metric, L2.
    --------------------------
	=#
    kbest, mp, xc, ds = find_best_cluster(MI, 1:7                     ; 
                                          dmetric=L2                  , 
										  seed = 1                    ,
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [2.694 3.0612244897959178 3.409803921568628;
         5.77  6.7918367346938755 5.003921568627451 ]
    CM = [ 50 0 0  ;
           0 38 12 ;
           0 15 35  ]

    best_var = 62.61158509950636

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test sortcols(xc) ≈ C   rtol=TOL
    @test ds ≈ best_var      rtol=TOL

    N, M = size(iris)
    iris[!, :Cluster] = map(i -> mp[i], 1:N)
    specs = Symbol.(iris[!, :Species])

	# Find the best mapping between the attribute, :Species, and the cluster numbers.
	cmap = find_cluster_map(Symbol.(iris[:, :Species]), iris[:, :Cluster])
	
	#= Given the Sepal width and height data from the IRIS data set, find the nearest
	   cluster number, then use the map, cmap, to find the associated predicted 
	   attribute, pattr = :Species.
	=#
	pattr = predict(permutedims(Matrix(iris[:, [:SepalWidth, :SepalLength]])), xc, cmap)

	# See how close the predicted attributes, pattr,  compares with the 
	# actual species, specs.
    res = raw_confusion_matrix(specs, pattr)

    @test res[3] == CM

    #=------------------------
    ----- L1 metric.
    --------------------------
	=#
    L1_metric = (x,y;kwargs...) -> LP(x,y,1;kwargs...) 
    kbest, mp, xc, ds = find_best_cluster(MI, 1:7                     ; 
										  seed = 1                    ,
                                          dmetric    = L1_metric      ,
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS ,
                                          threshold  = KM_THRESHOLD    )

    C = [2.7347826086956517 3.062745098039215 3.3320754716981136;
         5.8108695652173905 6.762745098039212 4.986792452830188  ]
    best_var = 80.62774212227956
 
    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test sortcols(xc) ≈ C   rtol=TOL
    @test ds ≈ best_var      rtol=TOL


    #=------------------------------
    ----- Kullback-Leibler metric.
    --------------------------------
	=#
    kbest, mp, xc, ds = find_best_cluster(MI, 1:7                     ; 
										  seed = 1                    ,
                                          dmetric    = KL             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [2.671698113207547 3.0812499999999994 3.4510204081632656;
         5.745283018867926 6.795833333333331  5.016326530612244  ]
    best_var = 8.309019010482672

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test sortcols(xc) ≈ C   rtol=TOL
    @test ds ≈ best_var      rtol=TOL


    #=------------------------
    ----- Cosine metric.
    --------------------------
	=#
    kbest, mp, xc, ds = find_best_cluster(MI, 1:7                     ; 
										  seed = 1                    ,
                                          dmetric    = CD             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [2.866336633663367  3.4510204081632656;
         6.2445544554455426 5.016326530612244  ]
    best_var = 0.0999006220938734

    @test size(xc)   == (2, kbest)
    @test kbest      == 2
    @test sortcols(xc) ≈ C   rtol=TOL
    @test ds ≈ best_var      rtol=TOL


    #=------------------------
    ----- Jaccard metric.
    --------------------------
	=#
    kbest, mp, xc, ds = find_best_cluster(MI, 1:7                     ; 
										  seed = 1                    ,
                                          dmetric    = JD             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.057333333333334; 
         5.843333333333335 ]
    best_var = 150.0

    @test size(xc)   == (2, kbest)
    @test kbest      == 1
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL

end


@testset "Cluster (Test fit metrics)            " begin
    # find_cluster_map: each attribute maps to the value it co-occurs with most.
    @test find_cluster_map([:a, :a, :b, :b, :c, :c], [2, 2, 3, 3, 1, 1]) == Dict(1 => :c, 2 => :a, 3 => :b)
    @test find_cluster_map([:a, :a, :a, :b], [1, 1, 2, 2])               == Dict(1 => :a, 2 => :a)

    a_vals, p_vals, cm = raw_confusion_matrix([:x, :y, :y, :x], [1, 2, 2, 2])
    @test a_vals == [:x, :y] && p_vals == [1, 2]
    @test cm == [1 1; 0 2]
    @test_throws DomainError raw_confusion_matrix([:x, :y], [1])

    pm = confusion_matrix([:x, :y, :y, :x], [1, 2, 2, 2])
    @test size(pm) == (3, 3) && pm[1, 1] == "ACT\\PRED" && pm[2:3, 2:3] == [1 1; 0 2]

    # predict: nearest centroid, then the attribute map.
    centers = [0.0 10.0; 0.0 10.0]
    @test predict([1.0 9.0; 1.0 9.0], centers, Dict(1 => :lo, 2 => :hi)) == [:lo, :hi]
    @test_throws DimensionMismatch predict(rand(3, 2), centers, Dict(1 => :lo, 2 => :hi))
end
