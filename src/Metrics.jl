module Metrics 

# Export metrics metrics: L_2, L_p, L_∞, Kullback-Leibler, Cosine, and Jaccard.
# and fit metrics: raw_confusion_matrix, confusion_matrix, find_cluster_map
export L2, LP, LI, KL, CD, JD, raw_confusion_matrix, confusion_matrix, find_cluster_map, predict

import LinearAlgebra as LA


const TOL=1.0e-6

"""
    L2(x,y[; M=nothing])

Computes the ``L_2`` distance between two vectors.
One of the features that may be different from other packages
is the use of weighted metrics in some instances.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Keyword Arguments
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``M = {\\rm nothing} \\vee \\left( ({\\rm typeof}(M) = {\\rm Matrix}\\{T\\}) \\wedge M \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``

# Return
``L_2`` (optionally weighted) distance measure between the two vectors.
"""
function L2(x::AbstractVector{T},
            y::AbstractVector{T};
            M::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}

    d = x .- y
    if M === nothing
        return LA.norm2(d)
    else
        return sqrt(LA.dot(d, M, d))
    end
end

"""
    LP(x,y,p)

Computes the ``L_p`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.
- `p::Int`               : The power of the norm.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- `p > 0`

# Return
``L_p`` distance measure between the two vectors.
"""
function LP(x::AbstractVector{T},
            y::AbstractVector{T},
            p::Int     ) where {T <: Real}

    return LA.norm(x .- y, p)
end

"""
    LI(x,y)

Computes the ``L_\\infty`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``

# Return
``L_\\infty`` distance measure between the two vectors.
"""
function LI(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}

    return max.(abs.(x .- y))
end


"""
    JD(x,y)

Computes the `Jaccard` metric between two vectors of a "discrete" type.
For instance, the vectors could be integers; however, they can 
also be of non-numeric type. The metric can also be used with 
floating point values but, in that case, it may be more useful 
to round/truncate to a particular "block" size.

If both `x` and `y` are vectors of zero length, a distance of ``0`` is returned.

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Return
`Jaccard` distance measure between the two vectors.
"""
function JD(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}
    d = length(symdiff(x,y))
    u = length(union(x,y)) 

    return length(u) == 0 ? 0.0 : d / u
end


"""
    KL(x,y)

Computes the ``Kullback-Leibler`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Input Contract (Low level function -- Input contract not checked)
Let ``N = |{\\bf x}|``.
- ``|{\\bf x}| = |{\\bf y}|``
- ``\\forall i \\in [1, N]: x_i \\ge 0``
- ``\\forall i \\in [1, N]: y_i \\ge 0``
- ``\\sum_{i=1}^N x_i = 1``
- ``\\sum_{i=1}^N y_i = 1``

# Return
`KL` distance measure between the two vectors.
"""
function KL(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}

    z = zero(T)
    d1 = map((a, b) -> a == z ? z : a * log(a / b), x, y)
    d2 = map((a, b) -> b == z ? z : b * log(b / a), x, y)

    return sum(d1 .+ d2)
end


"""
    CD(x,y[; M=nothing])

Computes the "cosine" distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Keyword Arguments
- `M::Union{Nothing, Matrix{T}` : Optional Weight matrix.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``M = {\\rm nothing} \\vee \\left( ({\\rm typeof}(M) = {\\rm Matrix}\\{T\\}) \\wedge M \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``

# Return
Cosine distance measure between the two vectors.

"""
function CD(x::AbstractVector{T},
            y::AbstractVector{T};
            M::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}
    z = zero(T)
    o = one(T)
    tol = T(TOL)

    if all(abs.(x .- y) / (2.0 .* (abs.(x) .+ abs.(y))) .< tol)
        return o
    elseif all(abs.(x) .< tol)
        return o
    elseif all(abs.(y) .< tol)
        return o
    elseif M === nothing
        return o - LA.dot(x, y) / sqrt(LA.dot(x, x) * LA.dot(y, y))
    end

    return o - LA.dot(x, M, y) / sqrt(LA.dot(x, M, x) * LA.dot(y, M, y))
end


# Check if a vector is sortable.
function is_sortable(xs::AbstractVector)
	N = length(xs)
	sortable = true

	try
		for i in 1:(N-1)
			xs[i] < xs[i+1]
		end
	catch
		sortable = false
	end

	return sortable
end


"""
    raw_confusion_matrix(act, pred)

Computes a confusion matrix of the discrete variables, `act` and `pred`.
There are no row or column labels for this matrix.

# Type Constraints:
- Expects types, `A` and `P` to have discrete values.

# Arguments
- `act ::AbstractVector{A}` : A vector of the actual target values.
- `pred::AbstractVector{P}` : A vector of the predicted target values.

# Input Contract:
- |act| = |pred|

# Return
Tuple{Vector{A}, Vector{P}, Matrix{Int}}: A 3-tuple consisting of:
- Vector of unique values of `act`.  (Sorted from lowest to highest, otherwise the order returned from the function `unique`.)
- Vector of unique values of `pred`. (Sorted from lowest to highest, otherwise the order returned from the function `unique`.)
- A matrix of counts for all pairings of discrete values of `act` with `pred`.
"""
function raw_confusion_matrix(act::AbstractVector{A}, pred::AbstractVector{P}) where {A, P}
    N = length(act) 

    # Check Input Contract:
    if length(pred) != N
        throw(DomainError(N, "confusion_matrix: Vector inputs, `act` and `pred` do NOT have the same length"))
    end

    # Get unique values of actual values and their associated length.
    a_vals = unique(act)
    a_N = length(a_vals)

	# Check that the vector is sortable, if so -- sort.
	is_sortable(a_vals) && sort!(a_vals)
    
	# Get unique values of predicted values and their associated length.
    p_vals = unique(pred)
    p_N = length(p_vals)

	# Check that the vector is sortable, if so -- sort.
	is_sortable(p_vals) && sort!(p_vals)

    # Confusion Matrix -- to be filled in.
    CM = fill(0, a_N, p_N)
    da = Dict{A, Int}()
    dp = Dict{P, Int}()

    # Map the actual values to index order as assigned by either sort;
    # or, in case the values are not sortable, the function `unique`.
    @inbounds for i in 1:a_N
        da[a_vals[i]] = i
    end
    
    # Map the predicted values to index order as assigned by either sort;
    # or, in case the values are not sortable, the function `unique`.
    @inbounds for i in 1:p_N
        dp[p_vals[i]] = i
    end

    #= Fill in the non-zero entries of the confusion matrix
       as the number of counts for each pair of (actual, predicted) pairings
       as encoded by the actual and predicted index values.
	=#
    @inbounds for i in 1:N
        CM[da[act[i]], dp[pred[i]]] += 1
    end

    # Return the confusion matrix along with the associated 
    # ordered actual and predicted values.
    return (a_vals, p_vals, CM)    
end



"""
    confusion_matrix(act, pred)

Computes the confusion matrix of the *discrete* variables, `act` and `pred`.

# Type Constraints:
- Expects types, `A` and `P` to have discrete values.

# Arguments
- `act ::AbstractVector{A}` : A vector of the actual target values.
- `pred::AbstractVector{P}` : A prediction vector for the target.

# Input Contract:
- |act| = |pred|

# Return
Matrix{Any}:
- The raw confusion matrix augmented by a column on the left listing 
  all *actual* values (in sorted order if sortable) and 
  augmented on top with a row listing 
  all *predicted* values (in sorted order if sortable).
"""
function confusion_matrix(act::AbstractVector{A}, pred::AbstractVector{P}) where {A, P}
	# Get the raw confusion matrix.
    res = raw_confusion_matrix(act, pred)

	# Get the size of the raw confusion matrix.
    N, M = size(res[3])

	# Create the matrix for the confusion matrix.
    PM = Matrix(undef, N+1, M+1)

	# Fill it.
    PM[2:N+1, 2:M+1] = copy(res[3]) # Fill in confusion matrix in lower right of `PM`.
    PM[2:N+1, 1    ] = res[1]       # Fill in actual values on the left of `PM`.
    PM[1    , 2:M+1] = res[2]       # Fill in predicted values on the top of `PM`.

    # Upper left hand label: Act versus Pred.
    PM[1, 1] = "ACT\\PRED"

    return PM
end


"""
	find_cluster_map(vals, attrs) 

This function finds the best map between the alues `vals` and a target 
attribute, `attrs`. Both, `vals` and `attrs` are *assumed* to have 
discrete values.

# Arguments
- vals::AbstractVector{V}  -- The input values.
- attrs::AbstractVector{A} -- The attribute values.

# Return
::Dict{V, A} -- The map between the values and the attributes.
"""
function find_cluster_map(vals::AbstractVector{V}, attrs::AbstractVector{T}) where {V, T}
 	uvals, uattrs, mat = raw_confusion_matrix(vals, attrs)
	tvmap = Dict{T, V}()
	idxsm = argmax(mat, dims=1)
     idxs = @view idxsm[1, :]
	for idx in idxs
		tvmap[uattrs[idx[1]]] = uvals[idx[2]]
	end
	return tvmap
end


"""
	predict(data, cl_centers, c_num_map; metric=L2]) 

This function predicts the attributes from the map `c_num_map` based
from the input data, `data`.

# Arguments
- data::Matrix{Float64}       -- The input data that is compatible with the data used to create the cluster map, `cl_centers`. See the `Input Contract` below for details.
- cl_centers::Matrix{Float64} -- The geometric centers of the clusters.
- c_num_map::Dict{Int, A}     -- The map from the cluster number to an attribute.

# Optional Arguments
- metric::Function -- The metric used to measure the distance between data and cluster centers.

# Input Contract
- |data[:, 1]| = |cl_centers[:, 1]|

# Return
::Vector{A} -- The vector of attribute predictions.
"""
function predict(data      ::Matrix{Float64}, 
				 cl_centers::Matrix{Float64},
				 c_num_map ::Dict{Int, A}   ;
				 metric=L2 ::Function        ) where A
	dM, dN = size(data)
	cM, cN = size(cl_centers)

	cM == dM || error("Data matrix, `data`, and `cl_centers` do not have the same number of rows.")

	# Create the vector of attribute predictions.
	preds = Vector{A}(undef, dN)

	# Fill in the vector `preds`.
	@inbounds for i in 1:dN
		d = data[:, i]
		min_dis = Inf
		j_min = -1
		for j in 1:cN
			dis = metric(d, cl_centers[:, j])
			if dis < min_dis
				min_dis = dis
				j_min = j
			end
		end
		preds[i] = c_num_map[j_min]
	end

	return preds
end


end # module Metrics
