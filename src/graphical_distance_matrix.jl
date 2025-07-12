"""
    GraphicalDistanceMatrix{T<:Real}

A structure representing a graphical distance matrix computed from a graph.
This specialized matrix type allows efficient storage and computation of resistance
distances, similar to how ToeplitzMatrices.jl defines specialized matrices.

# Fields
- `graph::SimpleGraph`: The underlying graph
- `matrix::Matrix{T}`: The precomputed resistance distance matrix (can be lazily evaluated)
- `algorithm::Symbol`: The algorithm used to compute resistance distances (:naive, :vectorized, :backsolve, :sparse, :compact)
- `computed::Bool`: Whether the matrix has been computed

# Examples
```julia
using GraphicalDistance, Graphs
g = barabasi_albert(100, 5)
R = GraphicalDistanceMatrix(g)
# Access as a standard matrix
R[1, 2]
# Or compute on demand using a specific algorithm
R = GraphicalDistanceMatrix(g, :compact)
```
"""
mutable struct GraphicalDistanceMatrix{T<:Real} <: AbstractMatrix{T}
    graph::AbstractMatrix{T}
    matrix::Union{Matrix{T},Nothing}
    laplacian::AbstractMatrix{T}
    degree::Vector{T}
    computed::Bool
    C₁₁⁻¹::SparseMatrixCSC{T}
    C₁₁::SparseMatrixCSC{T}
    πᵀ::Vector{Int}

    # Constructor from a SimpleGraph with optional algorithm choice
    function GraphicalDistanceMatrix(g::SimpleGraph, args...; kwargs...)
        GraphicalDistanceMatrix(adjacency_matrix(g), args...; kwargs...)
    end

    # Constructor from an adjacency matrix with optional algorithm choice
    function GraphicalDistanceMatrix(A::AbstractMatrix, perm::Union{Nothing,Tav}=nothing;
        force_laplacian::Bool=false, force_chol_inv::Bool=false, force_chol::Bool=false) where {Tav<:AbstractVector}

        n = size(A,1)
        degree = vec(sum(Float64, A; dims=1))
        laplacian = spdiagm(degree) - A

        if force_laplacian
            # do not form any component, simply return inverse input permutation
            πᵀ = [1:n;]
            if !isnothing(perm)
                laplacian = laplacian[perm, perm]
                πᵀ = invperm(perm)
            end
            C₁₁⁻¹, C₁₁, πᵀ = spzeros(0,0), spzeros(0,0), πᵀ
        else
            # construct the compressed representation
            C₁₁⁻¹, C₁₁, πᵀ = constructor(laplacian, perm; force_chol_inv, force_chol)
        end

        new{Float64}(A, nothing, laplacian, degree, false, C₁₁⁻¹, C₁₁, πᵀ)
    end

    # Constructor from an adjacency matrix with precomputed matrix
    function GraphicalDistanceMatrix(A::AbstractMatrix, matrix::Matrix{T}, perm::Union{Nothing,Tav}=nothing) where {T<:Real,Tav<:AbstractVector}

        degree = vec(sum(T, A; dims=1))
        laplacian = spdiagm(degree) - A

        # construct the compressed representation
        C₁₁⁻¹, C₁₁, πᵀ = constructor(laplacian, perm)

        new{T}(A, matrix, laplacian, degree, true, C₁₁⁻¹, C₁₁, πᵀ)
    end

end

"""
    _subidentity_csc(n, cols)

Build the nxm sparse matrix whose columns are e_{cols[1]},…,e_{cols[m]}
by directly constructing the internal CSC arrays.
"""
function _subidentity_csc(n::Integer, cols::AbstractVector{<:Integer})
    m = length(cols)       # new number of columns
    nnz = m                  # one nonzero per column

    # 1) column‐pointer: each column has exactly one entry,
    #    so entries start at 1,2,…,m and end at m+1
    colptr = collect(1:(nnz+1))

    # 2) row‐indices of the nonzeros:
    #    the k-th nonzero goes in row cols[k]
    rowval = Vector{Int}(cols)    # now always a Vector{Int}

    # 3) nonzero values (all ones for an identity)
    nzval = ones(eltype(rowval), nnz)

    # finally, build the SparseMatrixCSC
    return SparseMatrixCSC{eltype(nzval),Int}(n, m, colptr, rowval, nzval)
end

function constructor(L::AbstractMatrix{T}, perm::Union{Nothing,Tav}=nothing;
    τ::Float64=10.0, force_chol_inv::Bool=false, force_chol::Bool=false) where {T<:Real,Tav<:AbstractVector}

    n = size(L, 1)

    # 1. Partition the matrix into the frist n-1 rows/columns
    if isnothing(perm)
        i_n = n
        L₁₁ = L[1:n-1, 1:n-1]
        chol_perm = nothing
    else
        i_n = perm[n]
        L₁₁ = L[view(perm, 1:n-1), view(perm, 1:n-1)]
        chol_perm = 1:n-1
    end

    # 2. Get the sparse cholesky factorization of the matrix
    factor = cholesky(L₁₁; perm=chol_perm)
    πᵀ = invperm(factor.p)
    C₁₁ = sparse(factor.L)
    Iₛₚ = spdiagm(ones(Float64, n - 1))

    nnz_threshold = if force_chol_inv
        Inf
    elseif force_chol
        0.0
    else
        τ * nnz(C₁₁)
    end

    C₁₁⁻¹ = spsolve(C₁₁, Iₛₚ, nnz_threshold)

    # add element n at the end of the permutation
    append!(πᵀ, n)

    if ~isnothing(perm)
        # 3. Find the total inverse permutation, as the combined permutation of perm and factor.p
        πᵀ = πᵀ[invperm(perm)]
    end

    return C₁₁⁻¹, C₁₁, πᵀ

end

function query_laplacian(L::SparseMatrixCSC{Tv,Ti}, u::Vector{Int}) where {Tv<:Real, Ti<:Integer}

    n = size(L, 1)

    nq = length(u)

    Bij = -ones(n, nq) / n
    for (k,i) in enumerate(u)
        Bij[i, k] = 1 + Bij[i, k]
    end

    Xij = L \ Bij
    Xij = Xij .- ( sum(Xij; dims=1) / n )

    Xij = Xij[u,:]

    d⁺ = diag(Xij)

    return d⁺ .+ d⁺' .- 2Xij

end

function query(R::GraphicalDistanceMatrix{T}, u::Vector{Int}) where {T<:Real}
    # Check if the matrix is computed
    if R.computed
        # If already computed, return the resistance distance
        return @view R.matrix[u, u]
    end

    # Check if only laplacian is available
    if isempty(R.C₁₁) && isempty(R.C₁₁⁻¹)
        uₚ = R.πᵀ[u]
        L = R.laplacian
        return query_laplacian(L, uₚ)
    end

    # If not computed, query the resistance distance on-the-fly
    n = size(R, 1)

    uₚ = view(R.πᵀ, u)
    last_node = n in uₚ
    if last_node
        # Find the location of the last node
        idx_last = findfirst(x -> x == n, uₚ)
        idx_rest = setdiff(1:length(uₚ), idx_last)
        # Remove the last node from the query
        uₚ = setdiff(uₚ, n)
        # Build a view into uₚ that excludes idx
        uᵣ = view(1:length(u), idx_rest)
    else
        uᵣ = 1:length(u)
    end
    Rl = zeros(T, length(u), length(u))
    if isempty(R.C₁₁⁻¹)
        n = size(R, 1)
        # Iₛₚ = spdiagm(ones(Float64, n - 1))
        # rhs = Iₛₚ[ :, uₚ ];
        rhs = _subidentity_csc(n - 1, uₚ)
        x = R.C₁₁ \ rhs
        mul!(view(Rl, uᵣ, uᵣ), x', x)
    else
        x = R.C₁₁⁻¹[:, uₚ]
        xᵀ = permutedims(x)
        spmul_dense!(
            view(Rl, uᵣ, uᵣ),
            xᵀ, x
        )
    end

    d⁺ = diag(Rl)

    # Access the resistance distance
    return d⁺ .+ d⁺' .- 2Rl
end

# Lazy computation of the resistance distance matrix
function compute!(R::GraphicalDistanceMatrix{T}, algorithm::Symbol=:compact) where {T<:Real}
    if !R.computed
        adj_matrix = R.graph
        R.matrix = resistance_distance(adj_matrix, algorithm)
        R.computed = true
    end
    return R
end

# AbstractArray interface implementation
Base.size(R::GraphicalDistanceMatrix) = (size(R.graph, 1), size(R.graph, 1))

function Base.getindex(R::GraphicalDistanceMatrix{T}, i::Integer, j::Integer) where {T<:Real}
    i == j && return zero(T)
    if R.computed
        return R.matrix[i, j]
    else
        # on-the-fly computation of (i,j) entry
        return query(R, [i, j])[1, 2]
    end
end

function Base.setindex!(R::GraphicalDistanceMatrix{T}, v, i::Integer, j::Integer) where {T<:Real}
    error("GraphicalDistanceMatrix is read-only")
end

"""
    graph(R::GraphicalDistanceMatrix)

Get the underlying graph.
"""
graph(R::GraphicalDistanceMatrix) = R.graph

"""
    resistance_distance(R::GraphicalDistanceMatrix, i::Integer, j::Integer)

Get the resistance distance between vertices i and j.
"""
function resistance_distance(R::GraphicalDistanceMatrix{T}, i::Integer, j::Integer) where {T<:Real}
    return R[i, j]
end

# Convert to a standard matrix
Base.Matrix(R::GraphicalDistanceMatrix{T}) where {T<:Real} = Matrix(compute!(R).matrix)

# Show method for pretty printing
function Base.show(io::IO, R::GraphicalDistanceMatrix{T}) where {T<:Real}
    n = size(R.graph, 1)
    print(io, "$(n)×$(n) GraphicalDistanceMatrix{$T}")
end

# Pretty printing in REPL
function Base.show(io::IO, ::MIME"text/plain", R::GraphicalDistanceMatrix{T}) where {T<:Real}
    n = size(R.graph, 1)
    println(io, "$(n)×$(n) GraphicalDistanceMatrix{$T}")
end


# ---------------------------------------------------------------------------- #
#                               Benchmark helpers                              #
# ---------------------------------------------------------------------------- #

function benchmark_graphical_distance_matrix(g::SimpleGraph,args...; kwargs...)
    return benchmark_graphical_distance_matrix(adjacency_matrix(g), args...; kwargs...)
end
function benchmark_graphical_distance_matrix(A::AbstractMatrix, args...; kwargs...)
    # Benchmark the construction and computation of the GraphicalDistanceMatrix
    b = @benchmark GraphicalDistanceMatrix($A, $(args)...; $(kwargs)...)
    R = GraphicalDistanceMatrix(A, args...; kwargs...)
    nnz_laplacian = nnz(R.laplacian)
    nnz_chol = nnz(R.C₁₁)
    nnz_invchol = nnz(R.C₁₁⁻¹)
    results = Dict(
        "nnz_laplacian" => nnz_laplacian,
        "nnz_chol" => nnz_chol,
        "nnz_invchol" => nnz_invchol,
        "times" => b.times
    )
    return results
end


function benchmark_graphical_distance_query(R::GraphicalDistanceMatrix{T}, nq::Int; seed = 0) where {T<:Real}
    # Benchmark the construction and computation of the GraphicalDistanceMatrix
    n = size(R, 1)
    rng = MersenneTwister(seed)
    b = @benchmark query($R, vec) evals=1 samples=1000 seconds=60 setup=(vec=randperm($rng, $n)[1:$(nq)])
    nnz_laplacian = nnz(R.laplacian)
    nnz_chol = nnz(R.C₁₁)
    nnz_invchol = nnz(R.C₁₁⁻¹)
    results = Dict(
        "nnz_laplacian" => nnz_laplacian,
        "nnz_chol" => nnz_chol,
        "nnz_invchol" => nnz_invchol,
        "times" => b.times
    )
    return results
end
