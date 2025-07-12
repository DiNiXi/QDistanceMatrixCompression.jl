module QDistanceMatrixCompression

using Graphs, LinearAlgebra, SparseArrays, BenchmarkTools, Random

export spsolve, GraphicalDistanceMatrix, resistance_distance, graph, query

# Write your package code here.
add(x, y) = x + y

function resistance_distance(G::SimpleGraph, version::Symbol=:naive, args...)
    return resistance_distance(adjacency_matrix(G), Val(version), args...)
end

function resistance_distance(A::T, version::Symbol=:naive, args...) where {T<:AbstractMatrix}
    return resistance_distance(A, Val(version), args...)
end

profile_resistance_distance(G::SimpleGraph, version::Symbol=:naive) = profile_resistance_distance(adjacency_matrix(G), Val(version))

function profile_resistance_distance(A::T, version::Symbol=:naive) where {T<:AbstractMatrix}
    return profile_resistance_distance(A, Val(version))
end

include("spsolve.jl")

include("resdist_backsolve.jl")
include("resdist_naive.jl")
include("resdist_compact.jl")
include("graphical_distance_matrix.jl")


function extract_diag!(d::AbstractVector{T}, A::AbstractMatrix{T}) where T
    @assert length(d) == min(size(A)...)
    @inbounds for i in eachindex(d)
        d[i] = A[i,i]
    end
    return d
end

end
