function resistance_distance(A::SparseMatrixCSC{Tv,Ti}, ::Val{:backsolve}) where {Ti<:Integer,Tv<:Real}

    n  = size(A, 1)

    # Pre-allocate all the required memory for the matrices d, L, L⁺, and R
    # and use the appropriate method for the resistance distance calculation
    # based on the version specified.
    d⁺ = zeros(Float64, n)
    R  = zeros(n, n)

    # 0. Get the degree vector of the graph
    d = vec(sum(Float64, A; dims=1))

    # 1. Get the Laplacian matrix of the graph
    L = diagm(d) - Matrix(A)

    # 2. Compute the pseudo-inverse of the Laplacian matrix
    J  = 1/n
    L⁺ = (L .+ J) \ LinearAlgebra.I .- J

    # 3. Assemble the resistance distance matrix
    resistance_distance_backsolve!(R, d⁺, L⁺)

end

function resistance_distance_backsolve!(R::DenseMatrix, d⁺::DenseVector, L⁺::DenseMatrix)

    extract_diag!(d⁺,L⁺)
    @. R = d⁺ + d⁺' - 2*L⁺

    return R
end
