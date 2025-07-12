function resistance_distance(A::SparseMatrixCSC{Tv,Ti}, ::Val{:naive}) where {Ti<:Integer,Tv<:Real}

    n  = size(A, 1)

    # Pre-allocate all the required memory for the matrices d, L, L⁺, and R
    # and use the appropriate method for the resistance distance calculation
    # based on the version specified.
    R  = zeros(n, n)

    # 0. Get the degree vector of the graph
    d = vec(sum(Float64, A; dims=1))

    # 1. Get the Laplacian matrix of the graph
    L = spdiagm(d) - A

    # 2. Compute the pseudo-inverse of the Laplacian matrix
    L⁺ = pinv(Matrix(L))

    # 3. Assemble the resistance distance matrix
    resistance_distance_naive!(R, L⁺)

end

function resistance_distance_naive!(R::DenseMatrix, L⁺::DenseMatrix)

    n = size(L⁺, 1)

    @inbounds for i in 1:n, j in 1:n
        R[i, j] = L⁺[i, i] + L⁺[j, j] - 2L⁺[i, j]
    end

    return R
end


function resistance_distance(R::DenseMatrix, L::SparseMatrixCSC{Tv,Ti}, ::Val{:naive}) where {Ti<:Integer,Tv<:Real}

    # 1. Compute the pseudo-inverse of the Laplacian matrix
    L⁺ = pinv(Matrix(L))

    # 2. Assemble the resistance distance matrix
    n = size(L⁺, 1)
    @inbounds for i in 1:n, j in 1:n
        R[i, j] = L⁺[i, i] + L⁺[j, j] - 2L⁺[i, j]
    end

    return R
end
