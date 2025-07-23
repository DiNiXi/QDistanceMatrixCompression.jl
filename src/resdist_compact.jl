function profile_resistance_distance(A::SparseMatrixCSC{Tv,Ti}, ::Val{:compact}) where {Ti<:Integer,Tv<:Real}

    res, t = @timed resistance_distance(A, Val{:compact}())

    return res, t

end


function sparse_to_dense!(R::AbstractMatrix{T}, S::SparseMatrixCSC{T}) where T
    # @assert size(R) == size(S) "R and S must have the same dimensions"
    # fill!(R, zero(T))  # clear out any old entries in R
    colptr = S.colptr
    rowval = S.rowval
    nzval  = S.nzval

    @inbounds for j in 1:(length(colptr)-1)
        for p in colptr[j]:(colptr[j+1]-1)
            i = rowval[p]
            R[i, j] = nzval[p]
        end
    end

    return R
end

function _form_laplacian_and_partition_compact_v2(A::SparseMatrixCSC{Tv,Ti}, fix_perm::Bool=false) where {Ti<:Integer,Tv<:Real}

    n = size(A, 1)

    # 0. Get the degree vector of the graph
    d = vec(sum(Float64, A; dims=1))

    # 1. Get the Laplacian matrix of the graph
    L = spdiagm(d) - A

    # 2. Partition the matrix into the frist n-1 rows/columns
    L11 = L[1:n-1, 1:n-1]

    perm = fix_perm ? (1 : n-1) : nothing

    # 3. Get the sparse cholesky factorization of the matrix
    # factor = cholesky(L11; perm = 1:n-1)
    factor = cholesky(L11, perm=perm)
    ip = invperm(factor.p)
    C11 = sparse( factor.L )
    Isp = spdiagm(ones(Float64, n-1))
    tau = 0.1 * size(C11, 1) * size(C11, 1) / 2
    if nnz(C11) < tau / 3
        C11_inv = spsolve(C11, Isp, tau) # alternative: C11_inv = factor.L \ Isp (slower for most cases)
        dense = isempty(C11_inv)
    else
        dense = true
    end
    if dense
        C11 = Matrix(C11)
        C11_inv = inv(C11)
        if ~fix_perm
            C11_inv = C11_inv[ip, ip]
        end
    else
        resize!( C11.rowval, nnz(C11_inv) )
        resize!( C11.nzval, nnz(C11_inv) )
        if ~fix_perm
            # permute the rows and columns of C11_inv according to the inverse permutation ip
            permute!(C11_inv, 1:n-1, ip, C11)
        end
    end

    return L, C11, C11_inv

end

function spmul_dense!(C::AbstractMatrix{T},
                     A::SparseMatrixCSC{T,Ti},
                     B::SparseMatrixCSC{T,Ti}) where {T,Ti}
    # dimension check
    m, p1 = size(A)
    p2, n = size(B)
    @assert p1 == p2 "Inner dimensions must match: size(A) = ($m,$p1), size(B)=($p2,$n)"

    # zero out C once
    fill!(C, zero(T))

    # grab the internal CSC pointers
    colptrA, rowvalA, nzvalA = A.colptr, A.rowval, A.nzval
    colptrB, rowvalB, nzvalB = B.colptr, B.rowval, B.nzval

    # for each column j of B (hence of C)
    @inbounds for j in 1:n
        bj_start = colptrB[j]
        bj_end   = colptrB[j+1] - 1

        # for each nonzero B[k,j]
        @inbounds for idxB in bj_start:bj_end
            k   = rowvalB[idxB]
            bkj = nzvalB[idxB]

            # add bkj * A[:,k] into C[:,j]
            ak_start = colptrA[k]
            ak_end   = colptrA[k+1] - 1
            @inbounds for idxA in ak_start:ak_end
                i    = rowvalA[idxA]
                aik  = nzvalA[idxA]
                C[i,j] += aik * bkj
            end
        end
    end

    return C
end


function resistance_distance(A::SparseMatrixCSC{Tv,Ti}, ::Val{:compact}, fix_perm::Bool=false) where {Ti<:Integer,Tv<:Real}

    L, C11, C11_inv = _form_laplacian_and_partition_compact_v2(A, fix_perm)
    R = _do_resistance_distance_compact!(C11_inv)
    return R

end

function _do_resistance_distance_compact!(C⁻¹::Matrix{Tv}) where {Tv<:Real}

    n  = size(C⁻¹, 1)+1
    d⁺ = zeros(Float64, n)
    R  = zeros(n, n)
    C⁻ᵀ = transpose(C⁻¹)

    mul!( view(R,1:n-1,1:n-1), C⁻ᵀ, C⁻¹)
    extract_diag!(d⁺,R)
    @. R = d⁺ + d⁺' - 2*R

end

function _do_resistance_distance_compact!(C⁻¹::SparseMatrixCSC{Tv,Ti}) where {Ti<:Integer,Tv<:Real}

    n  = size(C⁻¹, 1)+1
    d⁺ = zeros(Float64, n)
    R  = zeros(n, n)
    C⁻ᵀ = permutedims(C⁻¹)

    spmul_dense!(R, C⁻ᵀ, C⁻¹)
    extract_diag!(d⁺,R)
    @. R = d⁺ + d⁺' - 2*R

    return R

end
