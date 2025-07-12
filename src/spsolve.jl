"""
    spsolve(A::SparseMatrixCSC, B::SparseMatrixCSC) -> SparseMatrixCSC

Solve the system AX = B where A is a lower triangular sparse matrix (SparseMatrixCSC)
and B is a sparse matrix (SparseMatrixCSC) representing the right-hand side.

# Arguments
- `A::SparseMatrixCSC`: A lower triangular sparse matrix.
- `B::SparseMatrixCSC`: A sparse matrix representing the right-hand side.

# Returns
- `X::SparseMatrixCSC`: The solution matrix in sparse format.

# Notes
- Assumes that `A` is lower triangular and non-singular.
"""
function spsolve(A::SparseMatrixCSC, B::SparseMatrixCSC, tau::Float64 = Inf)::SparseMatrixCSC
    # Ensure A is square
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square"))

    # Ensure dimensions match
    size(A, 1) == size(B, 1) || throw(ArgumentError("Matrix dimensions do not match"))

    # Solve the system
    X = solvemat(A,B,tau)

    return X
end


function transitiveclosure(L::SparseMatrixCSC{Tv,Ti},Jlen,CJ,mark;countonly=false) where {Tv,Ti<:Integer}
    b = 0
    c = Jlen
    for i=1:Jlen
        mark[CJ[i]] = true
    end
    cp = L.colptr
    rv = L.rowval
    while b<c
        b+=1
        i = CJ[b]
        p = cp[i]
        q = cp[i+1]-1
        for i=p:q
            j = rv[i]
            if !mark[j]
                c+=1
                mark[j]=true
                CJ[c]=j
            end
        end
    end
    for i=1:c
        mark[CJ[i]] = false
    end
    if countonly
        return c
    end
    sort!(view(CJ,1:c))
    return c
end

function solvemat(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti},tau::Float64) where {Tv,Ti<:Integer}
    @assert size(L,2)==size(B,1)
    cp = B.colptr
    rv = B.rowval
    nz = B.nzval
    CP = Vector{Ti}(undef,B.n+1)
    CP[1] = 1
    CJ = Vector{Ti}(undef,L.n)
    mark = falses(L.n)
    for i in 1:B.n
        p = cp[i]-1
        q = cp[i+1]-1
        Jlen = q-p
        for j=1:Jlen
            CJ[j]=rv[j+p]
        end
        CP[i+1]=CP[i]+transitiveclosure(L,Jlen,CJ,mark,countonly=true)
        CP[i+1] > tau && return spzeros(0,0)
    end
    N = CP[end]-1
    RV = Vector{Ti}(undef,N)
    NZ = Vector{Tv}(undef,N)
    x = zeros(Tv,L.n)
    for i = 1:B.n
        p = cp[i]
        q = cp[i+1]-1
        J = view(rv,p:q)
        p -= 1
        for j=1:length(J)
            x[J[j]] = nz[j+p]
        end
        d = solvevec(L,x,J,CJ,mark)
        c = CP[i]-1
        for j=1:d
            k = CJ[j]
            RV[c+j] = k
            NZ[c+j] = x[k]
            x[CJ[j]] = 0
        end
    end
    SparseMatrixCSC{Tv,Ti}(B.m,B.n,CP,RV,NZ)
end

function solvevec(L::SparseMatrixCSC{Tv,Ti},x::Vector{Tv},J,CJ,mark) where {Tv,Ti<:Integer}
    @assert size(L,2)==length(x)
    m = length(J)
    for i=1:m
        CJ[i] = J[i]
    end
    c = transitiveclosure(L,m,CJ,mark)
    cp = L.colptr
    rv = L.rowval
    nz = L.nzval
    (a,b,dir) = (1,c,1)
    for i=a:dir:b
        j = CJ[i]
        p = cp[j]
        q = cp[j+1]-1
        @assert rv[p]==j
        x[j] /= nz[p]
        p+=1
        alpha = x[j]
        for k=p:q
            x[rv[k]] -= alpha*nz[k]
        end
    end
    return c
end
