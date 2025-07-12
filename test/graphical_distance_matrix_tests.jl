using QDistanceMatrixCompression
using Graphs, LinearAlgebra, SparseArrays, Test, Random

function weighted_random(n, d; rng = MersenneTwister(42))

    A = sprand(rng, n, n, d / n)
    A = A + A'
    A = A - spdiagm(diag(A))

end

seed = 42
rng = MersenneTwister(seed)

@testset "GraphicalDistanceMatrix" begin
    # Create a Barabási-Albert graph with 10 nodes and 2 edges added at each step
    @testset "dataset: $(dataset)" for dataset in [(_ -> barabasi_albert(10, 2; complete=true, rng=rng)), (_ -> watts_strogatz(10, 5, 0.01; rng=rng)), (_ -> weighted_random(10, 3; rng=rng))]
        G = dataset(nothing)
        while G isa AbstractMatrix ? !is_connected(SimpleGraph(G)) : !is_connected(G)
            G = dataset(nothing)
        end
        n = size(G)
        # Test construction
        @testset "Construction" begin
            # From a graph
            R1 = GraphicalDistanceMatrix(G)
            @test size(R1) == (10, 10)

            # From an adjacency matrix
            A = G isa Graph ? adjacency_matrix(G) : G
            R2 = GraphicalDistanceMatrix(A)
            @test size(R2) == (10, 10)

            # Test that the matrix is not computed initially
            @test R1.computed == false
            @test R1.matrix === nothing
        end

        # Test computation and access
        @testset "Computation and Access" begin
            R = GraphicalDistanceMatrix(G)

            # Access should trigger computation
            val = R[1, 2]
            @test typeof(val) <: Real
            @test val >= 0  # Resistance distances are non-negative

            # Test full matrix conversion
            mat = Matrix(R)
            @test size(mat) == (10, 10)
            @test isapprox(mat, mat')  # Resistance distance matrix should be symmetric

            # Test diagonal elements are zero (resistance distance to self is zero)
            for i in 1:10
                @test isapprox(R[i, i], 0, atol=1e-10)
            end
        end

        # Test query function
        @testset "Query Function" begin
            R = GraphicalDistanceMatrix(G)
            nodes = [1, 3, 5]
            submat = query(R, nodes)
            @test size(submat) == (3, 3)
            @test isapprox(submat, submat', rtol=1e-10) # Should be symmetric

            # Access individual elements and compare with query
            for i in 1:length(nodes), j in 1:length(nodes)
                @test isapprox(R[nodes[i], nodes[j]], submat[i, j], rtol=1e-10)
            end
        end

        # Test equivalence with direct computation
        @testset "Equivalence with direct computation" begin
            R = GraphicalDistanceMatrix(G)
            direct = resistance_distance(G, :compact)

            for i in 1:10, j in 1:10
                @test isapprox(R[i, j], direct[i, j], rtol=1e-10)
            end
        end

        # Test equivalence with direct computation
        @testset "Equivalence with multiple options" begin
            direct = resistance_distance(G, :compact)
            R = GraphicalDistanceMatrix(G; force_laplacian=true)
            Rg = GraphicalDistanceMatrix(G; force_chol_inv=true)
            Rc = GraphicalDistanceMatrix(G; force_chol=true)


            for i in 1:10, j in 1:10
                @test isapprox(R[i, j], direct[i, j], rtol=1e-10)
                @test isapprox(Rg[i, j], direct[i, j], rtol=1e-10)
                @test isapprox(Rc[i, j], direct[i, j], rtol=1e-10)
            end

            nodes = [1, 3, 5]
            @test isapprox(direct[nodes, nodes], query(R, nodes), rtol=1e-10)
            @test isapprox(direct[nodes, nodes], query(Rg, nodes), rtol=1e-10)
            @test isapprox(direct[nodes, nodes], query(Rc, nodes), rtol=1e-10)

        end

        # Test equivalence with direct computation
        @testset "Equivalence with custom permutation" begin
            A = adjacency_matrix(binary_tree(8))
            n = size(A, 1)
            direct = resistance_distance(A, :compact)
            perm = n:-1:1
            R = GraphicalDistanceMatrix(A, perm; force_laplacian=true)
            Rg = GraphicalDistanceMatrix(A, perm; force_chol_inv=true)
            Rc = GraphicalDistanceMatrix(A, perm; force_chol=true)

            nq = 11 # Number of queries
            for i in 1:10 # Number of trials
                nodes = randperm(rng, n)[1:nq]
                @test isapprox(R[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
                @test isapprox(Rg[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
                @test isapprox(Rc[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
            end

        end

        # Test equivalence with direct computation
        @testset "Equivalence with custom permutation in full formation" begin
            A = adjacency_matrix(binary_tree(8))
            n = size(A, 1)
            perm = n:-1:1
            A = A[perm, perm]
            fix_perm = true
            direct = resistance_distance(A, :compact, fix_perm)
            R = GraphicalDistanceMatrix(A; force_laplacian=true)
            Rg = GraphicalDistanceMatrix(A; force_chol_inv=true)
            Rc = GraphicalDistanceMatrix(A; force_chol=true)

            nq = 11 # Number of queries
            for i in 1:10 # Number of trials
                nodes = randperm(rng, n)[1:nq]
                @test isapprox(R[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
                @test isapprox(Rg[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
                @test isapprox(Rc[nodes, nodes], direct[nodes, nodes], rtol=1e-10)
            end

        end

        # Test matrix computation and conversion
        @testset "Matrix computation" begin
            R = GraphicalDistanceMatrix(G)
            # Initially not computed
            @test !R.computed

            # Getting the full matrix triggers computation
            M = Matrix(R)
            @test R.computed
            @test !isnothing(R.matrix)
            @test size(M) == (10, 10)
        end

    end
end
