using QDistanceMatrixCompression

using Graphs, LinearAlgebra, SparseArrays
using ResistanceDistance: ResistanceDistance

using Test

@testset "QDistanceMatrixCompression.jl" begin

  @testset "Barabasi-Albert Graph" begin
    # Create a Barabási-Albert graph with 10 nodes and 2 edges added at each step
    G = barabasi_albert(100, 5)

    # Compute the resistance distance matrix using external library
    R_library = ResistanceDistance.resistance_distance_matrix(G)

    @testset "$(version) method" for version in [:naive, :backsolve, :compact]
      # Compute the resistance distance matrix using the specified method
      R_method = QDistanceMatrixCompression.resistance_distance(G, version)

      # Check if the two matrices are equal
      @test R_method ≈ R_library
    end

  end

  # Include GraphicalDistanceMatrix tests
  include("graphical_distance_matrix_tests.jl")

end
