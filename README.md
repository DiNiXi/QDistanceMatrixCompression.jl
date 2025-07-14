# Q-ray QDistanceMatrixCompression

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dinixi.github.io/QDistanceMatrixCompression.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dinixi.github.io/QDistanceMatrixCompression.jl/dev/)
[![Build Status](https://github.com/dinixi/QDistanceMatrixCompression.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dinixi/QDistanceMatrixCompression.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dinixi/QDistanceMatrixCompression.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dinixi/QDistanceMatrixCompression.jl)

## Graphical Resistance Distance

The *resistance distance* between two nodes $𝑖$ and $𝑗$ in a graph, is the resistance between two points on a corresponding electrical network, where each edge is replaced by a resistance of one ohm.

The resistance distance accounts for all possible paths between the two
nodes and reflects the effects of branching, diffusion, and network dynamics. It has proven a robust alternative to the geodesic distance (i.e., shortest path length), which can be
more sensitive to variations in topological structure, edge weight distribution, or both—especially in real-world sparse networks. 

This package provides `Q-ray`, a high-performance Julia implementation that precomputes a compressed representation of the resistance distance matrix so that it can be queried (one-to-all and all-to-all) efficiently.