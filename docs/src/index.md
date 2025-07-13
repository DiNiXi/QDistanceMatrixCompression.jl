```@meta
CurrentModule = QDistanceMatrixCompression
```

# Q-Distance Matrix Compression

Documentation for [QDistanceMatrixCompression](https://github.com/DiNiXi/QDistanceMatrixCompression.jl).

```math
\begin{aligned}
L_{n-1} &= C C^\top \\
L_{n-1}^{-1} &= C^{-\top} C^{-1} \\
L_{n-1}^{-1} b_q &= C^{-\top} C^{-1} b_q \\
\end{aligned}
```

This is why we do `C \ b_q`.

```@index
```

```@autodocs
Modules = [QDistanceMatrixCompression]
```
