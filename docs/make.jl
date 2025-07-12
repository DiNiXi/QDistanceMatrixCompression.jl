using QDistanceMatrixCompression
using Documenter

DocMeta.setdocmeta!(QDistanceMatrixCompression, :DocTestSetup, :(using QDistanceMatrixCompression); recursive=true)

makedocs(;
    modules=[QDistanceMatrixCompression],
    authors="Dimitris Floros, Nikos Pitsianis, Xiaobai Sun",
    sitename="QDistanceMatrixCompression.jl",
    format=Documenter.HTML(;
        canonical="https://fcdimitr.github.io/QDistanceMatrixCompression.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fcdimitr/QDistanceMatrixCompression.jl",
    devbranch="main",
)
