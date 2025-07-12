using QDistanceMatrixCompression
using Documenter

DocMeta.setdocmeta!(QDistanceMatrixCompression, :DocTestSetup, :(using QDistanceMatrixCompression); recursive=true)

makedocs(;
    modules=[QDistanceMatrixCompression],
    authors="Dimitris Floros, Nikos Pitsianis, Xiaobai Sun",
    sitename="QDistanceMatrixCompression.jl",
    format=Documenter.HTML(;
        canonical="https://dinixi.github.io/QDistanceMatrixCompression.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DiNiXi/QDistanceMatrixCompression.jl",
    devbranch="main",
)
