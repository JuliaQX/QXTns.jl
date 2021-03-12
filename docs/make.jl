using QXTn, Documenter

DocMeta.setdocmeta!(QXTn, :DocTestSetup, :(using QXTn); recursive=true)
makedocs(;
    modules=[QXTn],
    authors="QuantEx team",
    repo="https://github.com/JuliaQX/QXTn.jl/blob/{commit}{path}#L{line}",
    sitename="QXTn.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaQX.github.io/QXTn.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Data Structures" => "data_structures.md",
        "LICENSE" => "license.md"
    ],
)

deploydocs(;
    repo="github.com/JuliaQX/QXTn.jl",
)
