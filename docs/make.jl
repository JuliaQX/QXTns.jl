using QXTns, Documenter

DocMeta.setdocmeta!(QXTns, :DocTestSetup, :(using QXTns); recursive=true)
doctest(QXTns)
makedocs(;
    modules=[QXTns],
    authors="QuantEx team",
    repo="https://github.com/JuliaQX/QXTns.jl/blob/{commit}{path}#L{line}",
    sitename="QXTns.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaQX.github.io/QXTns.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Data Structures" => "data_structures.md",
        "LICENSE" => "license.md"
    ],
)

deploydocs(;
    repo="github.com/JuliaQX/QXTns.jl",
)
