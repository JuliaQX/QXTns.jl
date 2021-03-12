```@meta
CurrentModule = QXTn
```

# QXTn

QXTn is a Julia package with data structures and utilities for manipulating tensor networks.
As well as generic tensor network data structure, it also contains specific data structures
for handling tensor networks derived from quantum circuits. It was developed as part of the QuantEx project, one of the individual software projects of WP8 of PRACE 6IP.

It uses some features from [ITensors](https://github.com/ITensor/ITensors.jl) and [NDTensors](https://github.com/ITensor/NDTensors.jl) for representing tensors and indices and performing contractions.

## Installation

QXTn is a Julia package and can be installed using Julia's inbuilt package manager from the Julia REPL using.

```
import Pkg
Pkg.add("QXTn")
```

To ensure everything is working, the unittests can be run using

```
import Pkg; Pkg.test()
```

## Example usage

An example of creating a simple tensor network and contracting.

```
using QXTn

tn = TensorNetwork()

a, b, c, d = Index(2), Index(3), Index(5), Index(4)

# add a 2x3x5 rank tensor
push!(tn, [a, b, c], rand(2, 3, 5))
# add a 5x4 matrix
push!(tn, [c, d], rand(5, 4))

# contract network
simple_contraction!(tn)

# number of tensors after contraction
@show length(tn)

# resulting tensor has dimensions should have dimensions 2x3x4
@show size(first(tn))
```

## Contributing
Contributions from users are welcome and we encourage users to open issues and submit merge/pull requests for any problems or feature requests they have. The
CONTRIBUTING.md on the top level of the source folder has further details of the contribution guidelines.

## Building documentation

QXTn.jl uses [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) to generate documentation. To build the documentation locally run the following from the root folder.

The first time it is will be necessary to instantiate the environment to install dependencies

```
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

and then to build the documentation

```
julia --project=docs/ docs/make.jl
```

The generated document will be in the `docs/build` folder. To serve these locally one can
use the LiveServer package as

```
julia --project -e 'import Pkg; Pkg.add("LiveServer");
julia --project -e  'using LiveServer; serve(dir="docs/build")'
```

Or with python3 using from the `docs/build` folder using

```
python3 -m http.server
```

The generated documentation should now be viewable locally in a browser at `http://localhost:8000`.

## API Reference

```@index
```

```@autodocs
Modules = [QXTn]
```
