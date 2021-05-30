using ITensors
using LinearAlgebra
using NDTensors

"""
Tensor data structure for representing tensors and keeping track of hyper indices.
"""

export QXTensor, Index, hyperindices, contract_tensors, tensor_data

"""Datastructure representing tensors"""
mutable struct QXTensor
    rank::Int64
    indices::Array{<:Index}
    hyper_indices::Array{Array{Int64, 1}, 1}
    storage::AbstractArray
end

"""Custom show for QXTensors"""
function Base.show(io::IO, ::MIME"text/plain", t::QXTensor)
    print(io, "QXTensor, rank: $(t.rank), " *
              "dims: $(Tuple(dim.(t.indices))), " *
              "storage: $(typeof(t.storage)), " *
              "hyper_indices: $(t.hyper_indices)")
end

"""Custom size function"""
Base.size(a::QXTensor) = Tuple(dim.(a.indices))
"""Implement inds for QXTensor"""
ITensors.inds(a::QXTensor) = a.indices
"""Implement store for QXTensor"""
ITensors.store(a::QXTensor) = a.storage

"""
    QXTensor(a::T) where T <: Number

QXTensor constructor which creates a new instance of QXTensor corresponding to a scalar
"""
function QXTensor(a::Number)
    QXTensor(0, Array{Index, 1}(), Array{Array{Index, 1}, 1}(), fill(a, ()))
end

"""
    QXTensor(indices::Vector{<:Index},
             hyper_indices::Union{Nothing, Vector{<:Vector{Int64}}},
             storage::Union{Nothing, <: AbstractArray}=nothing;
             diagonal_check::Bool=true)

QXTensor constructor creates a new instance of QXTensor with the given indices
and hyper indices. If no storage data structure is given then a MockTensor of that shape
is added as the storage. If diagonal_check is true, it will automaticallly check which indices are hyper indices
and record in the hyper_indices field. If hyper_indices are given, then these are used.
"""
function QXTensor(indices::Vector{<:Index},
                  hyper_indices::Union{Nothing, Vector{<:Vector{Int64}}}=nothing,
                  storage::Union{Nothing, <: AbstractArray}=nothing;
                  diagonal_check::Bool=true)
    if storage === nothing
        storage = MockTensor(Tuple(dim.(indices)))
    end

    # if hyper indices not given, diagonal check enabled and not mock tensor,
    # then attempt to detect hyperindices directly from the tensor
    if !(storage isa MockTensor)
        if diagonal_check && hyper_indices === nothing
            hyper_indices = find_hyper_edges(storage)
        end
        # if it is not already in reduced form then we reduce it
        if length(hyper_indices) > 0 && ndims(storage) == length(indices)
            storage = reduce_tensor(storage, hyper_indices)
        end
    end

    if hyper_indices === nothing
        hyper_indices = Vector{Vector{Int64}}()
    end

    QXTensor(length(indices), indices, hyper_indices, storage)
end

QXTensor(i::Vector{<:Index}, t::Union{Nothing, <: AbstractArray}; kwargs...) =
         QXTensor(i, nothing, t; kwargs...)
QXTensor(t::Union{Nothing, <: AbstractArray}; kwargs...) =
         QXTensor([Index(x) for x in size(t)], nothing, t; kwargs...)

"""
    tensor_data(tensor::QXTensor; consider_hyperindices::Bool=true)

Get the data associated with the given tensor. If the consider_hyperindices flag is true
then the rank is reduced to merge related indices. For example for a 5 rank tensor where
the 2nd and 4th indices form a group of hyper indices, with this option set to true would
return a rank 4 tensor where the 2nd index. With hyperindices set to false a rank 5 tensor
is returned.
"""
function tensor_data(tensor::QXTensor; consider_hyperindices::Bool=true)
    data = store(tensor)
    if consider_hyperindices
        # data is stored in reduced form
        return data
    else
        hi_ranks = indices2ranks(tensor, hyperindices(tensor))
        return collect(expand_tensor(data, hi_ranks))
    end
end

function indices2ranks(tensor::QXTensor, hi::Vector{<:Vector{<:Index}})
    # create an array of the ranks from the groups of hyper indices
    hi_ranks = Array{Int64, 1}[]
    all_indices = inds(tensor)
    for group in hi
        push!(hi_ranks, map(x -> findfirst(y -> y == x, all_indices) ,group))
    end
    hi_ranks
end

"""
    hyperindices(t::QXTensor)

Function to get the hyper indices as an array of Indices. If the
all_indices flag is true, then all indices are returned, if false
then just the groups of 2 or more are returned.
"""
function hyperindices(t::QXTensor; all_indices=false)
    indices = copy(t.indices)
    hyper_indices = map(x -> Index[x], indices)
    for group in t.hyper_indices
        group = sort(group)
        append!(hyper_indices[group[1]], indices[group[2:end]])
        empty!.(hyper_indices[group[2:end]])
    end
    filter(x -> length(x) > (all_indices ? 0 : 1), hyper_indices)
end

"""
    disable_hyperindices!(t::QXTensor)

Function to disable use of hyper indices with this tensor by removing the hyper
indices and reshaping storage
"""
function disable_hyperindices!(t::QXTensor)
    t.storage = collect(expand_tensor(t.storage, t.hyper_indices))
    empty!(t.hyper_indices)
    nothing
end