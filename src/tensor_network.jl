using DataStructures
using ITensors
using QXTns

using OMEinsum

# TensorNetwork struct and public functions
export next_tensor_id!
export TensorNetwork, bonds, simple_contraction, simple_contraction!, neighbours
export decompose_tensor!, replace_with_svd!
export contract_tn!, contract_pair!, replace_tensor_symbol!
export get_hyperedges, disable_hyperindices!, find_connected_indices
export contraction_indices, contract_pair

"""Tensor network data-structure"""
mutable struct TensorNetwork
    tensor_map::OrderedDict{Symbol, QXTensor}
    bond_map::OrderedDict{Index, Vector{Symbol}}
    next_id::Int64
end

# constructors
TensorNetwork() = TensorNetwork(OrderedDict{Symbol, QXTensor}(), OrderedDict{Index, Vector{Symbol}}(), 1)

"""
    TensorNetwork(array::Vector{<: QXTensor})

Outer constructor to create a tensor network object from an array of ITensor objects.
"""
function TensorNetwork(array::Vector{<: QXTensor})
    tensor_map = OrderedDict{Symbol, QXTensor}()
    bond_map = OrderedDict{Index, Vector{Symbol}}()
    next_id = 1
    for t in array
        tensor_id = Symbol("t$(next_id)")
        next_id += 1
        tensor_map[tensor_id] = t
        for bond in inds(t)
            if haskey(bond_map, bond)
                push!(bond_map[bond], tensor_id)
            else
                bond_map[bond] = [tensor_id]
            end
        end
    end
    TensorNetwork(tensor_map, bond_map, next_id)
end

Base.copy(tn::TensorNetwork) = TensorNetwork(deepcopy(tn.tensor_map), deepcopy(tn.bond_map), tn.next_id)
Base.length(tn::TensorNetwork) = length(tn.tensor_map)
Base.values(tn::TensorNetwork) = values(tn.tensor_map)
Base.iterate(tn::TensorNetwork) = iterate(values(tn))
Base.iterate(tn::TensorNetwork, state) = iterate(values(tn), state)
Base.eltype(::TensorNetwork) = QXTensor
Base.keys(tn::TensorNetwork) = keys(tn.tensor_map)
Base.getindex(tn::TensorNetwork, i::Symbol) = tn.tensor_map[i]
Base.haskey(tn::TensorNetwork, i::Symbol) = haskey(tn.tensor_map, i)
Base.getindex(tn::TensorNetwork, i::T) where T <: Index = tn.bond_map[i]
Base.haskey(tn::TensorNetwork, i::T) where T <: Index = haskey(tn.bond_map, i)
Base.show(io::IO, ::MIME"text/plain", tn::TensorNetwork) = print(io, "TensorNetwork(tensors => $(length(tn)), bonds => $(length(bonds(tn))))")

next_tensor_id!(tn::TensorNetwork) = begin tn.next_id += 1; return Symbol("t$(tn.next_id - 1)") end
bonds(tn::TensorNetwork) = keys(tn.bond_map)
disable_hyperindices!(tn) = begin disable_hyperindices!.(tn); nothing; end


"""
    hyperindices(tn::TensorNetwork, i::Symbol; global_hyperindices=true)

Find groups of hyper indices for the given tensor. When global_hyperindices is set to true, then
indices which are identified as hyperindices because of groups of hyperindices in conneted tensors
in the network are also included.
"""
function hyperindices(tn::TensorNetwork, i::Symbol; global_hyperindices=true, all_indices=false)
    if !global_hyperindices
        return hyperindices(tn[i], all_indices=all_indices)
    else
        all_hyperindices = Vector{Vector{Index}}()
        tensor_indices = copy(inds(tn[i]))
        while length(tensor_indices) > 0
            index = tensor_indices[1]
            connected_indices = find_connected_indices(tn, index)
            push!(all_hyperindices, connected_indices)
            setdiff!(tensor_indices, connected_indices)
        end
        return filter(x -> length(x) >= (all_indices ? 1 : 2), all_hyperindices)
    end
end

"""
    tensor_data(tn::TensorNetwork, i::Symbol; consider_hyperindices=true, global_hyperindices=true)

Retrieve the tensor data for the given tensor. If the consider_hyperindices flag is true then
then the data is reshaped to take into account the local hyperindices of the tensor. If the global_hyperindices
index is also true then groups of hyperindices related via hyperindices for other tensors in the network are
also considered.
"""
function tensor_data(tn::TensorNetwork, i::Symbol; consider_hyperindices=true, global_hyperindices=true)
    if consider_hyperindices && global_hyperindices
        global_hi = hyperindices(tn, i; global_hyperindices=true)
        # filter any indices not connected to tensor
        global_hi = map(global_hi) do x
            x = filter(y -> y ∈ inds(tn[i]), x)
        end
        global_hi = filter(x -> length(x) > 1, global_hi)

        local_hi = hyperindices(tn, i; global_hyperindices=false)
        missing_hi = false
        for g in global_hi
            if findfirst(x -> Set(x) == Set(g), local_hi) === nothing
                missing_hi = true
            end
        end
        if missing_hi
            t = store(tn[i])
            if length(local_hi) > 0
                local_hi_ranks = indices2ranks(tn[i], local_hi)
                t = expand_tensor(t, local_hi_ranks)
            end
            global_hi_ranks = indices2ranks(tn[i], global_hi)
            t = reduce_tensor(t, global_hi_ranks)
            return t
        end
    end
    # can delegate to tensor specific tensor_data function
    return tensor_data(tn.tensor_map[i]; consider_hyperindices)
end

"""
    neighbours(tn::TensorNetwork, tensor::Symbol)

Function get the symbols of the neighbouring tensors
"""
function neighbours(tn::TensorNetwork, tensor::Symbol)
    tensor_indices = inds(tn[tensor])
    connected_tensors = unique(vcat([tn[x] for x in tensor_indices]...))
    setdiff(connected_tensors, [tensor])
end

"""
    merge(a::TensorNetwork, b::TensorNetwork)

Join two networks together
"""
function Base.merge(a::TensorNetwork, b::TensorNetwork)
    c = copy(a)
    for b_tensor in b
        push!(c, b_tensor)
    end
    c
end

"""
    push!(tn::TensorNetwork,
          indices::Vector{Index},
          data::Array{T, N}
          tid::Union{Nothing, Symbol}=nothing) where {T, N}

Function to add a tensor to the tensor network.

# Keywords
- `tid::Union{Nothing, Symbol}=nothing`: the id for the new tensor in `tn`. An id is
generated if one is not set.
"""
function Base.push!(tn::TensorNetwork,
                    indices::Vector{<:Index},
                    data::Array{T, N};
                    tid::Union{Nothing, Symbol}=nothing) where {T, N}
    @assert size(data) == Tuple(dim.(indices))
    tensor = QXTensor(indices, data)
    if tid === nothing tid = next_tensor_id!(tn) end
    @assert !(tid in keys(tn))
    tn.tensor_map[tid] = tensor
    for bond in indices
        if haskey(tn.bond_map, bond)
            push!(tn.bond_map[bond], tid)
        else
            tn.bond_map[bond] = [tid]
        end
    end
    tid
end

"""
    push!(tn::TensorNetwork,
          tensor::QXTensor;
          tid::Union{Nothing, Symbol}=nothing)

Function to add a tensor to the tensor network.

# Keywords
- `tid::Union{Nothing, Symbol}=nothing`: the id for the new tensor in `tn`. An id is
generated if one is not set.
"""
function Base.push!(tn::TensorNetwork,
                    tensor::QXTensor;
                    tid::Union{Nothing, Symbol}=nothing)
    if tid === nothing tid = next_tensor_id!(tn) end
    # TODO: It might be a good idea to assert tid doesn't already exist in tn.
    tn.tensor_map[tid] = tensor
    for bond in inds(tensor)
        if haskey(tn.bond_map, bond)
            push!(tn.bond_map[bond], tid)
        else
            tn.bond_map[bond] = [tid]
        end
    end
    tid
end

"""
    simple_contraction(tn::TensorNetwork)

Function to perfrom a simple contraction, contracting all tensors in order.
Only useful for very small networks for testing.
"""
function simple_contraction(tn::TensorNetwork)
    a = QXTensor(1.)
    for t in tn a = contract_tensors(a, t) end
    tensor_data(a; consider_hyperindices=false)
end

"""
    simple_contraction!(tn::TensorNetwork)

Function to perfrom a simple contraction, contracting all tensors in order.
Only useful for very small networks for testing.
"""
function simple_contraction!(tn::TensorNetwork)
    tensor_syms = collect(keys(tn))
    A = tensor_syms[1]
    for B in tensor_syms[2:end]
        A = contract_pair!(tn, A, B)
    end
    store(tn[A])
end

"""
    contract_pair(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol; mock::Bool=false)

Contract the tensors in 'tn' with ids 'a_sym' and 'b_sym'. If the mock flag is true then the
new tensor will be a mock tensor with the right dimensions but without the actual data.
"""
function contract_pair(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol; mock::Bool=false)
    r = contraction_indices(tn, a_sym, b_sym)

    # identify positions of local hyper index groups in c
    pos = 1
    c_hyper_indices = map(r.c_indices) do g
        pos += length(g)
        collect(pos-length(g):pos-1)
    end
    c_hyper_indices = convert(Vector{Vector{Int}}, c_hyper_indices)
    filter!(x -> length(x) > 1, c_hyper_indices)

    if mock || (tn[a_sym].storage isa MockTensor || tn[b_sym].storage isa MockTensor)
        # flatten c_indices to get indices
        c = QXTensor(convert(Vector{Index}, vcat(r.c_indices...)), c_hyper_indices)
    else
        c_data = EinCode((Tuple(r.a_labels), Tuple(r.b_labels)), Tuple(r.c_labels))(tensor_data(tn, a_sym), tensor_data(tn, b_sym))
        c = QXTensor(convert(Vector{Index}, vcat(r.c_indices...)), c_hyper_indices, c_data)
    end
    c
end

"""
    contract_pair!(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol, c_sym::Symbol=:_; mock::Bool=false)

Contract the tensors in 'tn' with ids 'a_sym' and 'b_sym'. If the mock flag is true then the
new tensor will be a mock tensor with the right dimensions but without the actual data.

The resulting tensor is stored in `tn` under the symbol `c_sym` if one is provided, otherwise
a new id is created for it.
"""
function contract_pair!(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol, c_sym::Symbol=:_;
                        mock::Bool=false)

    c = contract_pair(tn, a_sym, b_sym; mock=mock)

    # Get and contract the tensors a and b to create tensor c
    a = tn.tensor_map[a_sym]
    b = tn.tensor_map[b_sym]
    c_sym == :_ && (c_sym = next_tensor_id!(tn))

    # Remove the contracted indices from the bond map in tn. Also, replace all references
    # in tn to tensors a and b with a reference to tensor c
    common_indices = intersect(inds(a), inds(b))
    for ind in common_indices
        delete!(tn.bond_map, ind)
    end
    for ind in symdiff(inds(a), inds(b))
        tn.bond_map[ind] = replace(tn.bond_map[ind], a_sym => c_sym, b_sym => c_sym)
    end

    # Add tensor c to the tn and remove both a and b
    tn.tensor_map[c_sym] = c
    delete!(tn.tensor_map, a_sym); delete!(tn.tensor_map, b_sym)
    c_sym
end

"""
    contraction_indices(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol)

Function to work out the contraction indices that would be used to contract the given
tensors. Expected indices in Einstein notation using positive integers
"""
function contraction_indices(tn::TensorNetwork, a_sym::Symbol, b_sym::Symbol)
    a_indices = hyperindices(tn, a_sym; all_indices=true)
    b_indices = hyperindices(tn, b_sym; all_indices=true)
    common_indices = intersect(inds(tn[a_sym]), inds(tn[b_sym]))
    remaining_indices = symdiff(inds(tn[a_sym]), inds(tn[b_sym]))

    # label a_indices in order
    a_labels = collect(1:length(a_indices))
    num_unique = length(a_indices)
    b_labels = map(b_indices) do g
        pos = findfirst(x -> length(intersect(g, x)) > 0, a_indices)
        if pos === nothing
            pos = num_unique += 1
        end
        pos
    end

    # to work out final indices we look at the indices which survive from each tensor
    # find all index groups from a that are not fully contracted
    c_indices = Vector{Vector{Index}}()
    a_labels′ = map(zip(a_labels, a_indices)) do (i, g)
        if length(setdiff(g, common_indices)) == 0
            return nothing
        else
            push!(c_indices, intersect(g, remaining_indices))
            return i
        end
    end

    # find any index groups from b that survive
    # these are groups that are not fully contracted or also present in a
    b_labels′ = map(zip(b_labels, b_indices)) do (i, g)
        if any(map(x -> length(intersect(g, x)) > 0, a_indices))
            return nothing
        else
            push!(c_indices, intersect(g, remaining_indices))
            return i
        end
    end
    c_labels = convert(Vector{Int}, filter(x -> x !== nothing, [a_labels′..., b_labels′...]))
    (a_labels = a_labels, b_labels = b_labels, c_labels = c_labels, c_indices = c_indices)
end

"""
    contract_tn!(tn::TensorNetwork, plan)

Contract the indices of 'tn' according to 'plan'.
"""
function contract_tn!(tn::TensorNetwork, plan::Array{NTuple{3, Symbol}, 1})
    for (A_id, B_id, C_id) in plan
        @assert haskey(tn.tensor_map, A_id)
        @assert haskey(tn.tensor_map, B_id)
        contract_pair!(tn, A_id, B_id, C_id)
    end

    # Contract any disjoint tensors that may remain before returning the result.
    simple_contraction!(tn)
end

"""
    decompose_tensor!(tn::TensorNetwork,
                      tensor_id::Symbol,
                      left_indices::Array{<:Index, 1};
                      contract_S_with::Symbol=:V,
                      kwargs...)

Function to decompose a tensor in a tensor network using svd.

# Keywords
- `contract_S_with::Symbol=:V`: the maxtrix which should absorb the matrix of singular values
- `maxdim::Int`: the maximum number of singular values to keep.
- `mindim::Int`: the minimum number of singular values to keep.
- `cutoff::Float64`: set the desired truncation error of the SVD.
"""
function decompose_tensor!(tn::TensorNetwork,
                           tensor_id::Symbol,
                           left_indices::Array{<:Index, 1};
                           contract_S_with::Symbol=:V,
                           kwargs...)
    # Decompose the tensor into its svd factors.
    U_id, S_id, V_id = replace_with_svd!(tn, tensor_id, left_indices; kwargs)

    # Absorb the singular values tensor into eith U or V.
    if contract_S_with == :V_id
        return U_id, contract_pair!(tn, S_id, V_id)
    else
        return contract_pair!(tn, S_id, U_id), V_id
    end
end

# """
#     replace_with_svd!(tn::TensorNetwork,
#                       tensor_id::Symbol,
#                       left_indices::Array{<:Index, 1};
#                       kwargs...)

# Function to replace a tensor in a tensor network with its svd.

# The indices contained in 'left_indices' are considered the row indices of the tensor when
# the svd is performed.

# # Keywords
# - `maxdim::Int`: the maximum number of singular values to keep.
# - `mindim::Int`: the minimum number of singular values to keep.
# - `cutoff::Float64`: set the desired truncation error of the SVD.
# """
# function replace_with_svd!(tn::TensorNetwork,
#                            tensor_id::Symbol,
#                            left_indices::Array{<:Index, 1};
#                            kwargs...)
#     # Get the tensor and decompose it.
#     tensor = tn.tensor_map[tensor_id]
#     U, S, V = svd(convert(ITensor, tensor), left_indices; use_absolute_cutoff=true, kwargs...)
#     S_data = reshape(collect(Diagonal(store(S))), prod(size(S)))
#     S_QXTensor = QXTensor(S_data, collect(inds(S)), diagonal_check=false)

#     # Remove the original tensor and add its svd factors to the network.
#     delete!(tn, tensor_id)
#     U_id = push!(tn, convert(QXTensor, U))
#     S_id = push!(tn, S_QXTensor)
#     V_id = push!(tn, convert(QXTensor, V))
#     U_id, S_id, V_id
# end

"""
    delete!(tn::TensorNetwork, tensor_id::Symbol)

Function to remove a tensor from a tensor network.
"""
function Base.delete!(tn::TensorNetwork, tensor_id::Symbol)
    tensor = tn.tensor_map[tensor_id]
    for index in inds(tensor)
        # remove tensor_id from tn.bond_map[index]
        filter!(id -> id ≠ tensor_id, tn.bond_map[index])
    end
    delete!(tn.tensor_map, tensor_id)
end

"""
    replace_tensor_symbol!(tn::TensorNetwork, orig_sym::Symbol, new_sym::Symbol)

Replace the given symbol with the given new symbol
"""
function replace_tensor_symbol!(tn::TensorNetwork, orig_sym::Symbol, new_sym::Symbol)
    tensor = tn[orig_sym]
    for i in inds(tensor)
        replace!(tn[i], orig_sym => new_sym)
    end
    tn.tensor_map[new_sym] = tensor
    delete!(tn.tensor_map, orig_sym)
end

"""
    get_hyperedges(tn::TensorNetwork)::Array{Array{Symbol, 1}, 1}

Return an array of hyperedges in the given tensornetwork `tn`.

Hyperedges are represented as arrays of tensor symbols.
"""
function get_hyperedges(tn::TensorNetwork)
    hyperedges = Array{Array{Symbol, 1}, 1}()
    edges = collect(bonds(tn))
    while length(edges) > 0
    # for edge in edges
        edge = edges[1]
        hyper_indices = find_connected_indices(tn, edge)
        push!(hyperedges, union(map(x -> tn[x], hyper_indices)...))
        setdiff!(edges, hyper_indices)
    end
    hyperedges
end

"""
    find_connected_indices(tn::TensorNetwork, bond::Index)

Given a tensor network and an index in the network, find all indices that are related via hyper edge
relations. Involves recurisively checking bonds connected to neighbouring tensors of any newly
related edges found. Returns an array of all edges in the group including the initial edge.
"""
function find_connected_indices(tn::TensorNetwork, bond::Index)
    tensors_to_visit = Set{Symbol}()
    push!.([tensors_to_visit], tn[bond])
    related_edges = Set{Index}([bond])
    while length(tensors_to_visit) > 0
        tensor_sym = pop!(tensors_to_visit)
        for g in hyperindices(tn[tensor_sym])
            if length(intersect(related_edges, g)) > 0
                new_edges = setdiff(g, related_edges)
                for e in new_edges
                    push!(related_edges, e)
                    for t in tn[e]
                        push!(tensors_to_visit, t)
                    end
                end
            end
        end
    end
    collect(related_edges)
end

"""
    contraction_indices((a::QXTensor,b::QXTensor)

Function to work out the contraction indices that would be used to contract the given
tensors. Exptected indices in Einstein notation using positive integers
"""
function contraction_indices(a::QXTensor, b::QXTensor)
    tn = TensorNetwork([a, b])
    contraction_indices(tn, keys(tn)...)
end

"""
    contract_tensors(a::QXTensor, b::QXTensor; mock::Bool=false)

Function to contract two QXTensors and return another QXTensor. If the mock flag
is false or either of the input tensors use MockTensor then the storage for the final
tensor will be of type MockTensor.
"""
function contract_tensors(a::QXTensor, b::QXTensor; mock::Bool=false)
    tn = TensorNetwork([a, b])
    contract_pair(tn, keys(tn)...; mock=mock)
end
