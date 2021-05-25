using DataStructures
using ITensors
using QXTns

# TensorNetwork struct and public functions
export next_tensor_id!
export TensorNetwork, bonds, simple_contraction, simple_contraction!, neighbours
export decompose_tensor!, replace_with_svd!
export contract_tn!, contract_pair!, replace_tensor_symbol!, contract_ncon_indices
export get_hyperedges, disable_hyperindices!, find_connected_indices

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
tensor_data(tn::TensorNetwork, i::Symbol; kwargs...) = tensor_data(tn.tensor_map[i]; kwargs...)
disable_hyperindices!(tn) = begin map(t -> filter!(x -> false, t.hyper_indices), tn); return nothing end

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
    tensor = QXTensor(data, indices)
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
    store(reduce(contract_tensors, tn, init=QXTensor(1.)))
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
    contract_pair!(tn::TensorNetwork, A_id::Symbol, B_id::Symbol; mock::Bool=false)

Contract the tensors in 'tn' with ids 'A_id' and 'B_id'. If the mock flag is true then the
new tensor will be a mock tensor with the right dimensions but without the actual data.

The resulting tensor is stored in `tn` under the symbol `C_id` if one is provided, otherwise
a new id is created for it.
"""
function contract_pair!(tn::TensorNetwork, A_id::Symbol, B_id::Symbol, C_id::Symbol=:_;
                        mock::Bool=false)
    # Get and contract the tensors A and B to create tensor C.
    A = tn.tensor_map[A_id]
    B = tn.tensor_map[B_id]
    C_id == :_ && (C_id = next_tensor_id!(tn))
    C = contract_tensors(A, B, mock=mock)

    # Remove the contracted indices from the bond map in tn. Also, replace all references
    # in tn to tensors A and B with a reference to tensor C.
    common_indices = intersect(inds(A), inds(B))
    for ind in common_indices
        delete!(tn.bond_map, ind)
    end
    for ind in setdiff(union(inds(A), inds(B)), common_indices)
        tn.bond_map[ind] = replace(tn.bond_map[ind], A_id=>C_id, B_id=>C_id)
    end

    # Add tensor C to the tn and remove both A and B.
    tn.tensor_map[C_id] = C
    delete!(tn.tensor_map, A_id); delete!(tn.tensor_map, B_id)
    C_id
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

"""
    replace_with_svd!(tn::TensorNetwork,
                      tensor_id::Symbol,
                      left_indices::Array{<:Index, 1};
                      kwargs...)

Function to replace a tensor in a tensor network with its svd.

The indices contained in 'left_indices' are considered the row indices of the tensor when
the svd is performed.

# Keywords
- `maxdim::Int`: the maximum number of singular values to keep.
- `mindim::Int`: the minimum number of singular values to keep.
- `cutoff::Float64`: set the desired truncation error of the SVD.
"""
function replace_with_svd!(tn::TensorNetwork,
                           tensor_id::Symbol,
                           left_indices::Array{<:Index, 1};
                           kwargs...)
    # Get the tensor and decompose it.
    tensor = tn.tensor_map[tensor_id]
    U, S, V = svd(convert(ITensor, tensor), left_indices; use_absolute_cutoff=true, kwargs...)
    S_data = reshape(collect(Diagonal(store(S))), prod(size(S)))
    S_QXTensor = QXTensor(S_data, collect(inds(S)))

    # Remove the original tensor and add its svd factors to the network.
    delete!(tn, tensor_id)
    U_id = push!(tn, convert(QXTensor, U))
    S_id = push!(tn, S_QXTensor)
    V_id = push!(tn, convert(QXTensor, V))
    U_id, S_id, V_id
end

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
    contract_ncon_indices(tn::TensorNetwork, A_sym::Symbol, B_sym)

Function return indices in ncon format for contraction of tensors with given symbols.
Returns two tuples for indices in each with convention that negative values are remaining
indices and positive values are indices being contracted over.

For example if (1, -1), (-2, 1) is returned, this menas that the first index of tensor A
A is contracted with the second index of  tensor B and the resulting tensor will have
indices corresponding to the second index of the first tensor and first index of the second
tensor.
"""
function contract_ncon_indices(tn::TensorNetwork, A_sym::Symbol, B_sym::Symbol)
    _contract_ncon_indices(IndexSet(inds(tn[A_sym])), IndexSet(inds(tn[B_sym])))
end

"""
    _contract_ncon_indices(A_inds::IndexSet{M}, B_inds::IndexSet{N}) where {M, N}

Function return indices in ncon format for contraction of tensors with given index sets.
Returns two tuples for indices in each with convention that negative values are remaining
indices and positive values are indices being contracted over.

For example if (1, -1), (-2, 1) is returned, this menas that the first index of tensor A
A is contracted with the second index of  tensor B and the resulting tensor will have
indices corresponding to the second index of the first tensor and first index of the second
tensor.
"""
function _contract_ncon_indices(A_inds::IndexSet{M}, B_inds::IndexSet{N}) where {M, N}
    labels = ITensors.compute_contraction_labels(A_inds, B_inds)
    # ITensors uses a different convention with negatie and positive reversed and
    # find lowest positive
    all_positive_labels = [x for x in vcat(collect.(labels)...) if x > 0]
    offset = length(all_positive_labels) > 0 ? minimum(all_positive_labels) - 1 : 0
    [Tuple([x > 0 ? -(x - offset) : -x for x in ls]) for ls in labels]
end

"""
    replace_tensor_symbol!(tn::TensorNetwork, orig_sym::Symbol, new_sym::Symbol)

Replace the given symbol with the given new symbol
"""
function replace_tensor_symbol!(tn::TensorNetwork, orig_sym::Symbol, new_sym::Symbol)
    tensor = tn[orig_sym]
    for ind in inds(tensor)
        replace!(tn[ind], orig_sym => new_sym)
    end
    tn.tensor_map[new_sym] = tensor
    delete!(tn.tensor_map, orig_sym)
end


"""
    get_hyperedges(tn::TensorNetwork)::Array{Array{Symbol, 1}, 1}

Return an array of hyperedges in the given tensornetwork `tn`.

Hyperedges are represented as arrays of tensor symbols.
"""
function get_hyperedges(tn::TensorNetwork)::Array{Array{Symbol, 1}, 1}
    # hyperedges are represented as arrays of tensor symbols.
    hyperedges = Array{Array{Symbol, 1}, 1}()

    # Create an array of edges in the network which have not yet been assigned to a
    # hyperedge. Also create a queue which will be used to assign edges to hyperedges.
    edges = collect(bonds(tn))
    q = Queue{Index}()

    while !isempty(edges)
        # For the next edge in the network, which has not yet been assigned to a hyperedge,
        # create a new hyperedge. Then add the edge to the queue where it will wait to be
        # assigned to the new hyperedge.
        push!(hyperedges, [])
        enqueue!(q, pop!(edges))

        while !isempty(q)
            # While the queue is not empty, take the next edge in the queue and append
            # the corresponding tensor symbols to the new hyperedge.
            edge = dequeue!(q)
            tensors = tn.bond_map[edge]
            hyperedges[end] = union(hyperedges[end], tensors)

            # Check if the neighbouring edges, of the edge just added to the new hyperedge,
            # belong to the same hyperedge. If they do, add them to the queue.
            for tensor_symbol in tensors
                tensor = tn.tensor_map[tensor_symbol]
                i = findfirst(group -> edge in tensor.indices[group], tensor.hyper_indices)
                if !(i === nothing)
                    # Create an array of neighbouring edges not yet assigned to the
                    # hyperedge
                    connected_edges = Array{Index, 1}()
                    for j in tensor.hyper_indices[i]
                        index = tensor.indices[j]
                        if !(index == edge) && index in edges
                            push!(connected_edges, index)
                        end
                    end
                    # Add these edges to the queue where they'll be assigned to the new
                    # hyperedge and remove them from the array of edges not yet assigned to
                    # a hyperedge.
                    for e in connected_edges enqueue!(q, e) end
                    setdiff!(edges, connected_edges)
                end
            end
        end
    end
    hyperedges
end

"""
    find_connected_indices(tn::TensorNetwork, bond::Index)

Given a tensor network and an index in the network, find all indices that are related via hyper edge
relations. Involves recurisively checking bonds connected to neighbouring tensors of any newly
related edges found. Returns an array in all edges in the group including the intial edge.
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