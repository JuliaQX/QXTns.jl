"""
In this module we define a MockTensor which inherits from the NDTensors.TensorStorage.
This can be used as the store for ITensor objects, but only stores the dimensions and so
can be used for tracking the dimension of contractions which would not be possible if
storing all the data.
"""

using NDTensors
using ITensors

export MockTensor


""" Tensor store struct that just tracks tensor dimensions"""
struct MockTensor <: NDTensors.TensorStorage{ComplexF64}
    size::Array{Int64, 1}
end

MockTensor(size::NTuple{N, Int64}) where N = MockTensor(collect(size))

"""Overload functions from base to make MockTensor usable"""
Base.copy(a::MockTensor) = MockTensor(a.size)
Base.length(a::MockTensor) = prod(size(a))
Base.size(a::MockTensor) = a.size
Base.getindex(::MockTensor, ::Int64) = NaN
Base.getindex(::MockTensor, i...) = NaN
NDTensors.tensor(a::MockTensor, inds) = NDTensors.Tensor{ComplexF64, length(inds), MockTensor, ITensors.IndexSet}(inds, a)
Base.getindex(::NDTensors.Tensor{ComplexF64, N, StoreT, IndsT}, ::Any) where {ComplexF64, N , StoreT <: MockTensor, IndsT} = NaN
Base.getindex(::NDTensors.Tensor{ComplexF64, N, StoreT, IndsT}, i...) where {ComplexF64, N , StoreT <: MockTensor, IndsT} = NaN
Base.show(io::IO, ::MIME"text/plain", a::MockTensor) = print(io, "MockTensor with dims $(Tuple(a.size))")


"""
    mock_contract(T1::NDTensors.Tensor,
                  labelsT1,
                  T2::NDTensors.Tensor,
                  labelsT2,
                  labelsR = NDTensors.contract_labels(labelsT1, labelsT2))

Overloaded contract function from NDTensors which implements
contraction for tensors using MockTensor objects as storage.
"""
function mock_contract(T1::NDTensors.Tensor,
                       labelsT1,
                       T2::NDTensors.Tensor,
                       labelsT2,
                       labelsR = NDTensors.contract_labels(labelsT1, labelsT2))
    
    final_dims = zeros(Int64, length(labelsR))
    T1_dims, T2_dims = size(T1), size(T2)
    final_inds = Array{ITensors.Index}(undef, length(labelsR))

    for (dims, labels, inds) in zip([T1_dims, T2_dims], [labelsT1, labelsT2], [T1.inds, T2.inds])
        for (i, li) in enumerate(labels)
            pos = findfirst(x -> x == li, labelsR)
            if pos !== nothing
                final_dims[pos] = dims[i]
                final_inds[pos] = inds[i]
            end
        end
    end
    tensor_store = MockTensor(final_dims)

    return NDTensors.Tensor(tensor_store, final_inds)
end