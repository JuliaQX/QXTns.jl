"""
In this module we define a MockTensor which inherits from the NDTensors.TensorStorage.
This can be used as the store for ITensor objects, but only stores the dimensions and so
can be used for tracking the dimension of contractions which would not be possible if
storing all the data.
"""

using ITensors

export MockTensor


""" Tensor store struct that just tracks tensor dimensions"""
struct MockTensor{T,N} <: AbstractArray{T, N}
    size::NTuple{N, Int}
end

MockTensor(size::NTuple{N, Int64}) where N = MockTensor{ComplexF64, N}(size)

"""Overload functions from base to make MockTensor usable"""
Base.copy(a::MockTensor{T, N}) where {T, N} = MockTensor{T, N}(a.size)
Base.length(a::MockTensor) = prod(size(a))
Base.size(a::MockTensor) = a.size
Base.getindex(::MockTensor, ::Int64) = NaN
Base.getindex(::MockTensor, i...) = NaN
Base.show(io::IO, ::MIME"text/plain", a::MockTensor) = print(io, "MockTensor with dims $(Tuple(a.size))")