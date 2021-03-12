using LinearAlgebra

@testset "Test the tensor network utlities" begin
    # test single qubit gates
    h = [[1., 1.] [1., -1.]]
    @test QXTn.find_hyper_edges(h) == []
    z = [[1., 0.] [0., -1.]]
    @test QXTn.find_hyper_edges(z) == [[1, 2]]

    # test 2 qubit identity
    id = collect(Diagonal(ones(4)))
    @test QXTn.find_hyper_edges(id) == [[1, 2]]
    @test QXTn.find_hyper_edges(reshape(id, (2, 2, 2, 2))) == [[1, 3], [2, 4]]
    B, C = QXTn.decompose_gate(reshape(id, (2, 2, 2, 2)))
    @test size(B)[3] == 1
    @test size(C)[1] == 1
    @test QXTn.find_hyper_edges(B) == [[1, 2]]
    @test QXTn.find_hyper_edges(C) == [[2, 3]]

    # test 2 qubit cz
    id = collect(Diagonal(ones(4)))
    @test QXTn.find_hyper_edges(id) == [[1, 2]]
    @test QXTn.find_hyper_edges(reshape(id, (2, 2, 2, 2))) == [[1, 3], [2, 4]]
    B, C = QXTn.decompose_gate(reshape(id, (2, 2, 2, 2)))
    @test size(B)[3] == 1
    @test size(C)[1] == 1
    @test QXTn.find_hyper_edges(B) == [[1, 2]]
    @test QXTn.find_hyper_edges(C) == [[2, 3]]
end

@testset "Test reduce tensor function" begin
    # first a 2d diagonal tensor
    A = rand(5)
    Ar = QXTn.reduce_tensor(Diagonal(A), [[1, 2]])
    @test Ar == A

    # create a rank 5 tensor from a rank 3 tensor and reduce again
    A = rand(4, 5, 6)
    Afull = zeros(4, 5, 4, 5, 6)
    for i in CartesianIndices(size(A))
        Afull[i[1], i[2], i[1], i[2], i[3]] = A[i]
    end
    Ar = QXTn.reduce_tensor(Afull, [[1, 3], [2, 4]])
    @test Ar == A
end