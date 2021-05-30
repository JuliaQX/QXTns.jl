using LinearAlgebra

@testset "Test QXTensor constructors and basic usage" begin
    # create a QXTensor with no data
    a = QXTensor([Index(2)])
    @assert size(a) == (2,)

    # test hyper edge search
    data = Diagonal(ones(4))
    indices = [Index(4), Index(4)]
    a = QXTensor(indices, data)
    @test size(a) == (4, 4)
    @test hyperindices(a) == [indices]

    data = reshape(QXTns.Gates.cx(), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(indices, data)
    @test hyperindices(a) == [indices[[2, 4]]]
    @test hyperindices(a, all_indices=true) == [[indices[1]], indices[[2, 4]], [indices[3]]]

    data = reshape(Diagonal(ones(4)), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(indices, data)
    @test hyperindices(a) == [indices[[1, 3]], indices[[2, 4]]]
end

@testset "Test contraction of tensors with hyper indices" begin
    # we create two sets of indices and identify subsets of indices that are hyper indices
    # we then contract over these sets to ensure the results set has the expected hyper indices
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[1, 3]]
    a = QXTensor(as, a_hyper_indices)
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[1, 3]]
    b = QXTensor(bs, b_hyper_indices)

    c = contract_tensors(a, b)
    @test Set(hyperindices(c)...) == Set([as[1], bs[3]])

    # now an example with two sets of hyper indices
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[1, 3], [2, 4]]
    a = QXTensor(as, a_hyper_indices)
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[1, 3], [2, 4]]
    b = QXTensor(bs, b_hyper_indices)

    c = contract_tensors(a, b)
    @test Set(hyperindices(c)[1]) == Set([as[1], bs[3]])
    @test Set(hyperindices(c)[2]) == Set([as[2], bs[4]])

    # next an example where the first tensor will have a group of hyper indices remaining
    # but the second tensor won't
    as = [Index(2), Index(2), Index(2), Index(5), Index(5)]
    a_hyper_indices = [[1, 2, 3]]
    a = QXTensor(as, a_hyper_indices)
    bs = [as[1], Index(4), Index(5), as[4]]
    b_hyper_indices = [[3, 4]]
    b = QXTensor(bs, b_hyper_indices)

    c = contract_tensors(a, b)
    @test Set(hyperindices(c)[1]) == Set([as[2], as[3]])

    # next an example where the first tensor has two groups linked with group from second tensor
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[1, 2], [3, 4]]
    a = QXTensor(as, a_hyper_indices)
    bs = [as[1], as[4]]
    b_hyper_indices = [[1, 2]]
    b = QXTensor(bs, b_hyper_indices)

    c = contract_tensors(a, b)
    @test Set(hyperindices(c)[1]) == Set([as[2], as[3]])
end

@testset "Test tensor_data when considering hyperedges" begin
    # test a diagonal
    data = Diagonal(ones(4))
    indices = [Index(4), Index(4)]
    a = QXTensor(indices, data)
    @test tensor_data(a, consider_hyperindices=true) == ones(4)

    # test a rank 4 tensor with 2 sets of hyper edges
    data = reshape(Diagonal(ones(4)), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(indices, data)
    @test tensor_data(a, consider_hyperindices=true) == ones(2, 2)

    # test a 2x2x2 tensor with single group of hyper edges include all ranks
    data = zeros(2, 2, 2)
    for i in 1:2 data[i, i, i] = 1 end
    a = QXTensor([Index(2), Index(2), Index(2)], data)
    @test tensor_data(a, consider_hyperindices=true) == ones(2)
end
