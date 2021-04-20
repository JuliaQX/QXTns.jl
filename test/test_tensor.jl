using LinearAlgebra

@testset "Test QXTensor constructors and basic usage" begin
    # create a QXTensor with no data
    a = QXTensor([Index(2)])
    @assert size(a) == (2,)

    # test hyper edge search
    data = Diagonal(ones(4))
    indices = [Index(4), Index(4)]
    a = QXTensor(data, indices)
    @test size(a) == (4, 4)
    @test hyperindices(a) == [indices]

    data = reshape(QXTns.Gates.cx(), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(data, indices)
    @test hyperindices(a) == [indices[[2, 4]]]

    data = reshape(Diagonal(ones(4)), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(data, indices)
    @test hyperindices(a) == [indices[[1, 3]], indices[[2, 4]]]
end

@testset "Test contraction of hyper indices" begin
    # we create two sets of indices and identify subsets of indices that are hyper indices
    # we then contract over these sets to ensure the results set has the expected hyper indices
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[as[1], as[3]]]
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[bs[1], bs[3]]]

    @test QXTns.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[1], bs[3]]]

    # now an example with two sets of hyper indices
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[as[1], as[3]], [as[2], as[4]]]
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[bs[1], bs[3]], [bs[2], bs[4]]]

    @test QXTns.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[1], bs[3]], [as[2], bs[4]]]

    # next an example where the first tensor will have a group of hyper indices remaining
    # but the second tensor won't
    as = [Index(2), Index(2), Index(2), Index(2), Index(5)]
    a_hyper_indices = [[as[1], as[2], as[3]]]
    bs = [as[1], Index(4), Index(5), as[4]]
    b_hyper_indices = [[bs[3], bs[4]]]

    @test QXTns.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[2], as[3]]]

    # next an example where the first tensor has two groups linked with group from second tensor
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[as[1], as[2]], [as[3], as[4]]]
    bs = [as[1], as[4]]
    b_hyper_indices = [[bs[1], bs[2]]]

    @test QXTns.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[2], as[3]]]
end

@testset "Test tensor_data when considering hyperedges" begin
    # test a diagonal
    data = Diagonal(ones(4))
    indices = [Index(4), Index(4)]
    a = QXTensor(data, indices)
    @test tensor_data(a, consider_hyperindices=true) == ones(4)

    # test a rank 4 tensor with 2 sets of hyper edges
    data = reshape(Diagonal(ones(4)), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(data, indices)
    @test tensor_data(a, consider_hyperindices=true) == ones(2, 2)

    # test a 2x2x2 tensor with single group of hyper edges include all ranks
    data = zeros(2, 2, 2)
    for i in 1:2 data[i, i, i] = 1 end
    a = QXTensor(data, [Index(2), Index(2), Index(2)])
    @test tensor_data(a, consider_hyperindices=true) == ones(2)
end
