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

    data = reshape(QXTn.Gates.cx(), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(data, indices)
    @test hyperindices(a) == [indices[[2, 4]]]

    data = reshape(Diagonal(ones(4)), (2, 2, 2 ,2))
    indices = [Index(2), Index(2), Index(2), Index(2)]
    a = QXTensor(data, indices)
    @test hyperindices(a) == [indices[[1, 3]], indices[[2, 4]]]
end

@testset "Test contraction of hyper indices" begin
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[as[1], as[3]]]
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[bs[1], bs[3]]]

    @test QXTn.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[1], bs[3]]]

    # now an example with two sets of hyper indices
    as = [Index(2), Index(2), Index(2), Index(2)]
    a_hyper_indices = [[as[1], as[3]], [as[2], as[4]]]
    bs = [as[3], as[4], Index(2), Index(2)]
    b_hyper_indices = [[bs[1], bs[3]], [bs[2], bs[4]]]

    @test QXTn.contract_hyper_indices(as, a_hyper_indices, bs, b_hyper_indices) == [[as[1], bs[3]], [as[2], bs[4]]]
end
