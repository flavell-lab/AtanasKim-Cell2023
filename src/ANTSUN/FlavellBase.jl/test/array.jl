using FlavellBase, Statistics, Test
@testset "Array manipulation" begin
    a_test = zeros(Float64, 10,5)
    for i = 1:5 a_test[:,i] .= i end

    # maxprj
    @test maxprj(a_test, dims=1) == Float64.(1:5)
    @test maxprj(a_test, dims=2) == fill(5., 10)

    # meanprj
    @test meanprj(a_test, dims=1) == Float64.(1:5)
    @test meanprj(a_test, dims=2) == fill(3., 10)

    # map_data
    @test map_data(sum, a_test, dims=1) == 10. * collect(1:5)
    @test map_data(sum, a_test, dims=2) == fill(15., 10)

    # standardize
    a_rand = rand(100,100,100)
    a_rand_stand = standardize(a_rand)
    μ, σ = mean(a_rand_stand), std(a_rand_stand)
    @test isapprox(μ, 0.0, atol=1e-10)
    @test isapprox(σ, 1.0, atol=1e-10)
end                                                                                                                                                                                    
