using Test
using Random
using Statistics
using TotalVariation

#Initial signals
Random.seed!(0)
g_truth = [ones(100); 5*ones(200); -10*ones(100)]
noise   = randn(size(g_truth))
mixed   = g_truth .+ noise


#Denoising operation
denoised      = tv(mixed, 10.0)

#Results
noise_before  = mean((mixed   .-g_truth).^2)
noise_after   = mean((denoised.-g_truth).^2)

println("Noise before = ", noise_before)
println("Noise after  = ", noise_after)

@test noise_before > noise_after
