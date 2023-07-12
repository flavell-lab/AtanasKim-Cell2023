include("TotalVariation.jl")
using TotalVariation
using PyPlot


ground_truth = vcat(ones(1000),
-10sin.(linspace(0,pi,400)), 1ones(950))
noise        = 10randn(length(ground_truth))
combined     = ground_truth .+ noise
gstv_out     = TotalVariation.gstv(combined, 40, 15.0)
tv_out       = TotalVariation.tv(combined, 200.0)

figure(1)
PyPlot.clf();
subplot(2,2,1)
title("original signal")
plot(ground_truth)

subplot(2,2,2)
title("original + noise")
plot(combined)

subplot(2,2,3)
title("tv correction")
plot(tv_out)
plot(ground_truth, color="g")

subplot(2,2,4)
title("gstv correction")
plot(gstv_out)
plot(ground_truth, color="g")

rms(s) = sqrt(mean((ground_truth.-s).^2))

println("Initial Error        = ", rms(combined))
println("Output  Error [tv]   = ", rms(tv_out))
println("Output  Error [gstv] = ", rms(gstv_out))
