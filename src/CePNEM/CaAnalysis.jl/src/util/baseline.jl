function estimate_baseline(f::Array{T,1}; order=2, λ=1000, n_kernel=100) where T
    n_f = length(f)

    denoiser = DenoiserTrendfilter(order,λ)

    f_denoised = denoiser.f(f)
    f_baseline = zeros(n_f)

    kernel = centered(ones(n_kernel) ./ n_kernel)

    for i = 1:n_f
        start = max(1, i - n_kernel)
        stop = min(n_f, i + n_kernel)
        f_baseline[i] = minimum(f_denoised[start:stop])
    end

    f_baseline[f_baseline .< 1] .= 1.0
    f_baseline = imfilter(f_baseline, kernel, "replicate")

    f_baseline
end

function estimate_baseline(f::Array{T,2}; order=2, λ=1000, n_kernel=100) where T

    f_baseline = zeros(eltype(f), size(f))

    @showprogress for roi_ = 1:size(f, 1)
        f_baseline[roi_, :] = estimate_baseline(f[roi_,:], order=order,
            λ=λ, n_kernel=n_kernel )
    end

    f_baseline
end
