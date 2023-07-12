using TotalVariation, Lasso

"""
    preview_denoise(f::Array{T,1}, denoiser) where T

Applies denoiser to f and plots the result

Arguments
---------
* `f`: 1D array to be denoised
* `denoiser`: denoiser function
"""
function preview_denoise(f::Array{T,1}, denoiser) where T
    plot(f, "k", alpha=0.75)
    plot(denoiser(f), "r")
    legend(["original", "denoised"])

    nothing
end

"""
    denoise(f::Array{T,2}, denoiser) where T

Applies denoiser to rows of f

Arguments
---------
* `f`: N by T where N: number of units, T: number of time points.
* `denoiser`: denoiser function
"""
function denoise(f::Array{T,2}, denoiser) where T
    f_denoised = zero(f)
    @showprogress for roi_ = 1:size(f, 1)
        f_denoised[roi_, :] = denoiser(f[roi_, :])
    end

    f_denoised
end


"""
    denoise!(data_dict::Dict, denoiser)

Applies denoiser to rows of `data_dict["f"]` and saves to
`data_dict["f_denoised"]`

Arguments
---------
* `data_dict`: data dictionary
* `denoiser`: denoiser function
"""
function denoise!(data_dict::Dict, denoiser)
    data_dict["f_denoised"] = denoise(data_dict["f"], denoiser)

    nothing
end

"""
Group spase total variation filter denoiser

Constructors
------------
* `DenoiserGSTV()`
* `DenoiserGSTV(n=50, λ=5.0)`
Arguments
---------
* `n`: number of elements in group
* `λ`: cost function parameter
"""
struct DenoiserGSTV
    n::Int
    λ::Float64
    f

    function DenoiserGSTV(n=50, λ=5.0)
        new(n, λ, x -> TotalVariation.gstv(x, n, λ))
    end
end

"""
Trendfilter denoiser

Constructors
------------
* `DenoiserTrendfilter()`
* `DenoiserTrendfilter(order=2, λ=2.0)`
Arguments
---------
* `order`: order
* `λ`: cost function parameter
"""
struct DenoiserTrendfilter
    order::Int
    λ::Float64
    f

    function DenoiserTrendfilter(order=2, λ=2.0)
        new(order, λ, x -> fit(TrendFilter, x, order, λ).β)
    end
end

Base.show(io::IO, denoiser::DenoiserTrendfilter) = print(io,
    "DenoiserTrendfilter(order=$(denoiser.order), λ=$(denoiser.λ))")

Base.show(io::IO, denoiser::DenoiserGSTV) = print(io,
    "DenoiserGSTV(n=$(denoiser.n), λ=$(denoiser.λ))")
