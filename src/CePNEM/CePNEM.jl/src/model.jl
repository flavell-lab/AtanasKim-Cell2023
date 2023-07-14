zstd = FlavellBase.standardize
logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x - x0)
lesser(x,x0) = leaky_logistic(x0,x,50,1e-3)

const s_MEAN = 10
const σ_MEAN = 0.25

# v_STD, vT_STD, θh_STD, P_STD are defined in FlavellConstants.jl

const ℓ_MEAN = 20
const ℓ_STD = 1
const α_MEAN = 1
const α_STD = 2
const σ_RQ_MEAN = 0.5
const σ_RQ_STD = 1.0
const σ_SE_MEAN = 0.5
const σ_SE_STD = 1.0
const σ_NOISE_MEAN = 0.125
const σ_NOISE_STD = 0.5

@gen (static) function kernel_noewma(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c::Float64), (grad)(b::Float64), σ::Float64) # latent variables
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(xs[t], v_0)) * (c_v * xs[t] + c) + b, σ)
    return y
end

@gen (static) function kernel_v(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c::Float64), (grad)(s::Float64), (grad)(b::Float64), (grad)(σ::Float64)) # latent variables
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(xs[t], v_0)) * (c_v * xs[t] + c) / (s+1) + (y_prev - b) * s / (s+1) + b, σ)
    return y
end

@gen (grad, static) function kernel_nl7b(t::Int, (grad)(y_prev::Float64), std_v::Array{Float64}, 
            std_θh::Array{Float64}, std_P::Array{Float64}, v_0::Float64,
            (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c_θh::Float64),
            (grad)(c_P::Float64), (grad)(c::Float64),
            (grad)(s::Float64), (grad)(b::Float64), (grad)(σ::Float64))
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(std_v[t], v_0)) * 
            (c_v * std_v[t] + c_θh * std_θh[t] + c_P * std_P[t] + c) / (s+1)
            + (y_prev - b) * s / (s+1) + b, σ)
    return y
end

function model_nl8(max_t::Int, c_vT::T, c_v::T, c_θh::T, c_P::T, c::Union{Float64,T}, y0::T, s0::T, b::T, v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64}) where T
    std_v = v ./ v_STD
    std_θh = θh ./ θh_STD
    std_P = P ./ P_STD
    
    activity = zeros(T, max_t)
    s = compute_s(s0)
    activity_prev = y0

    for t=1:max_t
        activity[t] = ((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * (std_v[t] < 0)) * 
                (c_v * std_v[t] + c_θh * std_θh[t] + c_P * std_P[t] + c) / (s+1) + (activity_prev - b) * s / (s+1) + b
        activity_prev = activity[t]
    end
    return activity
end

@gen (grad, static) function kernel_nl8((grad)(z::Float64), (grad)(σ::Float64))
    y ~ normal(z, σ)
    return y
end

Gen.@load_generated_functions

chain = Gen.Unfold(kernel_noewma)

chain_v = Gen.Unfold(kernel_v)

chain_nl7b = Gen.Unfold(kernel_nl7b)

chain_nl8 = Gen.Map(kernel_nl8)


@gen (static) function unfold_nl7b(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0
    std_v = v/v_STD
    std_θh = θh/θh_STD
    std_P = P/P_STD

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0 ~ normal(0,1)
    
    s = compute_s(s0)
    σ = compute_σ(σ0)

    chain ~ chain_nl7b(t, y0, std_v, std_θh, std_P, v_0, c_vT, c_v, c_θh, c_P, c, s, b, σ)
    return 1
end

@gen function nl8(max_t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0 ~ normal(0,1)

    σ = compute_σ(σ0)
    
    z = model_nl8(max_t, c_vT, c_v, c_θh, c_P, c, y0, s0, b, v[1:max_t], θh[1:max_t], P[1:max_t])
    chain ~ chain_nl8(z, fill(σ, max_t))
    
    return 1
end

@gen (static) function nl9(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0
    std_v = v/v_STD
    std_θh = θh/θh_STD
    std_P = P/P_STD

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0_model ~ normal(0,1)
    σ0_measure ~ normal(0,1)
    
    s = compute_s(s0)
    σ_model = compute_σ(σ0_model)
    σ_measure = compute_σ(σ0_measure)

    chain_model ~ chain_nl7b(t, y0, std_v, std_θh, std_P, v_0, c_vT, c_v, c_θh, c_P, c, s, b, σ_model)
    chain ~ chain_nl8(chain_model, fill(σ_measure, t))
    return 1
end

function compute_cov_matrix_vectorized_RQ(max_t, α, ℓ, σ_RQ, σ_noise)
    ts = collect(1:max_t)
    Δt = ts .- ts'
    σ_RQ^2 .* (1 .+ Δt .* Δt ./ (2 * α * ℓ^2)) .^ (-α) + Matrix(σ_noise^2 * LinearAlgebra.I, max_t, max_t)
end

function compute_cov_matrix_vectorized_SE(max_t, ℓ, σ_SE, σ_noise)
    ts = collect(1:max_t)
    Δt = ts .- ts'
    σ_SE^2 .* exp.(-0.5 .* Δt .* Δt ./ (ℓ^2)) + Matrix(σ_noise^2 * LinearAlgebra.I, max_t, max_t)
end

@gen function nl10(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    
    ℓ0 ~ normal(0,1)
    α0 ~ normal(0,1)
    σ0_RQ ~ normal(0,1)
    σ0_noise ~ normal(0,1)
    
    ℓ = ℓ_MEAN * exp(ℓ0 * ℓ_STD)
    α = α_MEAN * exp(α0 * α_STD)
    σ_RQ = σ_RQ_MEAN * exp(σ0_RQ * σ_RQ_STD)
    σ_noise = σ_NOISE_MEAN * exp(σ0_noise * σ_NOISE_STD)

    cov_matrix = compute_cov_matrix_vectorized_RQ(t, α, ℓ, σ_RQ, σ_noise)
    z = model_nl8(t, c_vT, c_v, c_θh, c_P, c, y0, s0, b, v, θh, P)
    
    @trace(mvnormal(z, cov_matrix), :ys)
    return 1
end

@gen function nl10c(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    
    ℓ0 ~ normal(0,1)
    σ0_SE ~ normal(0,1)
    σ0_noise ~ normal(0,1)
    
    ℓ = ℓ_MEAN * exp(ℓ0 * ℓ_STD)
    σ_SE = σ_SE_MEAN * exp(σ0_SE * σ_SE_STD)
    σ_noise = σ_NOISE_MEAN * exp(σ0_noise * σ_NOISE_STD)

    cov_matrix = compute_cov_matrix_vectorized_SE(t, ℓ, σ_SE, σ_noise)
    z = model_nl8(t, c_vT, c_v, c_θh, c_P, c, y0, s0, b, v, θh, P)
    
    @trace(mvnormal(z, cov_matrix), :ys)
    return 1
end

"""
    function nl10d(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})

This function is a probabilistic model that generates a time series of neural data `ys` given the input data `v`, `θh`, and `P`.

# Arguments
- `t::Int`: The number of time points in the time series.
- `v::Array{Float64}`: An array of length `t` representing the worm's velocity.
- `θh::Array{Float64}`: An array of length `t` representing the worm's head curvature.
- `P::Array{Float64}`: An array of length `t` representing the worm's pumping (feeding).

# Returns
- `1`: A constant value indicating that the function has finished executing.

# Internal Variables
- `c_vT`: A scalar variable representing the locomotion direction rectification parameter.
- `c_v`: A scalar variable representing the velocity coefficient.
- `c_θh`: A scalar variable representing the head curvature coefficient.
- `c_P`: A scalar variable representing the pumping coefficient.
- `c`: A scalar variable representing the constant rectification term. This is disabled (set to 0) in `NL10d`.
- `y0`: A scalar variable representing the initial observation.
- `s0`: A scalar variable representing the timescale parameter of the neuron's encoding to behavior.
- `b`: A scalar variable representing the bias term.
- `ℓ0`: A scalar variable representing the timescale of the residual Gaussian process model.
- `σ0_SE`: A scalar variable representing the standard deviation of the squared exponential kernel in the residual Gaussian process model.
- `σ0_noise`: A scalar variable representing the standard deviation of the observation noise (modeled as white noise).
"""
@gen function nl10d(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c = 0.
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    
    ℓ0 ~ normal(0,1)
    σ0_SE ~ normal(0,1)
    σ0_noise ~ normal(0,1)
    
    ℓ = ℓ_MEAN * exp(ℓ0 * ℓ_STD)
    σ_SE = σ_SE_MEAN * exp(σ0_SE * σ_SE_STD)
    σ_noise = σ_NOISE_MEAN * exp(σ0_noise * σ_NOISE_STD)

    cov_matrix = compute_cov_matrix_vectorized_SE(t, ℓ, σ_SE, σ_noise)
    z = model_nl8(t, c_vT, c_v, c_θh, c_P, c, y0, s0, b, v, θh, P)
    
    @trace(mvnormal(z, cov_matrix), :ys)
    return 1
end

@gen (static) function unfold_v_noewma(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c_vT ~ uniform(-pi/2, pi/2)
    c_v ~ normal(0,1)
    c ~ normal(0,1)
    b ~ normal(0,2)
    σ ~ exponential(1.0)

    chain ~ chain(t, 0.0, std_v, v_0, c_vT, c_v, c, b, σ)
    return 1
end

@gen (static) function unfold_v(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,2)
    σ0 ~ normal(0,1)
    
    s = compute_s(s0)
    σ = compute_σ(σ0)

    chain ~ chain_v(t, y0, std_v, v_0, c_vT, c_v, c, s, b, σ)
    return 1
end

function compute_s(s0)
    return s_MEAN * exp(s0)
end

function compute_σ(σ0)
    return σ_MEAN * exp(σ0/2)
end

Gen.@load_generated_functions

"""
    get_free_params(trace, model)

Extracts the free parameters from a trace of a generative model.

Arguments:
- `trace`: A trace of a generative model.
- `model`: A symbol representing the name of the generative model.

Returns:
- An array of the free parameters of the generative model.

The `get_free_params` method takes a trace of a generative model and a symbol representing the name of the generative model as arguments.
It then extracts the free parameters from the trace and returns them as an array.
The number and names of the free parameters depend on the generative model, but for model NL10d there are 11.
"""
function get_free_params(trace, model)
    if model == :nl10d
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], 0., trace[:y0], trace[:s0], trace[:b], trace[:ℓ0], trace[:σ0_SE], trace[:σ0_noise]]
    elseif model == :nl10c
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:ℓ0], trace[:σ0_SE], trace[:σ0_noise]]
    elseif model == :nl10
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:α0], trace[:ℓ0], trace[:σ0_RQ], trace[:σ0_noise]]
    elseif model == :nl9
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0_model], trace[:σ0_measure]]
    elseif model in [:nl7b, :nl8]
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0]]
    elseif model == :v
        return [trace[:c_vT], trace[:c_v], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0]]
    elseif model == :v_noewma
        return [trace[:c_vT], trace[:c_v], trace[:c], trace[:b], trace[:σ]]
    end
end
