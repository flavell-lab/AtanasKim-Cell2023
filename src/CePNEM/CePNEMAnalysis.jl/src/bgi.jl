"""
    get_CePNEM_fit_score(
        model_params::Vector{Float64}, neuron_trace::Vector{Float64}, 
        v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64}
    )

Compute the CePNEM fit score, which is a measurement of how well the model fit the data.
Sets the initial condition parameter `y0` to the first timepoint, which enables comparisons of models across time better.

# Arguments:
- `model_params::Vector{Float64}`: The model parameters to use.
- `neuron_trace::Vector{Float64}`: The neuron trace to fit.
- `v::Vector{Float64}`: The worm's velocity.
- `θh::Vector{Float64}`: The worm's head curvature.
- `P::Vector{Float64}`: The worm's pumping rate.
"""
function get_CePNEM_fit_score(model_params::Vector{Float64}, neuron_trace::Vector{Float64}, 
        v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64})
    cmap = Gen.choicemap()
    cmap[:c_vT] = model_params[1]
    cmap[:c_v] = model_params[2]
    cmap[:c_θh] = model_params[3]
    cmap[:c_P] = model_params[4]
    cmap[:y0] = neuron_trace[1] # better estimate of this parameter than the trace from a possibly wrong time range
    cmap[:s0] = model_params[7]
    cmap[:b] = model_params[8]
    cmap[:ℓ0] = model_params[9]
    cmap[:σ0_SE] = model_params[10]
    cmap[:σ0_noise] = model_params[11]
    cmap[:ys] = neuron_trace

    return Gen.get_score(Gen.generate(nl10d, (length(neuron_trace), v, θh, P), cmap)[1])
end

"""
    get_CePNEM_prior_score(neuron_trace::Vector{Float64}, v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64})

Sample a random set of parameters from the CePNEM prior and compute the model score on the observed data.
Sets the initial condition parameter `y0` to the first timepoint, which enables comparisons of models across time better.

# Arguments:
- `neuron_trace::Vector{Float64}`: The neuron trace to fit.
- `v::Vector{Float64}`: The worm's velocity.
- `θh::Vector{Float64}`: The worm's head curvature.
- `P::Vector{Float64}`: The worm's pumping rate.
"""
function get_CePNEM_prior_score(neuron_trace::Vector{Float64}, v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64})
    cmap = Gen.choicemap()
    cmap[:ys] = neuron_trace
    cmap[:y0] = neuron_trace[1] # better estimate of this parameter than the trace from a possibly wrong time range

    return Gen.get_score(Gen.generate(nl10d, (length(neuron_trace), v, θh, P), cmap)[1])
end

"""
    get_CePNEM_full_posterior_score(full_posterior, neuron_trace, v, θh, P)
    
Sample a random set of parameters from the CePNEM fitted posterior and compute the model score on the observed data.

# Arguments:
- `full_posterior::Vector{Vector{Float64}}`: The full posterior distribution of the model parameters.
- `neuron_trace::Vector{Float64}`: The neuron trace to fit.
- `v::Vector{Float64}`: The worm's velocity.
- `θh::Vector{Float64}`: The worm's head curvature.
- `P::Vector{Float64}`: The worm's pumping rate.
"""
function get_CePNEM_full_posterior_score(full_posterior, neuron_trace, v, θh, P)
    model_params = [rand(full_posterior[i]) for i = 1:11]
    return get_CePNEM_fit_score(model_params, neuron_trace, v, θh, P)
end


"""
    compute_BGI(test_score, prior_score)

Compute the Bayesian Generalization Index (BGI) of the model, using fractional overperformance relative to the prior as the metric.

# Arguments:
- `test_score::Vector{Float64}`: The model scores on the test data.
- `prior_score::Vector{Float64}`: The model scores on the prior data.
"""
function compute_BGI(test_score, prior_score)
    return 2 * prob_P_greater_Q(test_score, prior_score) - 1
end
