"""
    struct HBParams

A structure for holding the mean (mu), standard deviation (sigma), and the spherical coordinates (x) of a model.

# Fields
- `mu::Vector{Float64}`: A vector of the global mean estimate for each parameter.
- `sigma::Vector{Float64}`: A vector of the global standard deviation estimate for each parameter.
- `x::Vector{Vector{Float64}}`: A vector of vectors representing the best parameters for each individual dataset.
"""
struct HBParams
    mu::Vector{Float64}
    sigma::Vector{Float64}
    x::Vector{Vector{Float64}}
end

"""
    angular_log_probability(theta, phi)

Compute the log-probability of a given (θ, φ) pair under the uniform prior over the surface
of a sphere.

# Arguments
- `theta::Real`: Polar angle θ ∈ [0, π]
- `phi::Real`: Azimuthal angle φ ∈ [0, 2π]

# Returns
- `log_prob::Real`: The natural logarithm of the joint probability density function of the
  input angles (θ, φ) under the uniform prior.
"""
function angular_log_probability(theta, phi)
    joint_pdf = abs(sin(theta)) / (2 * pi^2)
    log_prob = log(joint_pdf + 1e-8)
    return log_prob
end

"""
    joint_logprob_flat(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})

Compute the joint log probability of a flat parameter vector given the data, multivariate normal distributions (mvns), and scaling indices (idx_scaling).

# Arguments
- `params_flat::Vector`: A flat vector containing the concatenated mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: A vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
    Currently, it is not supported for this to be any value other than `[2,3,4]`.

# Returns
- `logprob`: The computed joint log probability of the given parameters.
"""
function joint_logprob_flat(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    @assert(idx_scaling==[2,3,4], "Currently, spherical coordinate transform must consist of parameters 2, 3, and 4.")
    n_params = size(data[1], 2)
    mu = params_flat[1:n_params]
    sigma = params_flat[n_params + 1:2 * n_params]
    x_flat = params_flat[2 * n_params + 1:end]
    x_spher = [x_flat[(i - 1) * n_params + 1 : i * n_params] for i in 1:length(data)]

    logprob = 0.0

    # Add log probability of higher-level parameters
    logprob += Distributions.logpdf(Normal(0.0,0.3), mu[1])  # Prior for c_vT
    logprob += Distributions.logpdf(Normal(0.1,0.4), mu[2]) # Prior for log(r)
    logprob += angular_log_probability(mu[3], mu[4]) # prior for theta and phi is uniform on the sphere
    logprob += Distributions.logpdf(Normal(0.7,0.7), mu[5])  # Prior for lambda

    logprob += sum(Distributions.logpdf.(Normal(-1.0,1.0), sigma[1]))  # Prior for log(sigma_cvT)
    logprob += Distributions.logpdf(Normal(-1.0,1.0), sigma[2])  # Prior for log(sigma_logr)
    logprob += Distributions.logpdf(Normal(1.0, 1.0), sigma[3])  # Prior for log(kappa)
    # sigma[4] is 0, a placeholder
    logprob += sum(Distributions.logpdf.(Normal(-1.0,1.0), sigma[5]))  # Prior for log(sigma_lambda)

    exp_sigma = exp.(sigma) .+ 2e-4

    # Add log probability of lower-level parameters and data
    for i in 1:length(data)
        x_i_spher = x_spher[i]

        x_i_cart = hbparams_to_cart(x_i_spher)

        P_i = data[i]
        for j in 1:size(P_i,2)
            if j == idx_scaling[2]
                mu_spherical = spher2cart([1.0, mu[3:4]...])
                vmf = VonMisesFisher(mu_spherical, exp_sigma[j])
                x_i_cart_j = spher2cart([1.0, x_i_spher[3:4]...])
                logprob += Distributions.logpdf(vmf, x_i_cart_j)
            elseif j != idx_scaling[3]
                logprob += Distributions.logpdf(Normal(mu[j], exp_sigma[j]), x_i_spher[j])
            end
        end
        logprob += Distributions.logpdf(mvns[i], x_i_cart)
    end

    return logprob
end


"""
    joint_logprob(params::HBParams, data::Vector{Matrix{Float64}}, mvns::Vector; idx_scaling::Vector{Int64}=[2,3,4])

Compute the joint log probability of a HBParams instance given the data and multivariate normal distributions (mvns).

# Arguments
- `params::HBParams`: A HBParams instance containing the mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: An optional vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
Currently, it is not supported for this to be any value other than its default, `[2,3,4]`.

# Returns
- `logprob`: The computed joint log probability of the given parameters.
"""
function joint_logprob(params::HBParams, data::Vector{Matrix{Float64}}, mvns::Vector; idx_scaling::Vector{Int64}=[2,3,4])
    mu = params.mu
    sigma = params.sigma
    x_spher = params.x

    params_flat = [mu; sigma; vcat(x_spher...)]

    return joint_logprob_flat(params_flat, data, mvns, idx_scaling)
end

"""
    joint_logprob_flat_negated(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})

Compute the negated joint log probability of a flat parameter vector given the data, multivariate normal distributions (mvns), and scaling indices (idx_scaling).

# Arguments
- `params_flat::Vector`: A flat vector containing the concatenated mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: A vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
Currently, it is not supported for this to be any value other than `[2,3,4]`.

# Returns
- `logprob`: The computed negated joint log probability of the given parameters.
"""
function joint_logprob_flat_negated(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    return -joint_logprob_flat(params_flat, data, mvns, idx_scaling)
end
