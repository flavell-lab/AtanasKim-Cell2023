"""
    optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::HBParams; idx_scaling::Vector{Int64}=[2,3,4])

Optimize the maximum a posteriori (MAP) estimate for the given parameter samples `Ps` and initial parameters `params_init`.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices containing the CePNEM parameter samples for each dataset.
- `params_init::HBParams`: A HBParams instance containing the initial values for mu, sigma, and x parameters.
- `idx_scaling::Vector{Int64}`: An optional vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
    Currently, it is not supported for this to be any value other than its default `[2,3,4]`.
- `max_iters::Int64`: An optional integer indicating the maximum number of iterations for the optimization process. Default 200.
- `mvns::Union{Vector, Nothing}`: An optional vector of multivariate normal distributions corresponding to the data. If `nothing`, they will be computed.

# Returns
- `params_opt_struct`: A HBParams instance containing the optimized values for mu, sigma, and x parameters.
- `result`: The result of the optimization process.
"""
function optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::HBParams; idx_scaling::Vector{Int64}=[2,3,4], max_iters::Int64=200, mvns::Union{Vector, Nothing}=nothing)
    # Initialize the multivariate normal approximations
    if isnothing(mvns)
        mvns = fit_multivariate_normals(Ps) 
    end

    # Flatten the initial parameters
    mu_init = params_init.mu
    sigma_init = params_init.sigma
    x_init_flat = vcat([params_init.x[i] for i in 1:length(Ps)]...)

    params_init_flat = vcat(mu_init, sigma_init, x_init_flat)

    # Compute the gradient
    joint_logprob_grad = params -> ForwardDiff.gradient(p -> joint_logprob_flat_negated(p, Ps, mvns, idx_scaling), params)

    # Compute the Hessian
    joint_logprob_hessian = params -> ForwardDiff.hessian(p -> joint_logprob_flat_negated(p, Ps, mvns, idx_scaling), params)

    function g!(G, x)
        if G !== nothing
            G .= joint_logprob_grad(x)
        end
    end

    function h!(H, x)
        H .= joint_logprob_hessian(x)
    end

    # Perform Newton optimization
    result = optimize(x->joint_logprob_flat_negated(x, Ps, mvns, idx_scaling), g!, h!, params_init_flat, Optim.Newton(), Optim.Options(iterations=max_iters, store_trace=true))

    # Unpack the optimized parameters
    params_opt = Optim.minimizer(result)
    mu_opt = params_opt[1:size(Ps[1],2)]
    sigma_opt = params_opt[size(Ps[1],2) + 1:2 * size(Ps[1],2)]
    x_opt_flat = params_opt[2 * size(Ps[1],2) + 1:end]
    x_opt = [x_opt_flat[(i - 1) * size(Ps[1],2) + 1 : i * size(Ps[1],2)] for i in 1:length(Ps)]

    # Create the optimized HBParams struct
    params_opt_struct = HBParams(mu_opt, sigma_opt, x_opt)

    return params_opt_struct, result
end

"""
    initialize_params(Ps::Vector{Matrix{Float64}}; idx_scaling::Vector{Int64}=[2,3,4])

Initialize the model parameters based on the given parameter samples `Ps`.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices containing the CePNEM parameter samples for each dataset.
- `idx_scaling::Vector{Int64}`: An optional vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
    Currently, it is not supported for this to be any value other than its default `[2,3,4]`.

# Returns
- `HBParams`: A HBParams instance containing the initialized values for mu, sigma, and x parameters.
"""
function initialize_params(Ps::Vector{Matrix{Float64}}; idx_scaling::Vector{Int64}=[2,3,4])
    means = [mean(P, dims=1)[1, :] for P in Ps]
    mu_init_cart = mean(means, dims=1)[1]
    mu_init = params_to_spher(mu_init_cart)

    x_init = means

    for i in 1:length(Ps)
        x_init[i] = params_to_spher(x_init[i])
    end

    sigma_init = [log(std([x_init[i][j] for i=1:length(Ps)])) for j=1:2]
    data_spher = [mu[2:4] ./ norm(mu[2:4]) for mu in means]
    append!(sigma_init, log(estimate_kappa(data_spher, mean_direction(data_spher)[1])))
    append!(sigma_init, 0.0)
    append!(sigma_init, log(std([x_init[i][5] for i=1:length(Ps)])))

    return HBParams(mu_init, sigma_init, x_init)
end