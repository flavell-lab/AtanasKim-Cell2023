const BURNIN = 1000

"""
In case of c_v near 0, propose massive jump in c_vT and shift all other variables accordingly
"""
@gen function jump_c_vT(current_trace, neg)
    vT_coeff = -2 * current_trace[:c_vT] * current_trace[:c] / sqrt(current_trace[:c_vT]^2 + 1)
    c_coeff = (1+current_trace[:c_vT]) * current_trace[:c] / sqrt(current_trace[:c_vT]^2+1) + current_trace[:b]
    c_vT ~ normal(current_trace[:c_vT] * neg,1)
    c ~ normal(vT_coeff/(-2*c_vT/sqrt(c_vT^2+1)), 1e-4)
    b ~ normal(c_coeff - (1+c_vT)*c/sqrt(c_vT^2+1), 1e-4)
end

@gen function jump_c_vT_10d(current_trace, neg)
    c_vT ~ normal(current_trace[:c_vT] * neg, 1)
end

"""
In case of c_vT near 0, propose massive jump in c to the prior and shift all other variables accordingly
"""
@gen function jump_c(current_trace, neg)
    c_coeff = (1+current_trace[:c_vT]) * current_trace[:c] / sqrt(current_trace[:c_vT]^2+1) + current_trace[:b]
    c ~ normal(current_trace[:c] * neg, 1)
    b ~ normal(c_coeff - (1+current_trace[:c_vT])*c/sqrt(current_trace[:c_vT]^2+1), 1e-4)
end

"""
Use the fact that vT = (v < 0) and v are similar to propose effectively swapping the two variables
Specifically, moves parameter values along the manifold where v = (μ_vT - vT) / σ_vT. 
"""
@gen function jump_c_vvT(current_trace, neg_c_vT, neg_c_v, μ_vT, σ_vT, use_c)
    curr_c_vT = current_trace[:c_vT]
    curr_c_v = current_trace[:c_v]
    curr_c = use_c ? current_trace[:c] : 0.
    vT_coeff = (-2*curr_c_vT*curr_c*σ_vT + 2*curr_c_vT*curr_c_v - (1+curr_c_vT)*curr_c_v - 2*curr_c_vT*curr_c_v*μ_vT) / (sqrt(curr_c_vT^2+1) * σ_vT)
    c_coeff = current_trace[:b] + ((1+curr_c_vT)*curr_c) / sqrt(curr_c_vT^2+1) + ((1+curr_c_vT)*curr_c_v*μ_vT)/(sqrt(curr_c_vT^2+1)*σ_vT)
    
    c_vT ~ normal(current_trace[:c_vT] * neg_c_vT, 1)
    c_v ~ normal(current_trace[:c_v] * neg_c_v, 1)
    if use_c
        c ~ normal((-c_v + c_vT*c_v - 2*c_vT*c_v*μ_vT - sqrt(c_vT^2+1)*σ_vT*vT_coeff)/(2*c_vT*σ_vT), 1e-4)
    else
        c = 0.
    end
    b ~ normal(c_coeff - ((1+c_vT)*c+(1+c_vT)*c_v*μ_vT/σ_vT)/sqrt(c_vT^2+1), 1e-4)
end

function hmc_jump_update_noewma(tr, μ_vT, σ_vT)
    # apply HMC to the entire trace
    (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c, :b), eps=tr[:σ]/20)
    
    # apply "jump" transforms that attempt to exploit symmetries of the kernel
    neg = rand([-1, 1])
    (tr, accept) = mh(tr, jump_c_vT, (neg,))   
    neg = rand([-1, 1])
    (tr, accept) = mh(tr, jump_c, (neg,))   
    neg_c_vT = rand([-1,1])
    neg_c_v = rand([-1,1])
    (tr, accept) = mh(tr, jump_all, (neg_c_vT, neg_c_v, μ_vT, σ_vT))
    
    # update noise value
    (tr, accept) = mh(tr, select(:σ))
    
    return tr
end

@gen function drift_y0(current_trace)
    y0 ~ normal(current_trace[:y0], 0.5)
end

@gen function drift_chain_model(current_trace, t, σ)
    {(:chain_model => t => :y)} ~ normal(current_trace[:chain_model => t => :y], σ)
end

@gen function drift_params(tr, max_t, δ)
    α0 ~ normal(tr[:α0], δ*α_STD)
    ℓ0 ~ normal(tr[:ℓ0], δ*ℓ_STD)
    σ0_RQ ~ normal(tr[:σ0_RQ], δ*σ_RQ_STD)
    σ0_noise ~ normal(tr[:σ0_noise], δ*σ_NOISE_STD)
end

@gen function drift_α(tr, max_t, δ)
    α0 ~ normal(tr[:α0], δ*α_STD)
end

@gen function drift_ℓ(tr, max_t, δ)
    ℓ0 ~ normal(tr[:ℓ0], δ*ℓ_STD)
end

@gen function drift_σ_RQ(tr, max_t, δ)
    σ0_RQ ~ normal(tr[:σ0_RQ], δ*σ_RQ_STD)
end

@gen function drift_σ_SE(tr, max_t, δ)
    σ0_SE ~ normal(tr[:σ0_SE], δ*σ_SE_STD)
end

@gen function drift_σ_noise(tr, max_t, δ)
    σ0_noise ~ normal(tr[:σ0_noise], δ*σ_NOISE_STD)
end


function hmc_jump_update(tr, μ_vT, σ_vT, model; max_t=nothing)
    # update y0
    if model == :nl8 || model == :nl9
        (tr, accept) = mh(tr, select(:y0))
    elseif !(model == :v_noewma)
        (tr, accept) = mh(tr, drift_y0, ())
    end
    
    # update model error
    if model == :nl9
        σ = min(compute_σ(tr[:σ0_model]), compute_σ(tr[:σ0_measure]))
        for t=1:max_t
            (tr, accept) = mh(tr, drift_chain_model, (t,σ))
        end
    end

    # apply HMC to all other parameters
    if model == :nl9
        (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c_θh, :c_P, :c, :b, :y0, :s0, :σ0_model, :σ0_measure),
                eps=(compute_σ(tr[:σ0_model]) + compute_σ(tr[:σ0_measure]))/90)
    elseif model == :nl8
        (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c_θh, :c_P, :c, :b, :y0, :s0, :σ0), eps=compute_σ(tr[:σ0])/50)
    elseif model == :nl7b
        (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c_θh, :c_P, :c, :b, :s0, :σ0), eps=compute_σ(tr[:σ0])/30)
    elseif model == :v
        (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c, :b, :s0, :σ0), eps=compute_σ(tr[:σ0])/20)
    elseif model == :v_noewma
        (tr, accept) = hmc(tr, select(:c_vT, :c_v, :c, :b), eps=tr[:σ]/20)
    end
    
    # jump noise and EWMA parameters
    if !(model == :v_noewma)
        if model == :nl9
            (tr, accept) = mh(tr, select(:σ0_model))
            (tr, accept) = mh(tr, select(:σ0_measure))
        else
            (tr, accept) = mh(tr, select(:σ0))
        end
        (tr, accept) = mh(tr, select(:s0))
    else
        (tr, accept) = mh(tr, select(:σ))
    end
    
    # apply "jump" transforms that attempt to exploit symmetries of the kernel
    neg = rand([-1,1])
    (tr, accept) = mh(tr, jump_c_vT, (neg,))
    neg = rand([-1,1])
    (tr, accept) = mh(tr, jump_c, (neg,))
    neg_c_vT = rand([-1,1])
    neg_c_v = rand([-1,1])
    (tr, accept) = mh(tr, jump_c_vvT, (neg_c_vT, neg_c_v, μ_vT, σ_vT))
    
    # jump other parameters that haven't been jumped yet
    if model in [:nl7b, :nl8, :nl9]
        (tr, accept) = mh(tr, select(:c_θh))
        (tr, accept) = mh(tr, select(:c_P))
    end
    return tr
end

function run_mcmc_10(ys, v, θh, P; n_init=100000, n_iters=11000, lr_adjust=1.1, model=:nl10d)
    @assert(model in [:nl10c, :nl10d])
    n_params = (model == :nl10c) ? 11 : 10
    μ_vT = 0.0
    σ_vT = vT_STD
    max_t = length(ys)
    n_obs = max_t

    traces_fit = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_iters+1)
    accept = zeros(Bool,n_iters+1,7)
    δ_vals = fill(1.0,n_iters+1,4)

    traces_init = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_init)
    scores_init = zeros(n_init)
    cmap = Gen.choicemap()
    cmap[:ys] = ys

    println("Initializing MCMC chain...")
    flush(stdout)
    @time for i=1:n_init
        traces_init[i], _ = (model == :nl10c) ? generate(nl10c, (max_t,v,θh,P), cmap) : generate(nl10d, (max_t,v,θh,P), cmap)
        scores_init[i] = get_score(traces_init[i])
    end
    flush(stdout)
                    
    traces_sorted = traces_init[sort(1:n_init, by=x->scores_init[x])]
    traces_fit[1] = traces_sorted[end]

    println("Iterating MCMC chain...")
    flush(stdout)
    @time for i=1:n_iters
        traces_fit[i+1], accept[i+1,1] = mh(traces_fit[i], drift_ℓ, (max_t, δ_vals[i,1]))
        traces_fit[i+1], accept[i+1,2] = mh(traces_fit[i+1], drift_σ_SE, (max_t, δ_vals[i,2]))
        traces_fit[i+1], accept[i+1,3] = mh(traces_fit[i+1], drift_σ_noise, (max_t, δ_vals[i,3]))
        traces_fit[i+1], accept[i+1,4] = (model == :nl10c) ? hmc(traces_fit[i+1], select(:c_vT, :c_v, :c_θh, :c_P, :c, :b, :y0, :s0), eps=δ_vals[i,4]) :
                hmc(traces_fit[i+1], select(:c_vT, :c_v, :c_θh, :c_P, :b, :y0, :s0), eps=δ_vals[i,4])
        for k=1:4
            δ_vals[i+1,k] = (accept[i+1,k]) ? δ_vals[i,k] * lr_adjust : δ_vals[i,k] / lr_adjust
        end  
        neg = rand([-1,1])
        (traces_fit[i+1], accept[i+1,5]) = (model == :nl10c) ? mh(traces_fit[i+1], jump_c_vT, (neg,)) : mh(traces_fit[i+1], jump_c_vT_10d, (neg,))
        neg = rand([-1,1])
        if model == :nl10c
            (traces_fit[i+1], accept[i+1,6]) = mh(traces_fit[i+1], jump_c, (neg,))
        end
        neg_c_vT = rand([-1,1])
        neg_c_v = rand([-1,1])
        (traces_fit[i+1], accept[i+1,7]) = mh(traces_fit[i+1], jump_c_vvT, (neg_c_vT, neg_c_v, μ_vT, σ_vT, (model == :nl10c)))
    end
    flush(stdout)
    return traces_fit, accept, δ_vals
end

nl10c_traces_to_params = traces_fit -> mapreduce(permutedims, vcat, get_free_params.(traces_fit[BURNIN+1:end], :nl10c))
nl10d_traces_to_params = traces_fit -> mapreduce(permutedims, vcat, get_free_params.(traces_fit[BURNIN+1:end], :nl10d))

function particle_filter_incremental(num_particles::Int, v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64},
         ys::Vector{Float64}, num_steps::Int, model::Symbol; always_rejuvenate=false)
    μ_vT = 0.0
    σ_vT = vT_STD
    init_obs = Gen.choicemap((:chain => 1 => :y, ys[1]))
    if model == :nl9
        state = Gen.initialize_particle_filter(nl9, (1,v,θh,P), init_obs, num_particles)
    elseif model == :nl7b
        state = Gen.initialize_particle_filter(unfold_nl7b, (1,v,θh,P), init_obs, num_particles)
    elseif model == :nl8
        state = Gen.initialize_particle_filter(nl8, (1,v,θh,P), init_obs, num_particles)
    elseif model == :v
        state = Gen.initialize_particle_filter(unfold_v, (1,v), init_obs, num_particles)
    elseif model == :v_noewma
        state = Gen.initialize_particle_filter(unfold_v_noewma, (1,v), init_obs, num_particles)
    end
    for t=2:length(ys)
        if maybe_resample!(state, ess_threshold=num_particles/2) || always_rejuvenate
            for i=1:num_particles
                for step=1:num_steps
                    state.traces[i] = hmc_jump_update(state.traces[i], μ_vT, σ_vT, model, max_t=t-1)
                end
            end
        end
        obs = Gen.choicemap((:chain => t => :y, ys[t]))
        if model in [:nl7b, :nl8, :nl9]
            Gen.particle_filter_step!(state, (t,v,θh,P), (IntDiff(1), NoChange(), NoChange(), NoChange()), obs)
        elseif model == :v || model == :v_noewma
            Gen.particle_filter_step!(state, (t,v), (IntDiff(1), NoChange()), obs)
        end
    end
    return state
end

function output_state(state::Gen.ParticleFilterState, h5path::String, n_samples::Int, model::Symbol)
    if model == :nl9
        n_params = 10
    elseif model in [:nl7b, :nl8]
        n_params = 9
    elseif model == :v
        n_params = 7
    elseif model == :v_noewma
        n_params = 5
    end
    traces = Gen.get_traces(state)
    unweighted_traces = Gen.sample_unweighted_traces(state, n_samples)
    
    n_particles = length(traces)
    trace_params = zeros(n_particles, n_params)
    sampled_trace_params = zeros(n_samples, n_params)
    trace_scores = zeros(n_particles)
    
    for (i,tr) = enumerate(traces)
        trace_params[i,:] .= get_free_params(tr, model)
        trace_scores[i] = Gen.get_score(tr)
    end
    
    for (i,tr) = enumerate(unweighted_traces)
        sampled_trace_params[i,:] = get_free_params(tr, model)
    end        
    
    h5open(h5path, "w") do f
        f["trace_params"] = trace_params
        f["log_weights"] = Gen.get_log_weights(state)
        f["trace_scores"] = trace_scores
        f["n_particles"] = n_particles
        f["sampled_trace_params"] = sampled_trace_params
        f["log_ml_est"] = Gen.log_ml_estimate(state)
    end
end

function mcmc(v, θh, P, ys, n_iters, max_t, model, init_trace=nothing)
    μ_vT = 0.0
    σ_vT = vT_STD
    traces = Vector{Any}(undef, n_iters)
    
    if isnothing(init_trace)
        init_obs = Gen.choicemap()
        for t=1:max_t
            init_obs[:chain => t => :y] = ys[t]
        end
        if model == :nl7b
            (traces[1], _) = generate(unfold_nl7b, (max_t, v, θh, P), init_obs)
        elseif model == :nl8
            (traces[1], _) = generate(nl8, (max_t, v, θh, P), init_obs)
        elseif model == :v
            (traces[1], _) = generate(unfold_v, (max_t, v), init_obs)
        elseif model == :v_noewma
            (traces[1], _) = generate(unfold_v_noewma, (max_t, v), init_obs)
        end
    else
        traces[1] = init_trace
    end
    for iter=2:n_iters
        if model in [:v, :nl7b, :nl8]
            traces[iter] = hmc_jump_update(traces[iter-1], μ_vT, σ_vT, model)
        elseif model == :v_noewma
            traces[iter] = hmc_jump_update_noewma(traces[iter-1], μ_vT, σ_vT, :v_noewma)
        end
    end
    traces
end
