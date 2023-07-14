"""
    get_relative_encoding_strength!(ps::Vector{Float64}, b_v::Vector{Float64}, b_θh::Vector{Float64}, b_P::Vector{Float64}; b_null::Union{Vector{Float64}, Nothing}=nothing)

Computes the relative encoding strength of the three behaviors, together with standard deviations of full and deconvolved model fits, for a single CePNEM fit.

# Arguments
- `ps::Vector{Float64}`: Vector of model parameters.
- `b_v::Vector{Float64}`: Vector of behavioral values for the velocity behavior.
- `b_θh::Vector{Float64}`: Vector of behavioral values for the heading behavior.
- `b_P::Vector{Float64}`: Vector of behavioral values for the persistence behavior.
- `b_null::Union{Vector{Float64}, Nothing}` (optional, default `nothing`): Vector of behavioral values for the null behavior. If `nothing`, a vector of zeros is used.
"""
function get_relative_encoding_strength!(ps::Vector{Float64}, b_v::Vector{Float64}, b_θh::Vector{Float64}, b_P::Vector{Float64}; b_null::Union{Vector{Float64}, Nothing}=nothing)
    max_t = length(b_v)

    if isnothing(b_null)
        b_null = zeros(max_t)
    end

    model_full = model_nl8(max_t, ps..., b_v, b_θh, b_P)
    model_nov = model_nl8(max_t, ps..., b_null, b_θh, b_P)
    model_noθh = model_nl8(max_t, ps..., b_v, b_null, b_P)
    model_noP = model_nl8(max_t, ps..., b_v, b_θh, b_null)

    mean_v = mean(b_v)
    mean_θh = mean(b_θh)
    mean_P = mean(b_P)

    model_deconv = zeros(max_t)
    model_deconv_v = zeros(max_t)
    model_deconv_θh = zeros(max_t)
    model_deconv_P = zeros(max_t)

    for i_t = 1:max_t
        model_deconv[i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_θh[i_t], b_P[i_t])
        model_deconv_v[i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_null[i_t], b_null[i_t])
        model_deconv_θh[i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_θh[i_t], b_null[i_t])
        model_deconv_P[i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_null[i_t], b_P[i_t])
    end       

    ps[6] = mean(model_deconv) # set initial condition to mean so as not to contaminate convolution computation
    model_full_ps6corr = model_nl8(max_t, ps..., b_v, b_θh, b_P)

    ps[6] = mean(model_deconv_v)
    model_v = model_nl8(max_t, ps..., b_v, b_null, b_null)

    ps[6] = mean(model_deconv_θh)
    model_θh = model_nl8(max_t, ps..., b_null, b_θh, b_null)

    ps[6] = mean(model_deconv_P)
    model_P = model_nl8(max_t, ps..., b_null, b_null, b_P)

    std_full = std(model_full)
    std_v = std(model_v)
    std_θh = std(model_θh)
    std_P = std(model_P)

    std_full_ps6corr = std(model_full_ps6corr)
    std_deconv = std(model_deconv)
    std_deconv_v = std(model_deconv_v)
    std_deconv_θh = std(model_deconv_θh)
    std_deconv_P = std(model_deconv_P)

    ev = cost_mse(model_full, model_nov)
    eθh = cost_mse(model_full, model_noθh)
    eP = cost_mse(model_full, model_noP)

    s = ev + eθh + eP
    err_v = ev / s
    err_θh = eθh / s
    err_P = eP / s

    dict_ret = Dict()
    dict_ret["v"] = err_v
    dict_ret["θh"] = err_θh
    dict_ret["P"] = err_P
    dict_ret["std"] = std_full
    dict_ret["var"] = std_full ^ 2
    dict_ret["var_allbeh"] = s
    dict_ret["std_ps6corr"] = std_full_ps6corr
    dict_ret["std_v"] = std_v
    dict_ret["std_θh"] = std_θh
    dict_ret["std_P"] = std_P
    dict_ret["std_deconv"] = std_deconv
    dict_ret["std_deconv_v"] = std_deconv_v
    dict_ret["std_deconv_θh"] = std_deconv_θh
    dict_ret["std_deconv_P"] = std_deconv_P
    dict_ret["model_full"] = model_full

    return dict_ret
end


"""
    get_relative_encoding_strength_mt(fit_results::Dict, dataset::String, rng::Int, neuron::Int; max_idx::Int=10001, dataset_mapping=nothing)

Computes the relative encoding strength of the three behaviors, together with standard deviations of full and deconvolved model fits.

# Arguments
- `fit_results::Dict`: Gen fit results.
- `dataset::String`: Dataset to use
- `rng::Int`: Range in that dataset to use
- `neuron::Int`: Neuron to use
- `max_idx::Int` (optional, default `10001`): Maximum Gen posterior sample index.
- `dataset_mapping` (optional, default `nothing`): Dictionary mapping to use different behavioral dataset.
"""
function get_relative_encoding_strength_mt(fit_results::Dict, dataset::String, rng::Int, neuron::Int; max_idx::Int=10001, dataset_mapping=nothing)
    dataset_fit = fit_results[dataset]
    ps_fit = deepcopy(dataset_fit["sampled_trace_params"])
    rng_t = dataset_fit["ranges"][rng]
    max_t = length(rng_t)
    
    dset = (isnothing(dataset_mapping) ? dataset : dataset_mapping[dataset])
    b_v = fit_results[dset]["v"][rng_t]
    b_θh = fit_results[dset]["θh"][rng_t]
    b_P = fit_results[dset]["P"][rng_t]
    b_null = zeros(max_t)
    
    std_deconv = zeros(max_idx)
    std_deconv_v = zeros(max_idx)
    std_deconv_θh = zeros(max_idx)
    std_deconv_P = zeros(max_idx)

    std_full = zeros(max_idx)
    std_full_ps6corr = zeros(max_idx)
    std_v = zeros(max_idx)
    std_θh = zeros(max_idx)
    std_P = zeros(max_idx)

    err_v = zeros(max_idx)
    err_θh = zeros(max_idx)
    err_P = zeros(max_idx)
    
    model_deconv = zeros(max_idx, max_t)
    model_deconv_v = zeros(max_idx, max_t)
    model_deconv_θh = zeros(max_idx, max_t)
    model_deconv_P = zeros(max_idx, max_t)
        
    @sync Threads.@threads for idx=1:max_idx
        ps = ps_fit[rng,neuron,idx,1:8]

        model_full = model_nl8(max_t, ps..., b_v, b_θh, b_P)
        model_nov = model_nl8(max_t, ps..., b_null, b_θh, b_P)
        model_noθh = model_nl8(max_t, ps..., b_v, b_null, b_P)
        model_noP = model_nl8(max_t, ps..., b_v, b_θh, b_null)

        mean_v = mean(b_v)
        mean_θh = mean(b_θh)
        mean_P = mean(b_P)
        
        for i_t = 1:length(rng_t)
            model_deconv[idx, i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_θh[i_t], b_P[i_t])
            model_deconv_v[idx, i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_null[i_t], b_null[i_t])
            model_deconv_θh[idx, i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_θh[i_t], b_null[i_t])
            model_deconv_P[idx, i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_null[i_t], b_P[i_t])
        end       
        
        ps[6] = mean(model_deconv[idx,:]) # set initial condition to mean so as not to contaminate convolution computation
        model_full_ps6corr = model_nl8(max_t, ps..., b_v, b_θh, b_P)

        ps[6] = mean(model_deconv_v[idx,:])
        model_v = model_nl8(max_t, ps..., b_v, b_null, b_null)

        ps[6] = mean(model_deconv_θh[idx,:])
        model_θh = model_nl8(max_t, ps..., b_null, b_θh, b_null)

        ps[6] = mean(model_deconv_P[idx,:])
        model_P = model_nl8(max_t, ps..., b_null, b_null, b_P)
        
        std_full[idx] = std(model_full)
        std_v[idx] = std(model_v)
        std_θh[idx] = std(model_θh)
        std_P[idx] = std(model_P)

        std_full_ps6corr[idx] = std(model_full_ps6corr)
        std_deconv[idx] = std(model_deconv[idx,:])
        std_deconv_v[idx] = std(model_deconv_v[idx,:])
        std_deconv_θh[idx] = std(model_deconv_θh[idx,:])
        std_deconv_P[idx] = std(model_deconv_P[idx,:])

        ev = cost_mse(model_full, model_nov)
        eθh = cost_mse(model_full, model_noθh)
        eP = cost_mse(model_full, model_noP)

        s = ev + eθh + eP
        err_v[idx] = ev / s
        err_θh[idx] = eθh / s
        err_P[idx] = eP / s
    end
    
    return err_v, err_θh, err_P, std_full, std_full_ps6corr, std_v, std_θh, std_P, std_deconv, std_deconv_v, std_deconv_θh, std_deconv_P
end

"""
    get_relative_encoding_strength(fit_results::Dict, datasets::Vector{String}; dataset_mapping=nothing)

Computes the relative encoding strength of all neurons in all datasets.

# Arguments
- `fit_results::Dict`: Dictionary of Gen fit results.
- `datasets::Vector{String}`: List of datasets to compute relative encoding strength for.
"""
function get_relative_encoding_strength(fit_results::Dict, datasets::Vector{String}; dataset_mapping=nothing)
    relative_encoding_strength = Dict()
    @showprogress for dataset = datasets
        relative_encoding_strength[dataset] = Dict()
        for rng=1:length(fit_results[dataset]["ranges"])
            relative_encoding_strength[dataset][rng] = Dict()
            for neuron=1:fit_results[dataset]["num_neurons"]
                v, θh, P, σ, σ_nops6, σ_v, σ_θh, σ_P, σ_d, σ_dv, σ_dθh, σ_dP = get_relative_encoding_strength_mt(fit_results, dataset, rng, neuron, dataset_mapping=dataset_mapping)
                relative_encoding_strength[dataset][rng][neuron] = Dict()
                relative_encoding_strength[dataset][rng][neuron]["v"] = v
                relative_encoding_strength[dataset][rng][neuron]["θh"] = θh
                relative_encoding_strength[dataset][rng][neuron]["P"] = P
                relative_encoding_strength[dataset][rng][neuron]["std"] = σ
                relative_encoding_strength[dataset][rng][neuron]["std_ps6corr"] = σ_nops6
                relative_encoding_strength[dataset][rng][neuron]["std_v"] = σ_v
                relative_encoding_strength[dataset][rng][neuron]["std_θh"] = σ_θh
                relative_encoding_strength[dataset][rng][neuron]["std_P"] = σ_P
                relative_encoding_strength[dataset][rng][neuron]["std_deconv"] = σ_d
                relative_encoding_strength[dataset][rng][neuron]["std_deconv_v"] = σ_dv
                relative_encoding_strength[dataset][rng][neuron]["std_deconv_θh"] = σ_dθh
                relative_encoding_strength[dataset][rng][neuron]["std_deconv_P"] = σ_dP
            end
        end
    end
    return relative_encoding_strength
end
