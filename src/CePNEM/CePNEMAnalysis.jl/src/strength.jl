"""
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
Computes the relative encoding strength of all neurons in all datasets.

# Arguments
- `fit_results::Dict`: Dictionary of Gen fit results.
- `datasets::Vector{String}`: List of datasets to compute relative encoding strength for.
"""
function get_relative_encoding_strength(fit_results::Dict, datasets::Vector{String})
    relative_encoding_strength = Dict()
    @showprogress for dataset = datasets
        relative_encoding_strength[dataset] = Dict()
        for rng=1:length(fit_results[dataset]["ranges"])
            relative_encoding_strength[dataset][rng] = Dict()
            for neuron=1:fit_results[dataset]["num_neurons"]
                v, θh, P, σ, σ_nops6, σ_v, σ_θh, σ_P, σ_d, σ_dv, σ_dθh, σ_dP = get_relative_encoding_strength_mt(fit_results, dataset, rng, neuron)
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

"""
Computes the median encoding change and encoding strength for each neuron in each dataset.
Additionally, combines data from all encoding neurons in each dataset category.

# Arguments
- `fit_results::Dict`: Dictionary of Gen fit results.
- `analysis_dict::Dict`: Dictionary of analysis results, including neuron categorization and encoding changes
- `dataset_cats::Dict{String,Vector{String}}`: Dictionary of dataset categories to combine.
- `exclude_pumping::Bool` (optional, default `true`): Whether to exclude pumping encoding from analysis
"""
function compute_encoding_change_strength(fit_results::Dict, analysis_dict::Dict, dataset_cats::Dict{String,Vector{String}}, exclude_pumping::Bool=true)
    enc_change_strength = Dict()
    enc_strength_dict = Dict()

    for k in keys(dataset_cats)
        enc_change_strength[k] = Dict()
        enc_strength_dict[k] = Dict()
    end

    enc_change_strength["other"] = Dict()
    enc_strength_dict["other"] = Dict()

    @showprogress for dataset in keys(fit_results)
        if length(fit_results[dataset]["ranges"]) < 2
            continue
        end

        dsets = "other"

        for k in keys(dataset_cats)
            if dataset in dataset_cats[k]
                dsets = k
                break
            end
        end

        enc_change_strength[dataset] = Dict()
        enc_strength_dict[dataset] = Dict()

        for n in 1:fit_results[dataset]["num_neurons"]
            ℓ = length(fit_results[dataset]["v"])
            sample_fits = zeros(length(fit_results[dataset]["ranges"]),ℓ)
            for rng=1:length(fit_results[dataset]["ranges"])
                if !haskey(enc_strength_dict[dataset], rng)
                    enc_strength_dict[dataset][rng] = Dict()
                end

                if !haskey(enc_strength_dict[dsets], rng)
                    enc_strength_dict[dsets][rng] = Float64[]
                end


                ps = deepcopy(median(fit_results[dataset]["sampled_trace_params"][rng,n,:,1:8], dims=1))
                ps[6] = 0.0

                model = model_nl8(ℓ, ps..., fit_results[dataset]["v"], fit_results[dataset]["θh"], (exclude_pumping ? fit_results[dataset]["P"] : zeros(ℓ)))
                sample_fits[rng,:] .= (model .- mean(model)) .* analysis_dict["signal"][dataset][n]
                
                es = cost_mse(sample_fits[rng,:], zeros(ℓ))
                enc_strength_dict[dataset][rng][n] = es
                if n in analysis_dict["neuron_categorization"][dataset][rng]["all"]
                    push!(enc_strength_dict[dsets][rng], es)
                end
            end

            for rng1 in 1:length(fit_results[dataset]["ranges"])-1
                for rng2 in rng1+1:length(fit_results[dataset]["ranges"])
                    rngs = (rng1, rng2)

                    if !haskey(enc_change_strength[dataset], rngs)
                        enc_change_strength[dataset][rngs] = Dict()
                    end

                    if !haskey(enc_change_strength[dsets], rngs)
                        enc_change_strength[dsets][rngs] = Float64[]
                    end

                    enc_change_strength[dataset][n] = Dict()
                    enc_change_strength[dataset][rngs][n] = cost_mse(sample_fits[rng1,:], sample_fits[rng2,:]) / max(enc_strength_dict[dataset][rng1][n], enc_strength_dict[dataset][rng2][n])
                    
                    if n in analysis_dict["encoding_changing_neurons_msecorrect_mh"][dataset][rngs]["neurons"]
                        push!(enc_change_strength[dsets][rngs], enc_change_strength[dataset][rngs][n])
                    end
                end
            end    
        end
    end
    return enc_strength_dict, enc_change_strength
end

