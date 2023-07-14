
"""
    detect_encoding_changes(
        fit_results, p, θh_pos_is_ventral, threshold_artifact, rngs, 
        relative_encoding_strength, datasets; beh_percent=25
    )

Detects all neurons with encoding changes in all datasets across all time ranges.

# Arguments
- `fit_results`: Gen fit results.
- `p`: Significant `p`-value.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
- `threshold_artifact`: Motion artifact threshold for encoding change difference
- `rngs`: Dictionary of which ranges to use per dataset
- `beh_percent` (optional, default `25`): Location to compute behavior percentiles. 
- `relative_encoding_strength`: Relative encoding strength of neurons.
- `datasets`: Datasets to compute encoding changes for.
"""
function detect_encoding_changes(fit_results, p, θh_pos_is_ventral, threshold_artifact, rngs, 
        relative_encoding_strength, datasets; beh_percent=25)
    encoding_changes = Dict()
    encoding_change_p_vals = Dict()
    @showprogress for dataset in datasets
        n_neurons = fit_results[dataset]["num_neurons"]
        n_ranges = length(rngs[dataset])
        v = fit_results[dataset]["v"]
        θh = fit_results[dataset]["θh"]
        P = fit_results[dataset]["P"]
        
        encoding_changes[dataset] = Dict()
        encoding_change_p_vals[dataset] = Dict()

        for i1 = 1:n_ranges-1
            t1 = rngs[dataset][i1]
            range1 = fit_results[dataset]["ranges"][t1]
            v_range_1 = compute_range(v[range1], beh_percent, 1)
            θh_range_1 = compute_range(θh[range1], beh_percent, 2)
            P_range_1 = compute_range(P[range1], beh_percent, 3)

            for i2 = i1+1:n_ranges
                t2 = rngs[dataset][i2]
                range2 = fit_results[dataset]["ranges"][t2]
                v_range_2 = compute_range(v[range2], beh_percent, 1)
                θh_range_2 = compute_range(θh[range2], beh_percent, 2)
                P_range_2 = compute_range(P[range2], beh_percent, 3)

                v_rng = [max(v_range_1[1], v_range_2[1]), max(v_range_1[2], v_range_2[2]),
                            min(v_range_1[3], v_range_2[3]), min(v_range_1[4], v_range_2[4])]
                θh_rng = [max(θh_range_1[1], θh_range_2[1]), min(θh_range_1[2], θh_range_2[2])]
                P_rng = [max(P_range_1[1], P_range_2[1]), min(P_range_1[2], P_range_2[2])]

                # regions don't intersect 
                if sort(v_rng) != v_rng
                    @warn("velocity intervals don't overlap. Skipping...")
                    continue
                end
                if  sort(θh_rng) != θh_rng
                    @warn("head curvature intervals don't overlap. Using mean...")
                    θh_rng = [mean(θh_rng), mean(θh_rng)]
                end
                if sort(P_rng) != P_rng
                    P_rng = [mean(P_rng), mean(P_rng)]
                end
                
                deconvolved_activities_1 = Dict()
                deconvolved_activities_2 = Dict()
                
                for neuron = 1:n_neurons
                    sampled_trace_params_1 = fit_results[dataset]["sampled_trace_params"][t1,neuron,:,:]
                    sampled_trace_params_2 = fit_results[dataset]["sampled_trace_params"][t2,neuron,:,:]
                    
                    deconvolved_activities_1[neuron] = get_deconvolved_activity(sampled_trace_params_1, v_rng, θh_rng, P_rng)
                    deconvolved_activities_2[neuron] = get_deconvolved_activity(sampled_trace_params_2, v_rng, θh_rng, P_rng)
                end
                
                encoding_changes[dataset][(t1, t2)], encoding_change_p_vals[dataset][(t1, t2)] = categorize_neurons(deconvolved_activities_2,
                        deconvolved_activities_1, p, θh_pos_is_ventral[dataset], fit_results[dataset]["trace_original"], threshold_artifact, 0, relative_encoding_strength[dataset][i1],
                        ewma1=fit_results[dataset]["sampled_trace_params"][t2,:,:,7], ewma2=fit_results[dataset]["sampled_trace_params"][t1,:,:,7], compute_feeding=false)
            end
        end
    end
    return encoding_changes, encoding_change_p_vals
end


"""
    correct_encoding_changes(fit_results::Dict, analysis_dict::Dict)

Prunes encoding changes by deleting nonencoding neurons or EWMA-only encoding changes with partially-encoding neurons.

# Arguments:
- `fit_results::Dict`: Dictionary containing CePNEM fit results.
- `analysis_dict::Dict`: Dictionary containing CePNEM analysis results.
"""
function correct_encoding_changes(fit_results::Dict, analysis_dict::Dict)
    encoding_changes_corrected = Dict()
    @showprogress for dataset = keys(analysis_dict["encoding_changes"])
        encoding_changes_corrected[dataset] = Dict()
        for rngs = keys(analysis_dict["encoding_changes"][dataset])
            encoding_changes_corrected[dataset][rngs] = Dict()
            for cat = keys(analysis_dict["encoding_changes"][dataset][rngs])
                if typeof(analysis_dict["encoding_changes"][dataset][rngs][cat]) <: Dict
                    encoding_changes_corrected[dataset][rngs][cat] = Dict()
                    
                    for subcat = keys(analysis_dict["encoding_changes"][dataset][rngs][cat])
                        encoding_changes_corrected[dataset][rngs][cat][subcat] = Int32[]
                        for rng = 1:length(fit_results[dataset]["ranges"])                            
                            for neuron = analysis_dict["encoding_changes"][dataset][rngs][cat][subcat]
                                # neurons must be encoding to be encoding-changing
                                if !(neuron in analysis_dict["neuron_categorization"][dataset][rngs[1]]["all"] || neuron in analysis_dict["neuron_categorization"][dataset][rngs[2]]["all"])
                                    continue
                                end
                                if rng == 1
                                    push!(encoding_changes_corrected[dataset][rngs][cat][subcat], neuron)
                                end
                            end
                        end
                    end
                else
                    encoding_changes_corrected[dataset][rngs][cat] = Int32[]
                    for rng = 1:length(fit_results[dataset]["ranges"])
                        for neuron = analysis_dict["encoding_changes"][dataset][rngs][cat]
                            # neurons must be encoding to be encoding-changing
                            if !(neuron in analysis_dict["neuron_categorization"][dataset][rngs[1]]["all"] || neuron in analysis_dict["neuron_categorization"][dataset][rngs[2]]["all"])
                                continue
                            end
                            # EWMA-only encoding changes require encoding in both time segments
                            if !(neuron in analysis_dict["encoding_changes"][dataset][rngs]["v"]["all"] || neuron in analysis_dict["encoding_changes"][dataset][rngs]["θh"]["all"] || neuron in analysis_dict["encoding_changes"][dataset][rngs]["P"]["all"]) && 
                                     !(neuron in analysis_dict["neuron_categorization"][dataset][rngs[1]]["all"] && neuron in analysis_dict["neuron_categorization"][dataset][rngs[2]]["all"])
                                continue
                            end
                            if rng == 1
                                push!(encoding_changes_corrected[dataset][rngs][cat], neuron)
                            end
                        end
                    end
                end
            end
        end
    end
    return encoding_changes_corrected
end

"""
    get_enc_change_stats(
        fit_results::Dict, enc_change_p::Dict, neuron_p::Dict, datasets::Vector{String};
        rngs_valid::Union{Nothing,Array{Int,1}}=nothing, p::Float64=0.05
    )

Returns statistics on encoding changes across datasets.

# Arguments:
- `fit_results::Dict`: Dictionary containing CePNEM fit results.
- `enc_change_p::Dict`: Dictionary containing encoding change p-values.
- `neuron_p::Dict`: Dictionary containing encoding p-values.
- `datasets::Vector{String}`: List of datasets to analyze.
- `rngs_valid::Union{Nothing,Array{Int,1}}`: Time ranges to analyze.
- `p::Float64`: Significance threshold.

# Returns:
- `n_neurons_tot::Int`: Total number of neurons.
- `n_neurons_enc_change_all::Int`: Number of encoding changing neurons.
- `n_neurons_nenc_enc_change::Int`: Number of non-encoding neurons with encoding changes. This should be 0 if you've properly pruned your encoding changes.
- `n_neurons_enc_change_all::Int`: Number of encoding-changing neurons.
- `n_neurons_enc_change_beh::Array{Int,1}`: Number of encoding-changing neurons that encode behavior.
- `dict_enc_change::Dict`: Dictionary containing encoding change statistics.
- `dict_enc::Dict`: Dictionary containing encoding statistics.
"""
function get_enc_change_stats(fit_results::Dict, enc_change_p::Dict, neuron_p::Dict, datasets::Vector{String}; rngs_valid=nothing, p::Float64=0.05)
    n_neurons_tot = 0
    n_neurons_enc = 0
    n_neurons_nenc_enc_change = 0
    n_neurons_enc_change_all = 0
    n_neurons_enc_change_beh = [0,0,0]
    dict_enc_change = Dict()
    dict_enc = Dict()
    for dataset in datasets
        if rngs_valid == nothing
            rngs_valid = 1:length(fit_results[dataset]["ranges"])
        end
        rngs = [r for r in keys(enc_change_p[dataset]) if (r[1] in rngs_valid && r[2] in rngs_valid)]
        if length(rngs) == 0
            @warn("Dataset $(dataset) has no time ranges where pumping could be compared")
            continue
        end
        dict_enc_change[dataset] = Dict()
        dict_enc[dataset] = Dict()
        
        neurons_ec = [n for n in 1:fit_results[dataset]["num_neurons"] if sum(adjust([enc_change_p[dataset][i]["all"][n] for i=rngs], BenjaminiHochberg()) .< p) > 0]
        
        rngs_enc = [r[1] for r in rngs]
        append!(rngs_enc, [r[2] for r in rngs])
        rngs_enc = unique(rngs_enc)
        neurons_encode = [n for n in 1:fit_results[dataset]["num_neurons"] if sum(adjust([neuron_p[dataset][i]["all"][n] for i=rngs_enc], BenjaminiHochberg()) .< p) > 0]

        n_neurons_enc += length(neurons_encode)        
        n_neurons_enc_change_all += length(neurons_ec)
        n_neurons_nenc_enc_change += length([n for n in neurons_ec if !(n in neurons_encode)])
        dict_enc_change[dataset] = [n for n in neurons_ec if n in neurons_encode]
        dict_enc[dataset] = neurons_encode
        
        n_neurons_tot += fit_results[dataset]["num_neurons"]

        for n=1:fit_results[dataset]["num_neurons"]
            v_p = adjust([enc_change_p[dataset][r]["v"]["all"][n] for r=rngs], BenjaminiHochberg())
            θh_p = adjust([enc_change_p[dataset][r]["θh"]["all"][n] for r=rngs], BenjaminiHochberg())
            P_p = adjust([enc_change_p[dataset][r]["P"]["all"][n] for r=rngs], BenjaminiHochberg())
            if any(v_p .< p)
                n_neurons_enc_change_beh[1] += 1
            end
            
            if any(θh_p .< p)
                n_neurons_enc_change_beh[2] += 1
            end
            
            if any(P_p .< p)
                n_neurons_enc_change_beh[3] += 1
            end
        end
    end
    return n_neurons_tot, n_neurons_enc_change_all, n_neurons_enc, n_neurons_nenc_enc_change, n_neurons_enc_change_beh, dict_enc_change, dict_enc
end

""" 
    encoding_summary_stats(datasets, enc_stat_dict, dict_enc, dict_enc_change, consistent_neurons)

Computes summary statistics.
`function encoding_summary_stats(datasets, enc_stat_dict, dict_enc, dict_enc_change, consistent_neurons)`

`return n_neurons_tot, n_neurons_enc, n_neurons_enc_change, n_neurons_consistent, n_neurons_fully_static,
n_neurons_quasi_static, n_neurons_dynamic, n_neurons_indeterminable`
"""
function encoding_summary_stats(datasets, enc_stat_dict, dict_enc, dict_enc_change, consistent_neurons)
    n_neurons_tot = 0
    n_neurons_enc = 0
    n_neurons_enc_change = 0
    n_neurons_consistent = 0
    n_neurons_fully_static = 0
    n_neurons_quasi_static = 0
    n_neurons_dynamic = 0
    n_neurons_indeterminable = 0
    for dataset in datasets
        n_neurons_tot += enc_stat_dict[dataset]["n_neurons_tot_all"]
        n_neurons_enc += enc_stat_dict[dataset]["n_neurons_fit_all"]
        @assert(length(dict_enc[dataset]) == enc_stat_dict[dataset]["n_neurons_fit_all"],
                "Number of encoding neurons must equal length of encoding neurons.")
        for n in dict_enc[dataset]
            consistent = (n in consistent_neurons[dataset])
            enc_change = (n in dict_enc_change[dataset])
            n_neurons_consistent += consistent
            n_neurons_enc_change += enc_change
            n_neurons_fully_static += (consistent && ~enc_change)
            n_neurons_quasi_static += (consistent && enc_change)
            n_neurons_dynamic += (~consistent && enc_change)
            n_neurons_indeterminable += (~consistent && ~enc_change)
        end
    end
    return n_neurons_tot, n_neurons_enc, n_neurons_enc_change, n_neurons_consistent, n_neurons_fully_static,
            n_neurons_quasi_static, n_neurons_dynamic, n_neurons_indeterminable
end


function get_enc_change_cat_p_vals(enc_change_dict)
    p_val_dict = Dict()
    for beh = ["v", "θh", "P"]
        p_val_dict[beh] = Dict()
        subcats = get_subcats(beh)
        for (sc1, sc2) in subcats
            n1 = enc_change_dict[beh][sc1]
            n2 = enc_change_dict[beh][sc2]
            dist = Binomial(n1+n2, 0.5)
            p_val_dict[beh][sc1] = cdf(dist, n2)
            p_val_dict[beh][sc2] = cdf(dist, n1)
        end
    end
    return p_val_dict
end

function get_enc_change_cat_p_vals_dataset(enc_change_dict, rngs)
    p_val_dict = Dict()
    for beh = ["v", "θh", "P"]
        p_val_dict[beh] = Dict()
        subcats = get_subcats(beh)
        for (sc1, sc2) in subcats
            n1 = 0
            n2 = 0
            for dataset in keys(enc_change_dict)
                if !(rngs in keys(enc_change_dict[dataset]))
                    continue
                end
                n1 += enc_change_dict[dataset][rngs][beh][sc1] > enc_change_dict[dataset][rngs][beh][sc2]
                n2 += enc_change_dict[dataset][rngs][beh][sc2] > enc_change_dict[dataset][rngs][beh][sc1]
            end
            dist = Binomial(n1+n2, 0.5)
            p_val_dict[beh][sc1] = cdf(dist, n2)
            p_val_dict[beh][sc2] = cdf(dist, n1)
        end
    end
    return p_val_dict
end


function get_enc_change_category(dataset, rngs, neuron, encoding_changes)
    encoding_change = []
    for beh in ["v", "θh", "P"]
        for k in keys(encoding_changes[dataset][rngs][beh])
            if neuron in encoding_changes[dataset][rngs][beh][k]
                push!(encoding_change, (beh, k))
            end
        end
    end
    
    for beh in ["ewma_pos", "ewma_neg"]
        if neuron in encoding_changes[dataset][rngs][beh]
            push!(encoding_change, (beh,))
        end
    end

    return encoding_change
end

"""
    compute_feeding_encoding_changes(analysis_dict::Dict, fit_results::Dict, datasets_neuropal_baseline::Vector{String},
        datasets_neuropal_stim::Vector{String}, stim_times::Dict; percent_norm::Real=10, percent_P_thresh::Real=12.5)

This function computes the feeding encoding changes for a given set of datasets. It takes in the following arguments:

* `analysis_dict::Dict`: A dictionary containing the CePNEM analysis results.
* `fit_results::Dict`: A dictionary containing the CePNEM fit results.
* `datasets_neuropal_baseline::Vector{String}`: A list of names of the baseline datasets to use for training.
* `datasets_neuropal_stim::Vector{string}`: A list of names of the heat-stimulation datasets to use for testing.
* `stim_times::Dict`: A dictionary containing the stimulation times for each heat-stim dataset.
* `percent_norm::Real=10`: Fluorescence normalization percentile across datasets (ie: if set at 10, the analysis will use F/F10)
* `percent_P_thresh::Real=12.5`: Fraction of time points the animal needs  of neurons to threshold based on P-values.

The function returns a dictionary containing the following keys:

* `mse_train`: A dictionary containing the mean squared error (MSE) for each dataset in the training set.
* `mse_train_null`: A dictionary containing the MSE for the null model for each dataset in the training set.
* `p_val`: A dictionary containing the P-values for the encoding changes (ie: whether the pre-stimulus model outperformed the post-stimulus model).
* `mse_prestim`: A dictionary containing the MSE for the prestimulus datasets.
* `mse_poststim`: A dictionary containing the MSE for the poststimulus datasets.
* `mse_prestim_null`: A dictionary containing the MSE for the null model on the prestimulus datasets.
* `mse_poststim_null`: A dictionary containing the MSE for the null model on the poststimulus datasets.
* `prestim_fits`: A dictionary containing the fit to pumping for the appended prestimulus datasets.
* `poststim_fits`: A dictionary containing the fit to pumping for the appended poststimulus datasets.
* `prestim_neurons`: A dictionary containing the appended neural data for the prestimulus datasets.
* `poststim_neurons`: A dictionary containing the appended neural data for the poststimulus datasets.
* `prestim_P`: A dictionary containing the appended pumping values for the prestimulus datasets.
* `poststim_P`: A dictionary containing the appended pumping values for the poststimulus datasets.
"""
function compute_feeding_encoding_changes(analysis_dict::Dict, fit_results::Dict, datasets_neuropal_baseline::Vector{String},
        datasets_neuropal_stim::Vector{String}, stim_times::Dict; percent_norm::Real=10, percent_P_thresh::Real=12.5)

    dict_result = Dict()
    dict_result["mse_train"] = Dict()
    dict_result["mse_train_null"] = Dict()

    dict_result["p_val"] = Dict()
    dict_result["mse_prestim"] = Dict()
    dict_result["mse_poststim"] = Dict()
    dict_result["mse_prestim_null"] = Dict()
    dict_result["mse_poststim_null"] = Dict()
    dict_result["prestim_fits"] = Dict()
    dict_result["poststim_fits"] = Dict()
    dict_result["prestim_neurons"] = Dict()
    dict_result["poststim_neurons"] = Dict()
    dict_result["prestim_P"] = Dict()
    dict_result["poststim_P"] = Dict()

    @showprogress for neuron_name = keys(analysis_dict["matches"])
        datasets_use = datasets_neuropal_baseline

        P_appended = Float64[]
        n_appended = Float64[]
        for (dataset, n) = analysis_dict["matches"][neuron_name]
            if dataset ∉ datasets_use
                continue
            end
            # need to ensure F/F10 actually produces minimum value
            if percentile(fit_results[dataset]["P"], percent_P_thresh) > 0
                continue
            end
            append!(P_appended, fit_results[dataset]["P"])
            if n > size(fit_results[dataset]["trace_original"], 1)
                @warn("Neuron $neuron_name, dataset $dataset, n $n")
            end
            append!(n_appended, fit_results[dataset]["trace_original"][n,:] ./ percentile(fit_results[dataset]["trace_original"][n,:], percent_norm))
        end

        if length(n_appended) == 0
            continue
        end

        # fit GLM model
        data = DataFrame(P=P_appended, n=n_appended)
        model = lm(@formula(P ~ n + 1), data)

        y_pred = GLM.predict(model, data)

        dict_result["mse_train"][neuron_name] = mean((P_appended .- y_pred).^2)
        dict_result["mse_train_null"][neuron_name] = mean((P_appended .- mean(P_appended)).^2)


        datasets_fit = datasets_neuropal_stim
        prestim_neurons = Float64[]
        prestim_fits = Float64[]
        prestim_P = Float64[]
        poststim_neurons = Float64[]
        poststim_fits = Float64[]
        poststim_P = Float64[]
        mse_prestim = Float64[]
        null_mse_prestim = Float64[]
        mse_poststim = Float64[]
        null_mse_poststim = Float64[]
        for (dataset, n) = analysis_dict["matches"][neuron_name]
            if dataset ∉ datasets_fit
                continue
            end
            for rng=1:2
                # need to ensure F/F10 actually produces minimum value on the hypothesized-consistent data range (1)
                if percentile(fit_results[dataset]["P"][fit_results[dataset]["ranges"][1]], percent_P_thresh*maximum(fit_results[dataset]["ranges"][end])/maximum(fit_results[dataset]["ranges"][1])) > 0
                    continue
                end
                perc = fit_results[dataset]["trace_original"][n,fit_results[dataset]["ranges"][rng]] ./ percentile(fit_results[dataset]["trace_original"][n,:], percent_norm)
                n_vals = perc

                y_pred = GLM.predict(model, DataFrame(n=n_vals))
                if rng == 1
                    push!(mse_prestim, mean((fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]] .- y_pred).^2))
                    append!(prestim_fits, y_pred)
                    append!(prestim_neurons, n_vals)
                    append!(prestim_P, fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]])
                    push!(null_mse_prestim, mean((fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]] .- mean(P_appended)).^2))
                elseif rng == 2
                    push!(mse_poststim, mean((fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]] .- y_pred).^2))
                    append!(poststim_fits, y_pred)
                    append!(poststim_neurons, n_vals)
                    append!(poststim_P, fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]])
                    push!(null_mse_poststim, mean((fit_results[dataset]["P"][fit_results[dataset]["ranges"][rng]] .- mean(P_appended)).^2))
                end
            end
        end

        if length(mse_prestim) == 0
            continue
        end

        prestim = mse_prestim .- null_mse_prestim
        poststim = mse_poststim .- null_mse_poststim
        dict_result["mse_prestim"][neuron_name] = mse_prestim
        dict_result["mse_poststim"][neuron_name] = mse_poststim
        dict_result["mse_prestim_null"][neuron_name] = null_mse_prestim
        dict_result["mse_poststim_null"][neuron_name] = null_mse_poststim
        dict_result["prestim_fits"][neuron_name] = prestim_fits
        dict_result["poststim_fits"][neuron_name] = poststim_fits
        dict_result["prestim_neurons"][neuron_name] = prestim_neurons
        dict_result["poststim_neurons"][neuron_name] = poststim_neurons
        dict_result["prestim_P"][neuron_name] = prestim_P
        dict_result["poststim_P"][neuron_name] = poststim_P

        if length(prestim) > 1 && length(poststim) > 1
            p_val = pvalue(SignedRankTest(prestim, poststim), tail=:left)
        else
            p_val = 1.0
        end
        dict_result["p_val"][neuron_name] = p_val
    end
    return dict_result
end
