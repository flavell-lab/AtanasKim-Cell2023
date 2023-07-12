
"""
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
function detect_encoding_changes(fit_results, p, θh_pos_is_ventral, threshold_artifact, rngs, relative_encoding_strength, datasets; beh_percent=25)
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
Corrects encoding changes by deleting nonencoding neurons or EWMA-only encoding changes with partially-encoding neurons.

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


function get_enc_change_stats(fit_results, enc_change_p, neuron_p, datasets; rngs_valid=nothing, p=0.05)
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

""" Computes summary statistics.
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

function get_subcats(beh)
    if beh == "v"
        subcats = [("rev_slope_neg", "rev_slope_pos"), ("rev", "fwd"), ("rect_neg", "rect_pos"), ("fwd_slope_neg", "fwd_slope_pos")]
    elseif beh == "θh"
        subcats = [("rev_dorsal", "rev_ventral"), ("dorsal", "ventral"), ("rect_dorsal", "rect_ventral"), ("fwd_dorsal", "fwd_ventral")]
    elseif beh == "P"
        subcats = [("rev_inh", "rev_act"), ("inh", "act"), ("rect_inh", "rect_act"), ("fwd_inh", "fwd_act")]
    end 
    return subcats
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

function MSE_correct_encoding_changes(analysis_dict, datasets)
    encoding_changing_neurons_msecorrect = Dict()
    encoding_changing_neurons_msecorrect_mh = Dict()
    for dataset = datasets
        encoding_changing_neurons_msecorrect[dataset] = Dict()
        encoding_changing_neurons_msecorrect_mh[dataset] = Dict()
        for rngs in keys(analysis_dict["encoding_changes_corrected"][dataset])
            encoding_changing_neurons_msecorrect[dataset][rngs] = Dict()
            encoding_changing_neurons_msecorrect_mh[dataset][rngs] = Dict()
            encoding_changing_neurons_msecorrect_mh[dataset][rngs]["p_vals"] = Dict()
            encoding_changing_neurons_msecorrect_mh[dataset][rngs]["neurons"] = Int32[]
            
            p_vals = Float64[]
            for neuron in analysis_dict["encoding_changes_corrected"][dataset][rngs]["all"]
                combined_better = [prob_P_greater_Q(analysis_dict["mse_ewma_skip"][dataset][neuron][rngs[i],rngs[i],:],
                        [analysis_dict["mse_fits_combinedtrain"][dataset][rngs][neuron]["cost_test"][rngs[i]]]) for i=1:2]
                encoding_changing_neurons_msecorrect[dataset][rngs][neuron] = combined_better
                push!(p_vals, min(1,minimum(combined_better) * length(combined_better)))
            end
            if length(p_vals) == 0
                continue
            end
            p_vals = adjust(p_vals, BenjaminiHochberg())
            for (i,neuron) in enumerate(analysis_dict["encoding_changes_corrected"][dataset][rngs]["all"])
                encoding_changing_neurons_msecorrect_mh[dataset][rngs]["p_vals"][neuron] = p_vals[i]
                if p_vals[i] < 0.05
                    push!(encoding_changing_neurons_msecorrect_mh[dataset][rngs]["neurons"], neuron)
                end
            end
        end
    end
    return encoding_changing_neurons_msecorrect, encoding_changing_neurons_msecorrect_mh
end
