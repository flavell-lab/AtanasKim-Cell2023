"""
    calculate_tuning_strength_per_neuron(
        neuron_deconv, relative_encoding_strength, signal, v_range, θh_range, P_range,
        θh_pos_is_ventral; stat::Function=median, zero_da=nothing
    )

Helper function to calculate tuning strength for a given neuron.

Args:
`neuron_deconv`: Deconvolved neural activity.
`zero_da`: Zero deconvolved activity array.
`relative_encoding_strength`: 
`signal`: Signal value for the current neuron.
`v_range`: Range 
`θh_range`: Difference between θh_ranges.
`P_range`: Difference between P_ranges.
`θh_pos_is_ventral`: Boolean indicating if θh_pos is ventral.

Returns:
A Dict containing tuning strength values for the neuron.
"""
function calculate_tuning_strength_per_neuron(neuron_deconv, relative_encoding_strength, signal, 
        v_range, θh_range, P_range, θh_pos_is_ventral; stat::Function=median, zero_da=nothing)
    if isnothing(zero_da)
        zero_da = zeros(size(neuron_deconv))
    end
    v_diff = v_range[4] - v_range[3] + v_range[2] - v_range[1]
    θh_diff = θh_range[2] - θh_range[1]
    P_diff = P_range[2] - P_range[1]

    neuron_tuning_strength = neuron_p_vals(neuron_deconv, zero_da, signal, 0.0, 0.0, 
            relative_encoding_strength, compute_p=false, metric=identity, stat=stat)

    for k in ["v_rect_neg", "v_rect_pos", "v_fwd", "v_rev"]
        neuron_tuning_strength[k] /= v_diff
    end
    s = size(zero_da)
    for (i,j) in VALID_V_COMPARISONS
        neuron_tuning_strength["v_encoding"][i,j,:,:] ./= abs(v_range[j] - v_range[i])
    end

    for k in ["rev_θh_encoding_inh", "rev_θh_encoding_act", "fwd_θh_encoding_inh", "fwd_θh_encoding_act", 
            "θh_rect_neg", "θh_rect_pos", "θh_neg", "θh_pos"]
        neuron_tuning_strength[k] /= θh_diff
    end

    for k in ["rev_P_encoding_inh", "rev_P_encoding_act", "fwd_P_encoding_inh", "fwd_P_encoding_act", 
            "P_rect_neg", "P_rect_pos", "P_neg", "P_pos"]
        neuron_tuning_strength[k] /= P_diff
    end

    if θh_pos_is_ventral
        neuron_tuning_strength["rev_θh_encoding_ventral"] = neuron_tuning_strength["rev_θh_encoding_act"] 
        neuron_tuning_strength["rev_θh_encoding_dorsal"] = neuron_tuning_strength["rev_θh_encoding_inh"] 
        neuron_tuning_strength["fwd_θh_encoding_ventral"] = neuron_tuning_strength["fwd_θh_encoding_act"] 
        neuron_tuning_strength["fwd_θh_encoding_dorsal"] = neuron_tuning_strength["fwd_θh_encoding_inh"] 
        neuron_tuning_strength["θh_rect_ventral"] = neuron_tuning_strength["θh_rect_pos"] 
        neuron_tuning_strength["θh_rect_dorsal"] = neuron_tuning_strength["θh_rect_neg"] 
        neuron_tuning_strength["θh_ventral"] = neuron_tuning_strength["θh_pos"] 
        neuron_tuning_strength["θh_dorsal"] = neuron_tuning_strength["θh_neg"]
    else
        neuron_tuning_strength["rev_θh_encoding_ventral"] = neuron_tuning_strength["rev_θh_encoding_inh"] 
        neuron_tuning_strength["rev_θh_encoding_dorsal"] = neuron_tuning_strength["rev_θh_encoding_act"] 
        neuron_tuning_strength["fwd_θh_encoding_ventral"] = neuron_tuning_strength["fwd_θh_encoding_inh"]
        neuron_tuning_strength["fwd_θh_encoding_dorsal"] = neuron_tuning_strength["fwd_θh_encoding_act"]
        neuron_tuning_strength["θh_rect_ventral"] = neuron_tuning_strength["θh_rect_neg"]
        neuron_tuning_strength["θh_rect_dorsal"] = neuron_tuning_strength["θh_rect_pos"]
        neuron_tuning_strength["θh_ventral"] = neuron_tuning_strength["θh_neg"]
        neuron_tuning_strength["θh_dorsal"] = neuron_tuning_strength["θh_pos"]
        end
        for k in ["rev_θh_encoding_inh", "rev_θh_encoding_act", "fwd_θh_encoding_inh", "fwd_θh_encoding_act",
                "θh_rect_neg", "θh_rect_pos", "θh_neg", "θh_pos"]
        delete!(neuron_tuning_strength, k)
    end
    return neuron_tuning_strength
end

"""
    get_tuning_strength(fit_results::Dict, deconvolved_activity::Dict, relative_encoding_strength::Dict, θh_pos_is_ventral::Dict, v_ranges::Dict, θh_ranges::Dict, P_ranges::Dict; stat::Function=median, metric::Function=identity)

Helper function to calculate tuning strength for all neurons in all datasets.

Args:
- `fit_results`: A dictionary containing the results of the fitting process.
- `deconvolved_activity`: A dictionary containing the deconvolved neural activity.
- `relative_encoding_strength`: A dictionary containing the relative encoding strength.
- `θh_pos_is_ventral`: A dictionary containing a boolean indicating if θh_pos is ventral.
- `v_ranges`: A dictionary containing the range of v values.
- `θh_ranges`: A dictionary containing the range of θh values.
- `P_ranges`: A dictionary containing the range of P values.
- `stat`: A function to compute the statistic of interest. Default is `median`.
- `metric`: A function to compute the metric of interest. Default is `identity`.

Returns:
A dictionary containing the tuning strength values for all neurons in all datasets.
"""
function get_tuning_strength(fit_results::Dict, deconvolved_activity::Dict, relative_encoding_strength::Dict, θh_pos_is_ventral::Dict, v_ranges::Dict, θh_ranges::Dict, P_ranges::Dict; stat::Function=median, metric::Function=identity)
    tuning_strength = Dict()
    zero_da = nothing
    @showprogress for dataset = keys(fit_results)
        tuning_strength[dataset] = Dict()
        for rng = 1:length(fit_results[dataset]["ranges"])
            tuning_strength[dataset][rng] = Dict()
            v_diff = v_ranges[dataset][rng][4] - v_ranges[dataset][rng][3] + v_ranges[dataset][rng][2] - v_ranges[dataset][rng][1]
            θh_diff = θh_ranges[dataset][rng][2] - θh_ranges[dataset][rng][1]
            P_diff = P_ranges[dataset][rng][2] - P_ranges[dataset][rng][1]
            for neuron = 1:fit_results[dataset]["num_neurons"]
                signal = std(fit_results[dataset]["trace_original"][neuron,:]) / mean(fit_results[dataset]["trace_original"][neuron, :])
                if isnothing(zero_da)
                    zero_da = zeros(size(deconvolved_activity[dataset][rng][neuron]))
                end
                tuning_strength[dataset][rng][neuron] = neuron_p_vals(deconvolved_activity[dataset][rng][neuron], zero_da, signal, 0.0, 0.0, 
                        relative_encoding_strength[dataset][rng][neuron], compute_p=false, metric=metric, stat=stat)

                for k in ["v_rect_neg", "v_rect_pos", "v_fwd", "v_rev"]
                    tuning_strength[dataset][rng][neuron][k] /= v_diff
                end
                s = size(zero_da)
                for (i,j) in VALID_V_COMPARISONS
                    tuning_strength[dataset][rng][neuron]["v_encoding"][i,j,:,:] ./= abs(v_ranges[dataset][rng][j] - v_ranges[dataset][rng][i])
                end

                for k in ["rev_θh_encoding_inh", "rev_θh_encoding_act", "fwd_θh_encoding_inh", "fwd_θh_encoding_act", 
                        "θh_rect_neg", "θh_rect_pos", "θh_neg", "θh_pos"]
                    tuning_strength[dataset][rng][neuron][k] /= θh_diff
                end

                for k in ["rev_P_encoding_inh", "rev_P_encoding_act", "fwd_P_encoding_inh", "fwd_P_encoding_act", 
                    "P_rect_neg", "P_rect_pos", "P_neg", "P_pos"]
                    tuning_strength[dataset][rng][neuron][k] /= P_diff
                end
                if θh_pos_is_ventral[dataset]
                    tuning_strength[dataset][rng][neuron]["rev_θh_encoding_ventral"] = tuning_strength[dataset][rng][neuron]["rev_θh_encoding_act"] 
                    tuning_strength[dataset][rng][neuron]["rev_θh_encoding_dorsal"] = tuning_strength[dataset][rng][neuron]["rev_θh_encoding_inh"] 
                    tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_ventral"] = tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_act"] 
                    tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_dorsal"] = tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_inh"] 
                    tuning_strength[dataset][rng][neuron]["θh_rect_ventral"] = tuning_strength[dataset][rng][neuron]["θh_rect_pos"] 
                    tuning_strength[dataset][rng][neuron]["θh_rect_dorsal"] = tuning_strength[dataset][rng][neuron]["θh_rect_neg"] 
                    tuning_strength[dataset][rng][neuron]["θh_ventral"] = tuning_strength[dataset][rng][neuron]["θh_pos"] 
                    tuning_strength[dataset][rng][neuron]["θh_dorsal"] = tuning_strength[dataset][rng][neuron]["θh_neg"]
                else
                    tuning_strength[dataset][rng][neuron]["rev_θh_encoding_ventral"] = tuning_strength[dataset][rng][neuron]["rev_θh_encoding_inh"] 
                    tuning_strength[dataset][rng][neuron]["rev_θh_encoding_dorsal"] = tuning_strength[dataset][rng][neuron]["rev_θh_encoding_act"] 
                    tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_ventral"] = tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_inh"] 
                    tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_dorsal"] = tuning_strength[dataset][rng][neuron]["fwd_θh_encoding_act"] 
                    tuning_strength[dataset][rng][neuron]["θh_rect_ventral"] = tuning_strength[dataset][rng][neuron]["θh_rect_neg"] 
                    tuning_strength[dataset][rng][neuron]["θh_rect_dorsal"] = tuning_strength[dataset][rng][neuron]["θh_rect_pos"] 
                    tuning_strength[dataset][rng][neuron]["θh_ventral"] = tuning_strength[dataset][rng][neuron]["θh_neg"] 
                    tuning_strength[dataset][rng][neuron]["θh_dorsal"] = tuning_strength[dataset][rng][neuron]["θh_pos"]
                end
                for k in ["rev_θh_encoding_inh", "rev_θh_encoding_act", "fwd_θh_encoding_inh", "fwd_θh_encoding_act", 
                        "θh_rect_neg", "θh_rect_pos", "θh_neg", "θh_pos"]
                    delete!(tuning_strength[dataset][rng][neuron], k)
                end
            end
        end
    end
    return tuning_strength
end

"""
    get_forwardness(tuning_strength::Dict)

Given tuning strength values for a given neuron, returns the forwardness of that neuron.

Arguments:
- `tuning_strength::Dict`: A dictionary containing the tuning strength values for a given neuron.

Returns:
The forwardness of the neuron.
"""
function get_forwardness(tuning_strength)
    return tuning_strength["v_fwd"]
end

"""
    get_dorsalness(tuning_strength::Dict)

Given tuning strength values for a given neuron, returns the dorsalness of that neuron.

Arguments:
- `tuning_strength::Dict`: A dictionary containing the tuning strength values for a given neuron.

Returns:
The dorsalness of the neuron.
"""
function get_dorsalness(tuning_strength)
    return tuning_strength["θh_dorsal"]
end

"""
    get_feedingness(tuning_strength::Dict)

Given tuning strength values for a given neuron, returns the feedingness of that neuron.

Arguments:
- `tuning_strength::Dict`: A dictionary containing the tuning strength values for a given neuron.

Returns:
The feedingness of the neuron.
"""
function get_feedingness(tuning_strength)
    return tuning_strength["P_pos"]
end
