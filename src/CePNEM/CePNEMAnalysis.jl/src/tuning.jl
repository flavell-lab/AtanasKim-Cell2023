function get_tuning_strength(fit_results::Dict, deconvolved_activity::Dict, relative_encoding_strength::Dict, θh_pos_is_ventral::Dict, v_ranges::Dict, θh_ranges::Dict, P_ranges::Dict; stat::Function=median)
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
                        relative_encoding_strength[dataset][rng][neuron], compute_p=false, metric=identity, stat=stat)

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

function get_forwardness(tuning_strength)
    return tuning_strength["v_fwd"]
end

function get_dorsalness(tuning_strength)
    return tuning_strength["θh_dorsal"]
end
