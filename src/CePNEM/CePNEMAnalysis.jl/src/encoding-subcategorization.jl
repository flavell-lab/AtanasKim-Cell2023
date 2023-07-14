"""
    subcategorize_all_neurons!(fit_results::Dict, analysis_dict::Dict, datasets::Vector{String})

Subcategorizes all neurons in the given `datasets` based on their encoding properties and stores the results in `analysis_dict`.

# Arguments
- `fit_results::Dict`: A dictionary containing the results of fitting CePNEM to the data.
- `analysis_dict::Dict`: A dictionary to store the results of the analysis.
- `datasets::Vector{String}`: A vector of dataset names to analyze.

# Output
- `analysis_dict` is updated with the results of the analysis.
"""
function subcategorize_all_neurons!(fit_results::Dict, analysis_dict::Dict, datasets::Vector{String})
    v_keys = ["fwd_slope_pos", "fwd_slope_neg", "rev_slope_pos", "rev_slope_neg", "rect_pos", "rect_neg"]
    θh_keys = ["fwd_ventral", "fwd_dorsal", "rev_ventral", "rev_dorsal", "rect_ventral", "rect_dorsal"]
    P_keys = ["fwd_act", "fwd_inh", "rev_act", "rev_inh", "rect_act", "rect_inh"]

    analysis_dict["v_enc"] = Dict()
    for k in v_keys
        analysis_dict["v_enc"][k] = []
    end
    analysis_dict["θh_enc"] = Dict()
    for k in θh_keys
        analysis_dict["θh_enc"][k] = []
    end
    analysis_dict["P_enc"] = Dict()
    for k in P_keys
        analysis_dict["P_enc"][k] = []
    end
    num_possible_encodings = length(v_keys) + length(θh_keys) + length(P_keys)
    analysis_dict["joint_encoding"] = zeros(num_possible_encodings, num_possible_encodings)
    tot = 0
    analysis_dict["v_enc_matrix"] = zeros(3,3)
    analysis_dict["θh_enc_matrix"] = zeros(3,3)
    analysis_dict["P_enc_matrix"] = zeros(3,3)
    analysis_dict["neuron_subcategorization"] = Dict()
    subcategories = ["analog_pos", "analog_neg", "fwd_slope_pos_rect_pos", "rev_slope_pos_rect_neg", "fwd_slope_neg_rect_neg", "rev_slope_neg_rect_pos", "fwd_pos_rev_neg", "rev_pos_fwd_neg", "unknown_enc", "nonencoding"]
    analysis_dict["joint_subencoding"] = zeros(3*length(subcategories), 3*length(subcategories))

    # rect fwd inh, rect fwd, unknown
    # slow, linear fwd, rect rev inh
    # linear rev, speed, rect rev
    for dataset in datasets
        analysis_dict["neuron_subcategorization"][dataset] = Dict()
        for rng in 1:length(fit_results[dataset]["ranges"])
            analysis_dict["neuron_subcategorization"][dataset][rng] = Dict()
            for beh in ["v", "θh", "P"]
                analysis_dict["neuron_subcategorization"][dataset][rng][beh] = Dict()
                for cat in subcategories
                    analysis_dict["neuron_subcategorization"][dataset][rng][beh][cat] = []
                end
            end
                
            count = 0
            for neuron in 1:fit_results[dataset]["num_neurons"]
                encs_all = zeros(Bool, num_possible_encodings)
                idx=1
                
                for (beh, beh_enc, beh_keys, beh_enc_matrix) in [("v", "v_enc", v_keys, "v_enc_matrix"), 
                            ("θh", "θh_enc", θh_keys, "θh_enc_matrix"), ("P", "P_enc", P_keys, "P_enc_matrix")]

                    encs = zeros(Bool,length(beh_keys))
                    for (i,k) in enumerate(beh_keys)
                        if neuron in analysis_dict["neuron_categorization"][dataset][rng][beh][k]
                            push!(analysis_dict[beh_enc][k], (dataset, neuron))
                            encs[i] = 1
                        end
                    end
                    
                    @assert(!(encs[1] && encs[2]))
                    @assert(!(encs[3] && encs[4]))
                    @assert(!(encs[5] && encs[6]))
                    if encs[1] && encs[4] # speed neuron
                        @assert(encs[5]) # speed neurons must be rectified
                        analysis_dict[beh_enc_matrix][2,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_pos_rev_neg"], neuron)
                    elseif encs[2] && encs[3] # slow neuron
                        @assert(encs[6]) # slow neurons must be rectified
                        analysis_dict[beh_enc_matrix][1,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_pos_fwd_neg"], neuron)
                    elseif encs[1] && encs[5] # forward positively-rectified
                        analysis_dict[beh_enc_matrix][3,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_pos_rect_pos"], neuron)
                    elseif encs[4] && encs[5] # reversal positively-rectified
                        analysis_dict[beh_enc_matrix][2,3] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_neg_rect_pos"], neuron)
                    elseif encs[2] && encs[6] # reversal negatively-rectified
                        analysis_dict[beh_enc_matrix][3,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_neg_rect_neg"], neuron)
                    elseif encs[3] && encs[6] # forward negatively-rectified
                        analysis_dict[beh_enc_matrix][1,3] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_pos_rect_neg"], neuron)
                    elseif encs[2] && encs[4] # linear reversal
                        analysis_dict[beh_enc_matrix][2,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_neg"], neuron)
                    elseif encs[1] && encs[3] # linear forward
                        analysis_dict[beh_enc_matrix][1,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_pos"], neuron)
                    elseif any(encs[1:6])
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["unknown_enc"], neuron)
                        analysis_dict[beh_enc_matrix][3,3] += 1
                        @assert(sum(encs[1:6]) == 1)
                    else
                        @assert(sum(encs[1:6]) == 0)
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["nonencoding"], neuron)
                    end
                    
                    encs_all[idx:idx+length(beh_keys)-1] .= encs
                    idx += length(beh_keys)
                end
                
                for i=1:length(encs_all)
                    for j=i+1:length(encs_all)
                        if encs_all[i] && encs_all[j]
                            analysis_dict["joint_encoding"][i,j] += 1
                            analysis_dict["joint_encoding"][j,i] += 1
                        end
                    end
                    if encs_all[i]
                        analysis_dict["joint_encoding"][i,i] += 1
                    end
                end
                
                for (i, beh1) in enumerate(["v", "θh", "P"])
                    for (j, cat1) in enumerate(subcategories)
                        for (x, beh2) in enumerate(["v", "θh", "P"])
                            for (y, cat2) in enumerate(subcategories)
                                if neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh1][cat1] && neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh2][cat2] 
                                    analysis_dict["joint_subencoding"][length(subcategories)*(i-1)+j,length(subcategories)*(x-1)+y] += 1
                                end
                            end
                        end
                    end
                end
                tot += 1
            end
        end
    end
end

"""
    sum_subencodings(
        fit_results::Dict, analysis_dict::Dict, relative_encoding_strength::Dict,
        datasets::Vector{String}; mode::String="relative", sufficient_threshold::Float64=1/3
    )

Calculate the sum of subencodings for each behavior and subcategory.

# Arguments
- `fit_results::Dict`: A dictionary containing the results of fitting CePNEM to the data.
- `analysis_dict::Dict`: A dictionary containing the CePNEM analysis results, specifically the neuron categorization and relative encoding strength.
- `relative_encoding_strength::Dict`: A dictionary containing the relative encoding strength for each behavior in each neuron.
- `datasets::Vector{String}`: A vector of strings containing the names of the datasets to analyze.
- `mode::String`: A string indicating the mode of computation. Possible values are "relative", "encoding", and "sufficient". Default is "relative".
- `sufficient_threshold::Float64`: A float indicating the threshold for sufficient encoding. Default is 1/3. Does nothing for relative and sufficient modes.

# Returns
- `dict_result::Dict`: A dictionary containing the sum of subencodings for each behavior and subcategory.
"""
function sum_subencodings(fit_results::Dict, analysis_dict::Dict, relative_encoding_strength::Dict, datasets::Vector{String}; mode::String="relative", sufficient_threshold::Float64=1/3)
    dict_result = Dict()
    for dataset in datasets
        for rng=1:length(fit_results[dataset]["ranges"])
            for beh = ["v", "θh", "P"]
                if !haskey(dict_result, beh)
                    dict_result[beh] = Dict()
                end
                for k = keys(analysis_dict["neuron_subcategorization"][dataset][rng][beh])
                    if !haskey(dict_result[beh], k)
                        dict_result[beh][k] = 0.0
                    end
                    for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh][k]
                        val = NaN
                        if mode == "relative"
                            val = median(relative_encoding_strength[dataset][rng][neuron][beh])
                        elseif mode == "encoding"
                            val = 1
                        elseif mode == "sufficient"
                            val = (median(relative_encoding_strength[dataset][rng][neuron][beh]) >= sufficient_threshold ? 1 : 0)
                        end
                        dict_result[beh][k] += val
                    end
                end
            end
        end
    end
    return dict_result
end

"""
    normalize_subencodings(dict_result::Dict) -> Dict

Normalize the subencodings in the given dictionary per behavior. 

# Arguments
- `dict_result::Dict`: A dictionary containing the sum of subencodings for each behavior and subcategory.

# Returns
- `dict_result_norm::Dict`: A dictionary containing the normalized sum of subencodings for each behavior and subcategory.
"""
function normalize_subencodings(dict_result::Dict)
    dict_result_norm = Dict()
    for beh = keys(dict_result)
        dict_result_norm[beh] = Dict()
        keys_use = [k for k in keys(dict_result[beh]) if !(k in ["unknown_enc", "nonencoding"])]
        for k = keys_use
            dict_result_norm[beh][k] = dict_result[beh][k] / sum([dict_result[beh][k2] for k2 in keys_use])
        end
    end
    return dict_result_norm
end
