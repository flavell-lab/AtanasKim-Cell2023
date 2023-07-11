function generate_check_categorization(behavior="v", sub_behavior="fwd")
    function return_f(dict_fit, categorization, idx_neuron, i_rg_t)
        idx_neuron in categorization["$i_rg_t"][behavior][sub_behavior]
    end

    return return_f
end

function get_tau(dict_fit, categorization, idx_neuron, i_rg_t; f_aggregate_posterior::Function=median)
    median(dict_fit["tau"][i_rg_t,idx_neuron,:])
end

function neuropal_aggregate_data(list_dict_fit, list_match_dict, list_use_uid, list_class_dv;
    f_process::Function, f_aggregate_dataset::Function, f_aggregate_rg_t::Union{Nothing,Function}=nothing,
    enc_only=true, merge_rg_t=false)
    dict_data = Dict{String,Vector{Float64}}()

    if merge_rg_t && isnothing(f_aggregate_rg_t)
        error("when `merge_rg_t` is true, a function must be given for `f_aggregate_rg_t`")
    end

    for (idx_uid, (uid, list_rg_t_use)) = enumerate(list_use_uid)
        dict_fit = list_dict_fit[idx_uid]
        match_dict = list_match_dict[idx_uid][2]
        categorization = dict_fit["categorization"]

        for class_dv = list_class_dv
            if haskey(match_dict, class_dv)
                list_match = match_dict[class_dv]

                for match = list_match
                    idx_neuron = match[2]
                    
                    list_val = []
                    for i_rg_t = list_rg_t_use
                        if enc_only && !(idx_neuron ∈ categorization["$i_rg_t"]["all"])
                            # use only the encoding neurons but the neuron is not encoding
                        else
                            push!(list_val, f_process(dict_fit, categorization, idx_neuron, i_rg_t))
                        end
                    end
                    if length(list_val) > 0
                        if merge_rg_t
                            add_list_dict!(dict_data, class_dv, f_aggregate_rg_t(list_val))
                        else
                            for val = list_val
                                add_list_dict!(dict_data, class_dv, val)
                            end
                        end
                    end
                end
            end
        end
    end

    dict_result = Dict{String,Float64}()
    dict_count = Dict{String,Int}()
    for (k,list_v) = dict_data
        dict_result[k] = f_aggregate_dataset(list_v)
        dict_count[k] = length(list_v)
    end

    dict_result, dict_count
end

function neuropal_count(list_dict_fit, list_match_dict, list_use_uid, list_class_dv;
        f_count::Function, f_aggregate_rg_t::Union{Nothing,Function}=any,
        enc_only=false, merge_rg_t=true)
    dict_result = Dict{String,Float64}()
    dict_count = Dict{String,Vector{Bool}}()
    dict_n = Dict{String,Int}()

    if merge_rg_t && isnothing(f_aggregate_rg_t)
        error("when `merge_rg_t` is true, a function must be given for `f_aggregate_rg_t`")
    end

    for (idx_uid, (uid, list_rg_t_use)) = enumerate(list_use_uid)
        dict_fit = list_dict_fit[idx_uid]
        match_dict = list_match_dict[idx_uid][2]
        categorization = dict_fit["categorization"]

        for class_dv = list_class_dv
            if haskey(match_dict, class_dv)
                list_match = match_dict[class_dv]
                for match = list_match
                    idx_neuron = match[2]
                    
                    list_count = Bool[]
                    for i_rg_t = list_rg_t_use
                        if enc_only && !(idx_neuron ∈ categorization["$i_rg_t"]["all"])
                            # use only the encoding neurons but the neuron is not encoding
                        else
                            push!(list_count, f_count(dict_fit, categorization, idx_neuron, i_rg_t))
                        end
                    end
                    
                    if length(list_count) > 0
                        if merge_rg_t
                            add_list_dict!(dict_count, class_dv, f_aggregate_rg_t(list_count))
                            if haskey(dict_n, class_dv)
                                dict_n[class_dv] += 1
                            else
                                dict_n[class_dv] = 1
                            end
                        else
                            for val = list_count
                                add_list_dict!(dict_count, class_dv, val)
                            end
                            
                            if haskey(dict_n, class_dv)
                                dict_n[class_dv] += length(list_count)
                            else
                                dict_n[class_dv] = length(list_count)
                            end
                        end
                    end
                end # for list_match
            end # if haskey match 
        end # for list_class_dv
    end # for list_use_uid

    for (k,v) = dict_count
        dict_result[k] = sum(v) / dict_n[k]
    end

    dict_result, dict_n
end