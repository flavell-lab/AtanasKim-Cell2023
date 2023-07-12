"""
Computes mutual information for every pair of (neuron, behavior).
Returns dictionary and heatmap-compatible array.

# Arguments:
 - `data_dict`: Dictionary containing neuron traces and behavior data.
 - `vars`: Keys in `data_dict` corresponding to behavior data to use.
 - `multivar_idx` (optional, default 1): In the heatmap-compatible array, index to use to represent multi-dimensional data.
"""
function compute_mutual_information_dict(data_dict, vars; multivar_idx=1)
    mutual_info_dict = Dict()
    hmap_mi = zeros(data_dict["num_neurons"],length(vars))
    for (i,key) in enumerate(vars)
        if length(size(data_dict[key])) == 1
            mutual_info_dict[key] = zeros(data_dict["num_neurons"])
            for n=1:data_dict["num_neurons"]
                mutual_info_dict[key][n] = get_mutual_information(data_dict["zscored_traces_array"][n,:], data_dict[key], mode="bayesian_blocks")
                hmap_mi[n,i] = mutual_info_dict[key][n]
            end
        elseif length(size(data_dict[key])) == 2
            mutual_info_dict[key] = zeros(data_dict["num_neurons"], size(data_dict[key],1))
            for n=1:data_dict["num_neurons"]
                for x=1:size(data_dict[key],1)
                    mutual_info_dict[key][n,x] = get_mutual_information(data_dict["zscored_traces_array"][n,:], data_dict[key], mode="bayesian_blocks")
                end
                hmap_mi[n,i] = mutual_info_dict[key][multivar_idx,:]
            end
        else
            error("3-dimensional behavior variables not suppported.")
        end
    end
end
