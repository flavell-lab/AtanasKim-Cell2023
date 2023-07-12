"""
    merge_confocal_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, dataset::String; traces_key="traces_array", zscored_traces_key="raw_zscored_traces_array", F_F20_key="traces_array_F_F20")

This function merges neuron identities from two datasets from the same animal, `data_dict` and `data_dict_2`, into a new dictionary `combined_data_dict`.
The function is used to combine data from two different datasets.

# Arguments
- `combined_data_dict::Dict`: A dictionary that will contain the merged data.
- `data_dict::Dict`: A dictionary containing the first dataset to be merged.
- `data_dict_2::Dict`: A dictionary containing the second dataset to be merged.
- `dataset::String`: A string representing the name of the second dataset in the first dataset's dictionray.
- `traces_key::String`: A string representing the key for the traces array in the dictionaries. Default is "traces_array".
- `zscored_traces_key::String`: A string representing the key for the zscored traces array in the dictionaries. Default is "raw_zscored_traces_array".
- `F_F20_key::String`: A string representing the key for the F/F0 traces array in the dictionaries. Default is "traces_array_F_F20".

The function returns nothing, but modifies the `combined_data_dict` dictionary in place.

"""
function merge_confocal_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, dataset::String; traces_key="traces_array", zscored_traces_key="raw_zscored_traces_array", F_F20_key="traces_array_F_F20")
    data_dict["roi_match_$dataset"] = zeros(Int32, length(data_dict["valid_rois"]))
    for i in 1:length(data_dict["valid_rois"])
        if isnothing(data_dict["valid_rois"][i])
            continue
        end
        for k in keys(data_dict["matches_to_$dataset"][data_dict["valid_rois"][i]])
            if k in data_dict_2["valid_rois"] && data_dict["matches_to_$dataset"][data_dict["valid_rois"][i]][k] > 400
                data_dict["roi_match_$dataset"][i] = findall(x->x==k, data_dict_2["valid_rois"])[1]
            end
        end
    end

    data_dict["successful_idx_$dataset"] = [i for i in 1:length(data_dict["valid_rois"]) if (data_dict["roi_match_$dataset"][i] != 0)]
    
    max_t_all = size(data_dict[traces_key],2) + size(data_dict_2[traces_key],2)
    combined_data_dict[traces_key] = zeros(length(data_dict["successful_idx_$dataset"]), max_t_all)
    combined_data_dict[traces_key][:,1:data_dict["max_t"]] .= data_dict[traces_key][data_dict["successful_idx_$dataset"],:]
    mean_val_1 = mean(data_dict[traces_key][data_dict["successful_idx_$dataset"],:])
    mean_val_2 = mean(data_dict_2[traces_key][data_dict["roi_match_$dataset"][data_dict["successful_idx_$dataset"]],:])
    combined_data_dict[traces_key][:,data_dict["max_t"]+1:end] .= mean_val_1 / mean_val_2 .* data_dict_2[traces_key][data_dict["roi_match_$dataset"][data_dict["successful_idx_$dataset"]],:];

    combined_data_dict[F_F20_key] = zeros(size(combined_data_dict[traces_key]))
    combined_data_dict[zscored_traces_key] = zeros(size(combined_data_dict[traces_key]))
    for n=1:size(combined_data_dict[traces_key],1)
        combined_data_dict[zscored_traces_key][n,:] .= zscore(combined_data_dict[traces_key][n,:])
        combined_data_dict[F_F20_key][n,:] .= combined_data_dict[traces_key][n,:] ./ percentile(combined_data_dict[traces_key][n,:], 20)
    end
    combined_data_dict["num_neurons"] = size(combined_data_dict[zscored_traces_key], 1)

    combined_data_dict["timestamps"] = deepcopy(data_dict["timestamps"])
    append!(combined_data_dict["timestamps"], data_dict_2["timestamps"])
    combined_data_dict["timestamps"] = combined_data_dict["timestamps"] .- combined_data_dict["timestamps"][1] 
    
    combined_data_dict["nir_timestamps"] = deepcopy(data_dict["nir_timestamps"])
    append!(combined_data_dict["nir_timestamps"], data_dict_2["nir_timestamps"])
    combined_data_dict["nir_timestamps"] = combined_data_dict["nir_timestamps"] .- combined_data_dict["nir_timestamps"][1];

    combined_data_dict["stim_begin_nir"] = deepcopy(data_dict["stim_begin_nir"])
    append!(combined_data_dict["stim_begin_nir"], data_dict_2["stim_begin_nir"])
    combined_data_dict["stim_begin_confocal"] = deepcopy(data_dict["stim_begin_confocal"])
    append!(combined_data_dict["stim_begin_confocal"], data_dict_2["stim_begin_confocal"])

        
    valid_rois_remapped = [data_dict["valid_rois"][x] for x in data_dict["successful_idx_$dataset"]]
    combined_data_dict["valid_rois"] = valid_rois_remapped
    combined_data_dict["inv_map"] = data_dict["inv_map"]


    max_t_all = size(data_dict[zscored_traces_key],2) + size(data_dict_2[zscored_traces_key],2)
    combined_data_dict["max_t"] = max_t_all
    combined_data_dict["max_t_1"] = data_dict["max_t"];
end
