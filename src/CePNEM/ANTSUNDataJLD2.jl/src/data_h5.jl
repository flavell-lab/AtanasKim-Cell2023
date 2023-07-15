zstd = FlavellBase.standardize

"""
    export_jld2_h5(path_data_dict::String; path_h5::Union{String,Nothing}=nothing,
        path_data_dict_match::Union{String,Nothing}=nothing, n_recording::Int=1,
        jld2_dict_key::String="data_dict", velocity_filter=true,
        velocity_filter_threshold=0.2, verbose=false)

Loads JLD2 data and write it to HDF5 file

Arguments
---------
* `path_data_dict`: JLD2 file path to load
* `path_h5`: output file (HDF5) path
* `path_data_dict_match`: data_dict_match for NeuroPAL
* `n_recording`: number of recordings/segments in the file
* `jld2_dict_key`: key of the data in the JLD2 file
* `velocity_filter`: filtering velocity or not
* `velocity_filter_threshold`: velocity filter threshold
* `verbose`: if true, prints progress

## Example  
### Basic (single, continuous recording)  
```julia
export_nd2_h5(path_data_dict; path_h5=path_h5)  
```

### Split (split, 2 recordings)  
```julia
export_nd2_h5(path_data_dict; path_h5=path_h5,
    jld2_dict_key="combined_data_dict", n_recording=2)  
```
"""
function export_jld2_h5(path_data_dict::String; path_h5::Union{String,Nothing}=nothing,
        path_data_dict_match::Union{String,Nothing}=nothing, n_recording::Int=1,
        jld2_dict_key::String="data_dict", velocity_filter=true,
        velocity_filter_threshold=0.2, verbose=false)
    verbose && println("processing: $path_data_dict")
    data_dict = import_jld2_data(path_data_dict, jld2_dict_key);
    idx_splits = get_idx_splits(data_dict, n_recording)

    verbose && println("idx_splits: $idx_splits")
    
    # traces
    trace_F20 = trace_array_rm_nan_neuron(data_dict["traces_array_F_F20"])[1]
    trace_zstd = trace_array_rm_nan_neuron(data_dict["raw_zscored_traces_array"])[1]
    trace, match_org_to_skip, match_skip_to_org = trace_array_rm_nan_neuron(data_dict["traces_array"])
    n_neuron, n_t = size(trace)
    
    verbose && println("timepoints: $n_t n(neuron): $n_neuron")
    
    # velocity
    velocity = impute_list(data_dict["velocity_stage"])

    reversal_vec = Int.(data_dict["rev_times"] .> 0)
    reversal_events = let
        list_start = findall(diff(reversal_vec) .== 1)
        list_end = findall(diff(reversal_vec) .== -1)
        if reversal_vec[1] > 0
            prepend!(list_start, 1)
        end
        hcat(list_start, list_end)
    end
    
    # filter velocity
    velocity_filt = filter_velocity(velocity, velocity_filter_threshold)

    # NeuroPAL
    neuropal_q = false
    if !isnothing(path_data_dict_match) && isfile(path_data_dict_match)
        verbose && println("processing NeuroPAL")
        data_dict_match = import_jld2_data(path_data_dict_match, "data_dict")
        neuropal_q = true
    end
        
    # export hdf5
    verbose && println("writing to HDF5")
    h5open(path_h5, "w") do h5f
        # traces
        g1 = create_group(h5f, "gcamp")
        g1["traces_array_F_F20"] = trace_F20
        g1["trace_array"] = trace_zstd
        g1["trace_array_original"] = trace

        g1["idx_splits"] = Array(hcat(map(x->[x[1], x[end]], idx_splits)...)')
        g1["match_org_to_skip"] = let
                list_match = []
                for (k,v) = match_org_to_skip
                    push!(list_match, [k,v])
                end

                Array(hcat(list_match...)')
            end
        g1["match_skip_to_org"] = let
            list_match = []
            for (k,v) = match_skip_to_org
                push!(list_match, [k,v])
            end

            Array(hcat(list_match...)')
        end

        # behavior
        g2 = create_group(h5f, "behavior")
        g2["velocity"] = velocity_filter ? velocity_filt : velocity
        g2["reversal_vec"] = reversal_vec
        g2["reversal_events"] = reversal_events

        list_key = ["head_angle", "angular_velocity", "pumping", "worm_curvature",
            "body_angle_absolute", "body_angle_all", "body_angle"]
        for k = list_key
            try
                d = data_dict[k]
                g2[k] = isa(d, Array{Any}) ? concrete_type(d) : d
            catch e
                @warn "Error saving $(k): $e"
            end
        end

        # timing
        g3 = create_group(h5f, "timing")
        g3["timestamp_nir"] = data_dict["nir_timestamps"]
        g3["timestamp_confocal"] = data_dict["timestamps"]
        
        # heat stim data
        if haskey(data_dict, "stim_begin_confocal")
            if length(data_dict["stim_begin_confocal"]) >= 1
                d = data_dict["stim_begin_confocal"]
                g3["stim_begin_confocal"] = isa(d, Array{Any}) ? concrete_type(d) : d
            end
        end

        # NeuroPAL
        if neuropal_q
            g4 = create_group(h5f, "neuropal_registration")
            g4["roi_match_confidence"] = data_dict_match["roi_match_confidence"]
            g4["roi_match"] = data_dict_match["roi_matches"]
        end
    end
    
    verbose && println("writing to HDF5 complete")
    
    nothing
end