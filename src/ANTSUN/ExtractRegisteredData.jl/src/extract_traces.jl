"""
    extract_traces(inverted_map, gcamp_data_dir)

Extracts traces from neuron labels.

# Arguments

- `inverted_map`: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label
- `gcamp_data_dir`: Directory containing GCaMP activity data
"""
function extract_traces(inverted_map, gcamp_data_dir)
    traces = Dict()
    errors = Dict()
    for roi in keys(inverted_map)
        traces[roi] = Dict()
        for t in keys(inverted_map[roi])
            try
                if length(inverted_map[roi][t]) == 1
                    activity = read_activity(joinpath(gcamp_data_dir, "$(t).txt"))
                    idx = inverted_map[roi][t][1]
                    # Neuron out of FOV of green camera
                    if idx > length(activity)
                        traces[roi][t] = 0
                    else
                        traces[roi][t] = activity[idx]
                    end
                end
            catch e
                if !(roi in keys(errors))
                    errors[roi] = Dict()
                end
                errors[roi][t] = e
            end
        end
    end
    if length(keys(errors)) > 0
        @warn "Unable to extract traces in all time points"
    end
    return (traces, errors)
end

"""
    error_rate(traces; gfp_thresh::Real=1.5, num_thresh::Real=30)

Computes the error rate of the traces, assuming they are all supposed to have constant activity.

# Arguments

- `traces`: the traces

# Optional keyword arguments
- `gfp_thresh::Real`: threshold amount for a neuron to be counted as having GFP (error rate is adjusted to account for mismatched neurons that have similar activity).
    Assumption is that GFP-negative traces have value 0. If a trace ever differs from its median by more than this, it is counted as a mis-registration
- `num_thresh::Real`: minimum amount of neurons needed to consider a trace
"""
function error_rate(traces; gfp_thresh::Real=1.5, num_thresh::Real=30)
    tot = 0
    error = 0
    gfp = 0
    good_rois = 0
    err_rois = Dict()
    extracted_timepts = Dict()
    err_timepts = Dict()
    for (roi, trace) in traces
        vals = [v for v in values(trace) if (v == v)]
        if length(trace) < num_thresh
            continue
        end
        good_rois += 1
        m = median(vals)
        if m > gfp_thresh
            gfp += 1
        end
        err_rois[roi] = [t for t in keys(trace) if (trace[t] == trace[t]) && ((abs(trace[t]-m) > gfp_thresh)) && ((trace[t] > gfp_thresh) ⊻ (m > gfp_thresh))]
        for t in keys(trace)
            if !(t in keys(extracted_timepts))
                extracted_timepts[t] = []
                err_timepts[t] = []
            end
            append!(extracted_timepts[t], roi)
            if t in err_rois[roi]
                append!(err_timepts[t], roi)
            end
        end
        error += length(err_rois[roi])
        tot += length([v for v in vals])
    end
    gfp_frac = gfp/good_rois
    wrong_prob = 2 * gfp_frac * (1 - gfp_frac)
    return (err_rois, error/(tot*wrong_prob), gfp/good_rois, wrong_prob, extracted_timepts, err_timepts)
end

"""
    make_traces_array(traces::Dict; threshold::Real=1, valid_rois=nothing, contrast::Real=1, replace_blank::Bool=false, cluster::Bool=false)

Turns a traces dictionary into a traces array.
Also outputs a heatmap of the array, and the labels of which neurons correspond to which rows.

# Arguments

 - `traces::Dict`: Dictionary of traces.
 - `threshold::Real` (optional): Minimum number of time points necessary to include a neuron in the traces array.
        Default 1 (all ROIs displayed). It is recommended to set either this or the `valid_rois` parameter.
 - `valid_rois` (optional): If set, use this set of neurons (in order) in the array. Set entries to `nothing` to insert gaps in the data
        (eg: to preserve the previous labeling of neurons)
 - `contrast::Real` (optional): If set, increases the contrast of the heatmap. Does not change the output array.
 - `replace_blank::Bool` (optional): If set, replaces all blank entries in the traces (where the neuron was not found in that time point)
        with the median activity for that neuron.
 - `cluster::Bool` (optional): If set to `true`, cluster the neurons by activity similarity.
"""
function make_traces_array(traces::Dict; threshold::Real=1, valid_rois=nothing, contrast::Real=1, replace_blank::Bool=false, cluster::Bool=false)
    if isnothing(valid_rois)
        valid_rois = [roi for roi in keys(traces) if length(keys(traces[roi])) >= threshold]
    end
    t_max = maximum([isnothing(roi) ? 0 : maximum(keys(traces[roi])) for roi in valid_rois])
    traces_arr = zeros((length(valid_rois), t_max))
    count = 1
    for roi in valid_rois
        if isnothing(roi)
            traces_arr[count,:] .= NaN
            count += 1
            continue
        end
        med = median([x for x in values(traces[roi]) if x == x])
        for t = 1:t_max
            if t in keys(traces[roi])
                traces_arr[count,t] = traces[roi][t]
            elseif replace_blank
                traces_arr[count,t] = med
            else
                traces_arr[count,t] = 0
            end
        end
        count += 1
    end

    if cluster
        @assert(!any(isnothing.(valid_rois)), "Cannot cluster null ROIs.")
        dist = pairwise_dist(traces_arr)
        cluster = hclust(dist, linkage=:single)
        ordered_traces_arr = zeros(size(traces_arr))
        ordered_valid_rois = []
        for i=1:length(cluster.order)
            append!(ordered_valid_rois, valid_rois[cluster.order[i]])
            for j=1:t_max
                ordered_traces_arr[i,j] = traces_arr[cluster.order[i],j]
            end
        end
        traces_arr = ordered_traces_arr
        valid_rois = ordered_valid_rois
    end

    max_val = maximum(traces_arr)
    new_traces_arr = map(x->min(x,max_val/contrast), traces_arr)
    return (traces_arr, heatmap(new_traces_arr), valid_rois)
end


"""
    extract_activity_am_reg(
        param_path::Dict, param::Dict, shear_params_dict::Dict, crop_params_dict::Dict;
        nrrd_path_key::String="path_dir_nrrd", roi_dir_key::String="path_dir_roi_watershed", 
        transform_key::String="name_transform_activity_marker_avg"
    )

Extracts activity marker activity from camera-alignment registration. Returns any errors encountered.

# Arguments
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `param::Dict`: Directory of imaging parameters.
 - `shear_params_dict::Dict`: Dictionary of shear-correction parameters used in the marker channel.
 - `crop_params_dict::Dict`: Dictionary of cropping parameters used in the marker channel.
 - `nrrd_path_key::String` (optional): Key in `param_path` to path to unprocessed activity-channel NRRD files. Default `path_dir_nrrd`
 - `roi_dir_key::String` (optional): Key in `param_path` to directory of neuron ROI files. Default `path_dir_roi_watershed`.
 - `transform_key::String` (optional): Key in `param_path` to file name of transform parameter files. Default `path_dir_transformed_activity_marker_avg`.
"""
function extract_activity_am_reg(param_path::Dict, param::Dict, shear_params_dict::Dict, crop_params_dict::Dict;
        nrrd_path_key::String="path_dir_nrrd", roi_dir_key::String="path_dir_roi_watershed", transform_key::String="name_transform_activity_marker_avg")
    errors = Dict()
    create_dir(param_path["path_dir_transformed_activity_marker"])
    get_basename = param_path["get_basename"]
    nrrd_path = param_path[nrrd_path_key]
    ch_activity = param["ch_activity"]
    t_range = param["t_range"]
    println("Transforming activity channel data...")
    @showprogress for t in t_range
        try
            nrrd_str = joinpath(param_path["path_root_process"], nrrd_path, get_basename(t, ch_activity)*".nrrd")
            regpath = joinpath(param_path["path_dir_reg_activity_marker"], "$(t)to$(t)")
            create_dir(joinpath(param_path["path_dir_transformed_activity_marker"], "$(t)"))
            run_transformix_img(param_path["path_dir_reg_activity_marker"], 
                joinpath(param_path["path_dir_transformed_activity_marker"], "$(t)"), nrrd_str,
                joinpath(regpath, param_path[transform_key]), 
                joinpath(regpath, param_path["name_transform_activity_marker_roi"]), param_path["path_transformix"])
            
            mv(joinpath(param_path["path_dir_transformed_activity_marker"], "$(t)", "result.nrrd"),
                    joinpath(param_path["path_dir_transformed_activity_marker"], get_basename(t, ch_activity)*".nrrd"), force=true)
        catch e
            errors[t] = e
        end
    end
    param["t_range"] = [t for t in param["t_range"] if !(t in keys(errors))]
    t_range = param["t_range"]
    println("Shear-correcting activity channel data...")
    shear_correction_nrrd!(param_path, param, ch_activity, shear_params_dict, nrrd_in_key="path_dir_transformed_activity_marker")
    println("Cropping activity channel data...")
    crop_errors, out_of_focus_frames = crop_rotate!(param_path, param, t_range, [ch_activity], crop_params_dict)
    for t in keys(crop_errors)
        errors[t] = crop_errors[t]
    end

    println("Extracting activity...")
    @showprogress for t in t_range
        img = read_img(NRRD(joinpath(param_path["path_dir_nrrd_crop"], get_basename(t, ch_activity)*".nrrd")))
        img_roi = read_img(NRRD(joinpath(param_path[roi_dir_key], "$(t).nrrd")))

        # get activity
        activity = get_activity(img_roi, img)
        activity_file = joinpath(param_path["path_dir_activity_signal"], "$(t).txt") 
        write_activity(activity, activity_file)
    end
    if length(keys(errors)) > 0
        @warn "Unsuccessful camera alignment registration at some time points"
    end
    return errors
end

"""
    extract_roi_overlap(
        best_reg::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", reg_problems_key::String="path_reg_prob",
        param_path_moving::Union{Dict,Nothing}=nothing
    )

Extracts ROI overlaps and activity differences.

# Arguments
 - `best_reg::Dict`: Dictionary of best registration values.
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `param::Dict`: Dictionary of parameter values.
 - `reg_dir_key::String` (optional): Key in `param_path` corresponding to the registration directory. Default `path_dir_reg`.
 - `transformed_dir_key::String` (optional): Key in `param_path` corresponding to the transformed ROI save directory. Default `path_dir_transformed`.
 - `reg_problems_key::String` (optional): Key in `param_path` corresponding to the registration problems file. Default `path_reg_prob`.
 - `param_path_moving::Union{Dict, Nothing}`: If set, the `param_path` dictionary corresponding to the moving dataset. Otherwise, the method will assume
the moving and fixed datasets have the same path dictionary.
"""
function extract_roi_overlap(best_reg::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", reg_problems_key::String="path_reg_prob",
        param_path_moving::Union{Dict,Nothing}=nothing)

    problems = load_registration_problems([param_path[reg_problems_key]])

    roi_overlaps = Vector{Dict}(undef, length(problems))
    roi_activity_diff = Vector{Dict}(undef, length(problems))
    errors = Vector{Exception}(undef, length(problems))

    if isnothing(param_path_moving)
        param_path_moving = param_path
    end

    @showprogress for i in 1:length(problems)
        (moving, fixed) = problems[i]
        try
            dir = "$(moving)to$(fixed)"
            # Bspline registration failed
            if best_reg[(moving, fixed)][1] == 0
                continue
            end
            best = best_reg[(moving, fixed)]
            tf_base = joinpath(param_path[reg_dir_key], "$(dir)/TransformParameters.$(best[1]).R$(best[2])")
            img, result = run_transformix_roi(joinpath(param_path[reg_dir_key], "$(dir)"), 
                joinpath(param_path_moving["path_dir_roi_watershed"], "$(moving).nrrd"),  joinpath(param_path[transformed_dir_key], "$(dir)"), 
                "$(tf_base).txt", "$(tf_base)_roi.txt", param_path["path_transformix"])
            roi = read_img(NRRD(joinpath(param_path["path_dir_roi_watershed"], "$(fixed).nrrd")))
            roi_regmap = delete_smeared_neurons(img, param["smeared_neuron_threshold"])

            roi_overlap, roi_activity = register_neurons_overlap(roi_regmap, roi, 
                read_activity(joinpath(param_path_moving["path_dir_marker_signal"], "$(moving).txt")), 
                read_activity(joinpath(param_path["path_dir_marker_signal"], "$(fixed).txt")))
            roi_overlaps[i] = roi_overlap
            roi_activity_diff[i] = roi_activity
        catch e
            errors[i] = e
        end
    end

    roi_overlaps_dict = Dict()
    roi_activity_diff_dict = Dict()
    errors_dict = Dict()
    for (i, problem) in enumerate(problems)
        if isassigned(roi_overlaps, i)
            roi_overlaps_dict[problem] = roi_overlaps[i]
        end
        if isassigned(roi_activity_diff, i)
            roi_activity_diff_dict[problem] = roi_activity_diff[i]
        end
        if isassigned(errors, i)
            errors_dict[problem] = errors[i]
        end
    end
    if length(keys(errors_dict)) > 0
        @warn "Registration issues at some time points"
    end
    return roi_overlaps_dict, roi_activity_diff_dict, errors_dict
end



"""
    output_roi_candidates(traces, inv_map::Dict, param_path::Dict, param::Dict, get_basename::Function, channel::Integer, t_range, valid_rois)

Outputs neuron ROI candidates and a plot of their activity.

# Arguments
 - `traces`: Array of traces for all ROIs
 - `inv_map::Dict`: Dictionary that maps ROI identity to time points and UNet ROI labels
 - `param_path::Dict`: Dictionary of paths to relevant files
 - `param::Dict`: Dictionary of parameter values
 - `get_basename::Function`: Function that gets the basename of an NRRD file
 - `channel::Integer`: Channel of the image to be displayed
 - `t_range`: All time points
"""
function output_roi_candidates(traces, inv_map::Dict, param_path::Dict, param::Dict, get_basename::Function, channel::Integer, t_range, valid_rois)
    @showprogress for (idx, neuron) in enumerate(valid_rois)
        min_t = minimum(keys(inv_map[neuron]))
        img = maxprj(Float64.(read_img(NRRD(joinpath(param_path["path_dir_nrrd_filt"], get_basename(min_t, channel)*".nrrd")))), dims=3);
        
        centroids = read_centroids_roi(joinpath(param_path["path_dir_centroid"], "$(min_t).txt"))
        roi = centroids[inv_map[neuron][min_t][1]][1:2]
        
        fig, axes = PyPlot.subplots(ncols=2, figsize=(12,6));

        neuron_traces = traces[idx,t_range]
        axes[1].imshow(img);
        axes[1].scatter([roi[2]], [roi[1]], c="r", s=4);
        axes[2].plot(t_range, neuron_traces);
        axes[2].set_ylim(minimum(neuron_traces), maximum(neuron_traces));
        axes[2].set_xlim(minimum(t_range), maximum(t_range));
        PyPlot.savefig(joinpath(param_path["path_roi_candidates"], string(idx)*".png"));
        PyPlot.close(fig)
     end
end