"""
interpolate_splines!(data_dict)

Interpolates worm splines to time points where the spline computation crashed.
"""
function interpolate_splines!(data_dict)
    bad_timepts = [t for t=1:data_dict["max_t_nir"] if sum(abs.(data_dict["x_array"][t,:])) == 0 && sum(abs.(data_dict["y_array"][t,:])) == 0]
    data_dict["x_array"][bad_timepts,:] .= NaN
    data_dict["y_array"][bad_timepts,:] .= NaN
    for idx=1:size(data_dict["x_array"],2)
        data_dict["x_array"][:,idx] .= impute_list(data_dict["x_array"][:,idx])
        data_dict["y_array"][:,idx] .= impute_list(data_dict["y_array"][:,idx])
    end
end


"""
get_body_angles!(data_dict::Dict, param::Dict; prefix::String="")

Adds body angles to `data_dict` given `param`.
Can add a `prefix` (default empty string) to all confocal variables.
"""
function get_body_angles!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    s = size(data_dict["x_array"],1)
    m = maximum([length(x) for x in data_dict["segment_end_matrix"]])-1
    conf_len = length(data_dict["$(prefix)confocal_to_nir"])
    data_dict["nir_body_angle"] = zeros(param["max_pt"]-1,s)
    data_dict["nir_body_angle_all"] = zeros(m,s)
    data_dict["nir_body_angle_absolute"] = zeros(m,s)
    data_dict["$(prefix)body_angle"] = zeros(param["max_pt"]-1,conf_len)
    data_dict["$(prefix)body_angle_all"] = zeros(m,conf_len)
    data_dict["$(prefix)body_angle_absolute"] = zeros(m,conf_len)
    for pos in 1:m
        for t in 1:s
            if length(data_dict["segment_end_matrix"][t]) > pos
                Δx = data_dict["x_array"][t,data_dict["segment_end_matrix"][t][pos+1]] - data_dict["x_array"][t,data_dict["segment_end_matrix"][t][pos]]
                Δy = data_dict["y_array"][t,data_dict["segment_end_matrix"][t][pos+1]] - data_dict["y_array"][t,data_dict["segment_end_matrix"][t][pos]]
                data_dict["nir_body_angle_absolute"][pos,t] = recenter_angle(vec_to_angle([Δx, Δy])[1])
                data_dict["nir_body_angle_all"][pos,t] = recenter_angle(vec_to_angle([Δx, Δy])[1] - data_dict["nir_worm_angle"][t])
            else
                data_dict["nir_body_angle_absolute"][pos,t]= NaN
                data_dict["nir_body_angle_all"][pos,t]= NaN
            end
        end
        data_dict["nir_body_angle_absolute"][pos,:] .= local_recenter_angle(data_dict["nir_body_angle_absolute"][pos,:], delta=param["body_angle_t_lag"])
    end
    
    for t in 1:s
        data_dict["nir_body_angle_absolute"][:,t] .= local_recenter_angle(data_dict["nir_body_angle_absolute"][:,t], delta=param["body_angle_pos_lag"])
        data_dict["nir_body_angle_all"][:,t] .= local_recenter_angle(data_dict["nir_body_angle_all"][:,t], delta=param["body_angle_pos_lag"])
    end
    
    for pos in 1:m
        data_dict["$(prefix)body_angle_all"][pos,:] .= vec_to_confocal(data_dict["nir_body_angle_all"][pos,:])
        data_dict["$(prefix)body_angle_absolute"][pos,:] .= vec_to_confocal(data_dict["nir_body_angle_absolute"][pos,:])
        
        if pos < param["max_pt"]
            data_dict["nir_body_angle"][pos,:] .= data_dict["nir_body_angle_all"][pos,:]      
            data_dict["nir_body_angle"][pos,:] .= impute_list(data_dict["nir_body_angle"][pos,:])
            data_dict["$(prefix)body_angle"][pos,:] .= vec_to_confocal(data_dict["nir_body_angle"][pos,:])
            data_dict["$(prefix)body_angle_absolute"][pos,:] .= vec_to_confocal(data_dict["nir_body_angle_absolute"][pos,:]);
        end
    end
end

"""
    get_angular_velocity!(data_dict::Dict, param::Dict; prefix::String="")

Gets angular velocity from `data_dict` and `param`.
Can add a `prefix` (default empty string) to all confocal variables.
"""
function get_angular_velocity!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["$(prefix)worm_angle"] = vec_to_confocal(data_dict["nir_worm_angle"])

    nir_turning_angle = impute_list(data_dict["nir_body_angle_absolute"][param["head_pts"][1],:])
    data_dict["nir_angular_velocity"] = savitzky_golay_filter(nir_turning_angle, param["filt_len_angvel"], is_derivative=true, has_inflection=false) .* param["FLIR_FPS"]
    data_dict["$(prefix)angular_velocity"] = vec_to_confocal(data_dict["nir_angular_velocity"])
end

"""
    get_velocity!(data_dict::Dict, param::Dict; prefix::String="")

Gets velocity, speed, reversal, and related variables from `data_dict` and `param`.
Can add a `prefix` (default empty string) to all confocal variables.
"""
function get_velocity!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["filt_xmed"] = gstv(Float64.(data_dict["x_med"]), param["v_stage_m_filt"], param["v_stage_λ_filt"])
    data_dict["filt_ymed"] = gstv(Float64.(data_dict["y_med"]), param["v_stage_m_filt"], param["v_stage_λ_filt"]);

    Δx = [0.0]
    Δy = [0.0]
    append!(Δx, diff(data_dict["filt_xmed"]))
    append!(Δy, diff(data_dict["filt_ymed"]))
    Δt = 1.0 / param["FLIR_FPS"]
    data_dict["nir_mov_vec_stage"] = make_vec(Δx, Δy)
    data_dict["$(prefix)mov_vec_stage"] = vec_to_confocal(data_dict["nir_mov_vec_stage"])
    data_dict["nir_mov_angle_stage"] = impute_list(vec_to_angle(data_dict["nir_mov_vec_stage"]))
    data_dict["$(prefix)mov_angle_stage"] = vec_to_confocal(data_dict["nir_mov_angle_stage"])
    data_dict["nir_speed_stage"] = speed(Δx, Δy, Δt)
    data_dict["$(prefix)speed_stage"] = vec_to_confocal(data_dict["nir_speed_stage"])
    data_dict["nir_velocity_stage"] = data_dict["nir_speed_stage"] .* cos.(data_dict["nir_mov_angle_stage"] .- data_dict["pm_angle"])
    data_dict["$(prefix)velocity_stage"] = vec_to_confocal(data_dict["nir_velocity_stage"])
    data_dict["$(prefix)reversal_events"], data_dict["$(prefix)all_rev"] = get_reversal_events(param, data_dict["$(prefix)velocity_stage"], data_dict["$(prefix)t_range"], data_dict["$(prefix)max_t"]);
    data_dict["$(prefix)rev_times"] = compute_reversal_times(data_dict["$(prefix)all_rev"], data_dict["$(prefix)max_t"]);
end

"""
    get_curvature_variables!(data_dict::Dict, param::Dict; prefix::String="")

Gets curvature, head angle, and related variables from `data_dict` and `param`.
Can add a `prefix` (default empty string) to all confocal variables.
"""
function get_curvature_variables!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["$(prefix)worm_curvature"] = get_tot_worm_curvature(data_dict["$(prefix)body_angle"], size(data_dict["$(prefix)body_angle"],1));
    data_dict["$(prefix)ventral_worm_curvature"] = get_tot_worm_curvature(data_dict["$(prefix)body_angle"], size(data_dict["$(prefix)body_angle"],1), directional=true);
    data_dict["nir_head_angle"] = -impute_list(get_worm_body_angle(data_dict["x_array"], data_dict["y_array"], data_dict["segment_end_matrix"], param["head_pts"]))
    data_dict["nir_nose_angle"] = -impute_list(get_worm_body_angle(data_dict["x_array"], data_dict["y_array"], data_dict["segment_end_matrix"], param["nose_pts"]))

    data_dict["$(prefix)head_angle"] = vec_to_confocal(data_dict["nir_head_angle"])
    data_dict["$(prefix)nose_angle"] = vec_to_confocal(data_dict["nir_nose_angle"]);
end

"""
    get_nose_curling!(data_dict::Dict, param::Dict; prefix::String="")

Gets self intersection variables from `data_dict` and `param`.
Can add a `prefix` (default empty string) to all confocal variables.
"""
function get_nose_curling!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["nir_nose_curling"] = Vector{Float64}()
    for t=1:data_dict["max_t_nir"]
        if length(data_dict["segment_end_matrix"][t]) < param["max_pt"]
            push!(data_dict["nir_nose_curling"], NaN)
        else
            push!(data_dict["nir_nose_curling"], nose_curling(data_dict["x_array"][t,:], data_dict["y_array"][t,:], data_dict["segment_end_matrix"][t][1:param["max_pt"]], max_i=1))
        end
    end
    data_dict["nir_nose_curling"] = impute_list(data_dict["nir_nose_curling"])
    data_dict["$(prefix)nose_curling"] = vec_to_confocal(data_dict["nir_nose_curling"])
end

"""
    merge_nir_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, param::Dict)

Modifies `combined_data_dict` to add merged NIR data from two datasets `data_dict` and `data_dict_2`,
together with a parameter file `param` for the primary dataset.
"""
function merge_nir_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, param::Dict)
    combined_data_dict["confocal_to_nir_1"] = data_dict["confocal_to_nir"]
    combined_data_dict["confocal_to_nir_2"] = data_dict_2["confocal_to_nir"]
    combined_data_dict["nir_to_confocal_1"] = data_dict["nir_to_confocal"]
    combined_data_dict["nir_to_confocal_2"] = data_dict_2["nir_to_confocal"]

    combined_data_dict["pre_confocal_to_nir_1"] = data_dict["pre_confocal_to_nir"]
    combined_data_dict["pre_confocal_to_nir_2"] = data_dict_2["pre_confocal_to_nir"]
    combined_data_dict["pre_nir_to_confocal_1"] = data_dict["pre_nir_to_confocal"]
    combined_data_dict["pre_nir_to_confocal_2"] = data_dict_2["pre_nir_to_confocal"]

    combined_data_dict["max_t_1"] = data_dict["max_t"]
    combined_data_dict["max_t_2"] = data_dict_2["max_t"]
    combined_data_dict["t_range_1"] = data_dict["t_range"]
    combined_data_dict["t_range_2"] = data_dict_2["t_range"]
    combined_data_dict["pre_max_t_1"] = data_dict["pre_max_t"]
    combined_data_dict["pre_max_t_2"] = data_dict_2["pre_max_t"]
    combined_data_dict["pre_t_range_1"] = data_dict["pre_t_range"]
    combined_data_dict["pre_t_range_2"] = data_dict_2["pre_t_range"]
    combined_data_dict["max_t"] = combined_data_dict["max_t_1"] + combined_data_dict["max_t_2"]


    combined_data_dict["max_t_nir_1"] = data_dict["max_t_nir"]
    combined_data_dict["max_t_nir_2"] = data_dict_2["max_t_nir"]


    for var in param["concat_vars"]
        if length(size(data_dict[var])) == 1
            combined_data_dict[var] = zeros(combined_data_dict["max_t"])
            combined_data_dict[var][1:length(data_dict[var])] .= data_dict[var]
            combined_data_dict[var][length(data_dict[var])+1:end] .= data_dict_2[var]
        elseif length(size(data_dict[var])) == 2
            max_size = max(size(data_dict[var],1), size(data_dict_2[var],1))
            combined_data_dict[var] = zeros(max_size, combined_data_dict["max_t"])
            combined_data_dict[var][1:size(data_dict[var],1),1:size(data_dict[var],2)] .= data_dict[var][:,:]
            combined_data_dict[var][1:size(data_dict_2[var],1),size(data_dict[var],2)+1:end] .= data_dict_2[var][:,:]
        else
            throw(ErrorException("number of dimensions must be 1 or 2"))
        end
        if "pre_$(var)" in keys(data_dict)
            combined_data_dict["pre_$(var)_1"] = data_dict["pre_$(var)"]
        end
        if "pre_$(var)" in keys(data_dict_2)
            combined_data_dict["pre_$(var)_2"] = data_dict_2["pre_$(var)"]
        end
    end
    for var in param["t_concat_vars"]
        combined_data_dict[var] = deepcopy(data_dict[var])
        append!(combined_data_dict[var], combined_data_dict["max_t_1"] .+ data_dict_2[var])
        if "pre_$(var)" in keys(data_dict)
            combined_data_dict["pre_$(var)_1"] = data_dict["pre_$(var)"]
        end
        if "pre_$(var)" in keys(data_dict_2)
            combined_data_dict["pre_$(var)_2"] = data_dict_2["pre_$(var)"]
        end
    end
    for var in param["nir_concat_vars"]
        if length(size(data_dict[var])) == 1
            tot_len = length(data_dict[var]) + length(data_dict_2[var])
            combined_data_dict[var] = zeros(tot_len)
            combined_data_dict[var][1:length(data_dict[var])] .= data_dict[var]
            combined_data_dict[var][length(data_dict[var])+1:end] .= data_dict_2[var]
        elseif length(size(data_dict[var])) == 2
            max_size = max(size(data_dict[var],1), size(data_dict_2[var],1))
            tot_len = size(data_dict[var],2) + size(data_dict_2[var],2)
            combined_data_dict[var] = zeros(max_size, tot_len)
            combined_data_dict[var][1:size(data_dict[var],1),1:size(data_dict[var],2)] .= data_dict[var][:,:]
            combined_data_dict[var][1:size(data_dict_2[var],1),size(data_dict[var],2)+1:end] .= data_dict_2[var][:,:]
        else
            throw(ErrorException("number of dimensions must be 1 or 2"))
        end
    end
end

"""
    import_pumping!(combined_data_dict::Dict, param::Dict, paths_pumping; prefix::String="", is_split=false)

Import pumping data into a combined dataset given `param` from csv files.
Can add a `prefix` (default empty string) to all confocal variables.
If `prefix` is added, will output a list of pumping for each dataset rather than merging them together.
"""
function import_pumping!(combined_data_dict::Dict, param::Dict, paths_pumping; prefix::String="", is_split=false)
    combined_data_dict["nir_pumping_raw"] = Float64[]
    combined_data_dict["nir_pumping"] = Float64[]
    combined_data_dict["$(prefix)pumping_raw"] = []
    combined_data_dict["$(prefix)pumping"] = []

    dataset_combine_fn! = isempty(prefix) ? append! : push!

    for (d, file) in enumerate(paths_pumping)
        postfix = is_split ? "_$d" : ""
        pumping = readdlm(file, ',', Any, '\n')
        pumping_nir_raw = param["FLIR_FPS"] .* [(t in floor.(Int64.(pumping[2:end,2])./50)) ? 1 : 0 for t in 1:combined_data_dict["max_t_nir$(postfix)"]]
        append!(combined_data_dict["nir_pumping_raw"], pumping_nir_raw)
        pumping_raw = nir_vec_to_confocal(pumping_nir_raw, combined_data_dict["$(prefix)confocal_to_nir$(postfix)"], length(combined_data_dict["$(prefix)confocal_to_nir$(postfix)"]))
        
        pumping_nir_filt = savitzky_golay_filter(pumping_nir_raw, param["filt_len_pumping"], is_derivative=false, has_inflection=false)
        pumping_conf = nir_vec_to_confocal(pumping_nir_filt, combined_data_dict["$(prefix)confocal_to_nir$(postfix)"], length(combined_data_dict["$(prefix)confocal_to_nir$(postfix)"]))
        append!(combined_data_dict["nir_pumping"], pumping_nir_filt)
        
        pumping_conf = nir_vec_to_confocal(pumping_nir_filt, combined_data_dict["$(prefix)confocal_to_nir$(postfix)"], length(combined_data_dict["$(prefix)confocal_to_nir$(postfix)"]))

        dataset_combine_fn!(combined_data_dict["$(prefix)pumping_raw"], pumping_raw)
        dataset_combine_fn!(combined_data_dict["$(prefix)pumping"], pumping_conf)

        if length(paths_pumping) == 1 && is_split
            dataset_combine_fn!(combined_data_dict["$(prefix)pumping_raw"], nir_vec_to_confocal(pumping_nir_raw, combined_data_dict["$(prefix)confocal_to_nir_2"], length(combined_data_dict["$(prefix)confocal_to_nir_2"])))
            dataset_combine_fn!(combined_data_dict["$(prefix)pumping"], nir_vec_to_confocal(pumping_nir_filt, combined_data_dict["$(prefix)confocal_to_nir_2"], length(combined_data_dict["$(prefix)confocal_to_nir_2"])))
        end
    end

    if !isempty(prefix)
        combined_data_dict["$(prefix)pumping_raw_1"], combined_data_dict["$(prefix)pumping_raw_2"] = combined_data_dict["$(prefix)pumping_raw"]
        delete!(combined_data_dict, "$(prefix)_pumping_raw")
        combined_data_dict["$(prefix)pumping_1"], combined_data_dict["$(prefix)pumping_2"] = combined_data_dict["$(prefix)pumping"]
        delete!(combined_data_dict, "$(prefix)pumping")
    end
end
