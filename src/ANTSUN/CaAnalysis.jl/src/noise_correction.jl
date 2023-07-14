"""
    divide_by_marker_signal(traces::Dict, marker_traces::Dict)

Divides the activity traces `traces::Dict` by the marker signal `marker_traces::Dict`
"""
function divide_by_marker_signal(traces::Dict, marker_traces::Dict)
    new_traces = Dict()
    for neuron in keys(traces)
        new_traces[neuron] = Dict()
        for frame in keys(traces[neuron])
            if traces[neuron][frame] > 0 && marker_traces[neuron][frame] > 0
                new_traces[neuron][frame] = traces[neuron][frame] / marker_traces[neuron][frame]
            end
        end
    end
    return new_traces
end


"""
    get_background(timepts, get_basename::Function, nrrd_path::String, ch::Int)

Gets the background camera intensity from the NRRD files to be subtracted.

# Arguments
- `timepts`: Time points to get background of
- `get_basename::Function`: Function that outputs the name of the NRRD file from the time and channel
- `nrrd_path::String`: Path to NRRD files
- `ch::Int`: Channel
"""
function get_background(timepts, get_basename::Function, nrrd_path::String, ch::Int)
    frame_bkg = Dict()
    @showprogress for t in timepts
        frame_bkg[t] = median(read_img(NRRD(joinpath(nrrd_path, get_basename(t, ch))*".nrrd")))
    end
    return frame_bkg
end

"""
    bkg_subtract(traces::Dict, frame_bkg)

Subtracts from `traces` the corresponding background `frame_bkg`.
"""
function bkg_subtract(traces::Dict, frame_bkg)
    new_traces = Dict()
    warning = false
    for neuron in keys(traces)
        new_traces[neuron] = Dict()
        for t in keys(traces[neuron])
            new_traces[neuron][t] = traces[neuron][t] - frame_bkg[t]
            if new_traces[neuron][t] < 0
                warning = true
                new_traces[neuron][t] = 0
            end
        end
    end
    if warning
        @warn "Some neurons had lower intensity than the background."
    end
    return new_traces
end

"""
    normalize_traces(traces; zero::Bool=false, fn::Function=mean)

Normalizes traces.

# Arguments
- `traces::Dict`: Traces to normalize

# Optional keyword arguments
- `zero::Bool`: Whether to normalize to zero (so activities that are less than average would be negative). Default false.
- `fn::Function`: Function that determines average neuron activity
"""
function normalize_traces(traces; zero::Bool=false, fn::Function=mean)
    new_traces = zeros(size(traces))
    for neuron in 1:size(traces,1)
        new_traces[neuron,:] .= traces[neuron,:] ./ fn(traces[neuron,:]) .- 1 .* zero
    end
    return new_traces
end

"""
    interpolate_traces(traces::Dict, t_range; itp_method=Linear(), extrap_method=Interpolations.Flat(), fill_val=NaN)

Interpolates data at missing time points.

# Arguments
 - `traces::Dict`: Dictionary containing for each neuron a set of time points and activity values at those time points
 - `t_range`: Time points to interpolate to. Time points outside this range will be set to `fill_val`
 - `itp_method` (optional, default `Linear()`): Interpolation method.
 - `extrap_method` (optional, default `Interpolations.Flat()`): Extrapolation method (for interpolating data points outside all time points the neuron was detected in)
 - `fill_val` (optional, default `NaN`): Value to fill time points outside the interpolation range.
"""
function interpolate_traces(traces::Dict, t_range; itp_method=Linear(), extrap_method=Interpolations.Flat(), fill_val=NaN)
    new_traces = Dict()
    max_t = maximum(t_range)
    for neuron in keys(traces)
        t_vals = sort(collect(keys(traces[neuron])))
        activity_vals = [traces[neuron][t] for t in t_vals]
        itp = extrapolate(interpolate((t_vals,), activity_vals, Gridded(itp_method)), extrap_method)
        new_traces[neuron] = [(t in t_range) ? Float64(itp(t), RoundNearest) : fill_val for t in 1:max_t]
    end
    return new_traces
end

"""
    deconvolve_traces(traces, k::Real, max_t::Integer)

Deconvolves traces to correct for GCaMP decay with respect to confocal volume time, `k`
"""
function deconvolve_traces(traces, k::Real, max_t::Integer)
    g(t,k) = 2^(-t*k)*(1-2^(-k))
    deconvolved_traces = zeros(size(traces))
    for n=1:size(traces,1)
        deconvolved_traces[n,:] .= real.(ifft(fft(traces[n,:])./fft([g(t-1,k) for t=1:max_t])))
    end
    return deconvolved_traces
end

"""
    zscore_traces(traces)

Z-scores traces.
"""
function zscore_traces(traces)
    new_traces = zeros(size(traces))
    for neuron in 1:size(traces,1)
        m = mean(traces[neuron,:])
        s = std(traces[neuron,:])
        new_traces[neuron, :] .= (traces[neuron,:] .- m) ./ s
    end
    return new_traces
end

"""
    get_laser_intensity(percent_on, laser)

Interpolate laser intensity from percentage on, and from measurements.
"""
function get_laser_intensity(percent_on, laser)
    for i=2:size(laser,1)
        if laser[i,1] > percent_on
            frac_low = (laser[i,1] - percent_on) / (laser[i,1] - laser[i-1,1])
            return frac_low * laser[i-1,2] + (1 - frac_low) * laser[i,2]
        end
    end
end

"""
    get_laser_intensity(percent_on, intensity, laser_perc; zero_thresh=4.5, min_laser=5.2, max_laser=20.0, max_interpolate=2.0)

Interpolate laser intensity from `percent_on`, and from `intensity` measurments taken at `laser_perc` values.

# Arguments:
- `percent_on`: Laser percentage used in data acquisition.
- `intensity`: Measured calibration intensities.
- `laser_perc`: Measured laser percentages corresponding to those intensities.
- `zero_thresh` (optional, default 4.5): Maximum percentage to use for 0 calibration
- `min_laser` (optional, default 5.2): Minimum percentage to use for linear calibration
- `max_interpolate` (optional, default 2.0): Maximum laser percentage difference to use for linear interpolation
"""
function get_laser_intensity(percent_on, intensity, laser_perc; zero_thresh=4.5, min_laser=5.2, max_laser=20.0, max_interpolate=2.0)
    @assert(max_laser - min_laser >= max_interpolate, "Cannot interpolate over invalid laser intensities.")
    @assert(percent_on >= min_laser)
    zero_max = argmax(1.0 ./ (laser_perc .- zero_thresh))
    baseline = median(intensity[1:zero_max])

    min_x = max(min_laser, percent_on-max_interpolate/2.0)
    max_x = min(min_x + max_interpolate, max_laser)
    if max_x == max_laser
        min_x = max_x - max_interpolate
    end

    min_l = argmax(1.0 ./ (laser_perc .- min_x))        
    max_l = argmax(1.0 ./ (laser_perc .- max_x))


    a, b = [ones(length(laser_perc[min_l:max_l]), 1) laser_perc[min_l:max_l]] \ intensity[min_l:max_l]
    return a + b * percent_on - baseline
end

"""
    process_traces(
        param::Dict, activity_traces::Dict, marker_traces::Dict, threshold::Real, t_range; 
        activity_bkg=nothing, marker_bkg=nothing, min_intensity::Real=0, interpolate::Bool=false, 
        denoise::Bool=false, bleach_corr::Bool=false, divide::Bool=false, normalize_fn::Function=x->quantile(x,0.2),
        k::Union{Real,Nothing}=nothing, valid_rois=nothing
    )

Applies multiple data processing steps to the traces. The order of processing steps is:

- Background-subtraction
- Delete low S/N neurons
- Delete neurons detected in too few time points
- Interpolate traces at missing data points
- Denoise
- Bleach-correct
- Divide activity by marker channel
- Normalize
- Deconvolve
- Zscore

# Arguments
- `param::Dict`: Parameter dictionary containing laser information.
- `activity_traces::Dict`: traces in the activity channel
- `marker_traces::Dict`: traces in the marker channel
- `threshold::Real`: Number of timepoint detections necessary to include a neuron in the analysis
- `t_range`: Time points to extract traces over.

# Keyword arguments
- `activity_bkg`: Background in activity channel. If left blank, background will not be subtracted.
- `marker_bkg`: Background in marker channel. If left blank, background will not be subtracted.
- `min_intensity::Real`: Minimum average intensity in the activity channel for a neuron (after background subtraction).
    Neurons with less than this much signal will be removed. Default 0. 
- `interpolate::Bool`: Whether to interpolate missing data points within the time range.
- `denoise::Bool`: Whether to apply a total variation denoising step.
- `bleach_corr::Bool`: Whether to bleach-correct the traces.
- `divide::Bool`: Whether to divide the activity channel traces by the marker channel traces.
- `normalize_fn::Function`: Function to use to get "average" activity when normalizing traces.
- `k::Union{Real,Nothing}`: Deconvolution parameter. Set this to (time length of confocal volume) / (GCaMP decay half-life)
- `valid_rois`: List of ROIs to include in the analysis. If left blank, all ROIs will be included.
"""
function process_traces(param::Dict, activity_traces::Dict, marker_traces::Dict, threshold::Real, t_range; activity_bkg=nothing, marker_bkg=nothing,
        min_intensity::Real=0, interpolate::Bool=false, denoise::Bool=false, bleach_corr::Bool=false, divide::Bool=false, normalize_fn::Function=x->quantile(x,0.2),
        k::Union{Real,Nothing}=nothing, valid_rois=nothing)

    activity_traces = copy(activity_traces)
    marker_traces = copy(marker_traces)
    # background subtract activity
    if !isnothing(activity_bkg)
        activity_traces = bkg_subtract(activity_traces, activity_bkg)
    end
    # background subtract marker
    if !isnothing(marker_bkg)
        marker_traces = bkg_subtract(marker_traces, marker_bkg)
    end

    # delete neurons with too low S/N
    neuron_rois = [roi for roi in keys(activity_traces) if
        length([activity_traces[roi][t] for t in keys(activity_traces[roi]) if t in t_range]) > 0 &&
            mean([activity_traces[roi][t] for t in keys(activity_traces[roi]) if t in t_range]) > min_intensity]

    for roi in keys(activity_traces)
        if !(roi in neuron_rois) || length([t for t in keys(activity_traces[roi]) if t in t_range]) < threshold
            delete!(activity_traces, roi)
        end
    end
    for roi in keys(marker_traces)
        if !(roi in neuron_rois) || length([t for t in keys(marker_traces[roi]) if t in t_range]) < threshold
            delete!(marker_traces, roi)
        end
    end

    blue_1 = get_laser_intensity(param["blue_laser"][1], param["blue_laser_intensity"], param["blue_laser_perc"], zero_thresh=param["blue_zero_thresh"], min_laser=param["blue_min_laser"], max_laser=maximum(param["blue_laser_perc"]), max_interpolate=param["blue_max_interpolate"])
    blue_2 = blue_1
    # correct for different laser intensities
    if length(param["blue_laser"]) > 1
        @assert(length(param["blue_laser"]) == 2, "More than 1 laser intensity switch not supported.")
        blue_2 = get_laser_intensity(param["blue_laser"][2], param["blue_laser_intensity"], param["blue_laser_perc"], zero_thresh=param["blue_zero_thresh"], min_laser=param["blue_min_laser"], max_laser=maximum(param["blue_laser_perc"]), max_interpolate=param["blue_max_interpolate"])
        ratio_blue = blue_1 / blue_2
        for roi in keys(activity_traces)
            for t in keys(activity_traces[roi])
                if t > param["max_graph_num"]
                    activity_traces[roi][t] = ratio_blue * activity_traces[roi][t]
                end
            end
        end
    end
    
    if length(param["green_laser"]) > 1
        @assert(length(param["green_laser"]) == 2, "More than 1 laser intensity switch not supported.")
        intensity_g = h5read("/data1/shared/2022-05-11-laser-power.h5", "561nm/1/intensity")
        laser_perc_g = h5read("/data1/shared/2022-05-11-laser-power.h5", "561nm/1/laser_percent")
        ratio_1 = get_laser_intensity(param["green_laser"][1], param["green_laser_intensity"], param["green_laser_perc"], zero_thresh=param["green_zero_thresh"], min_laser=param["green_min_laser"], max_laser=maximum(param["green_laser_perc"]), max_interpolate=param["green_max_interpolate"])
        ratio_2 = get_laser_intensity(param["green_laser"][2], param["green_laser_intensity"], param["green_laser_perc"], zero_thresh=param["green_zero_thresh"], min_laser=param["green_min_laser"], max_laser=maximum(param["green_laser_perc"]), max_interpolate=param["green_max_interpolate"])
        ratio = ratio_1 / ratio_2
        for roi in keys(marker_traces)
            for t in keys(marker_traces[roi])
                if t > param["max_graph_num"]
                    marker_traces[roi][t] = ratio * marker_traces[roi][t]
                end
            end
        end
    end


    # interpolate traces
    if interpolate
        activity_traces = interpolate_traces(activity_traces, t_range)
        marker_traces = interpolate_traces(marker_traces, t_range)
    end

    # divide activity by marker
    if divide
        activity_traces = divide_by_marker_signal(activity_traces, marker_traces)
    end

    traces_arr, hmap, valid_rois = make_traces_array(activity_traces, threshold=threshold, replace_blank=true, valid_rois=valid_rois)

    processed_traces_arr = traces_arr

    data_dict = Dict()
    data_dict["f"] = traces_arr
    data_dict["idx_ok"] = 1:size(data_dict["f"])[1]

    # denoise
    if denoise
        denoiser = DenoiserGSTV(100, 10.)
        denoise!(data_dict, denoiser.f)
        processed_traces_arr = data_dict["f_denoised"]
    else
        data_dict["f_denoised"] = data_dict["f"]
    end

    # bleach-correct
    if bleach_corr
        fit_bleach!(data_dict, true, true, idx_t=t_range)
        processed_traces_arr[:,t_range] .= data_dict["f_bleach"]
    end

    # deconvolve
    if !isnothing(k)
        @assert(interpolate, "Cannot deconvolve non-interpolated traces")
        @assert(collect(t_range) == collect(1:maximum(t_range)), "Cannot deconvolve incomplete traces")
        processed_traces_arr = deconvolve_traces(processed_traces_arr, k, maximum(t_range))
    end

    traces_normalized = normalize_traces(processed_traces_arr, fn=normalize_fn)
    traces_zscored = zscore_traces(processed_traces_arr)

    if bleach_corr
        return processed_traces_arr, traces_normalized, traces_zscored, valid_rois, data_dict["bleach_param"], data_dict["bleach_curve"], data_dict["bleach_resid"]
    else
        return processed_traces_arr, traces_normalized, traces_zscored, valid_rois
    end
end


"""
    get_all_values(processed_traces)

Gets all values from a set of processed traces. Ensure that the traces have been normalized prior to using this function!
"""
function get_all_values(processed_traces)
    trace_vals = []
    for neuron in keys(processed_traces)
        append!(trace_vals, collect(values(processed_traces[neuron])))
    end
    return trace_vals
end
