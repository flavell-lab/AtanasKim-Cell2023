"""
    detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)

Detect the start and stop time of the NIR camera recording.

Arguments
---------
* `di_nir`: digital input from the NIR camera
* `img_id`: image id
* `q_iter_save`: whether the image is saved
* `n_img_nir`: number of NIR images
---------
"""
function detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)
    # behavior camera - FLIR
    list_nir_on = findall(diff(di_nir) .> 1) .+ 1
    list_nir_off = findall(diff(di_nir) .< -1) .+ 1
    nir_record_on = diff(list_nir_on) .> 500
    nir_record_off = diff(list_nir_off) .> 500

    if list_nir_on[1] > 500 # no trigger before the first
        s_nir_start = list_nir_on[1]
    elseif sum(nir_record_on) == 2
        s_nir_start = list_nir_on[findfirst(nir_record_on) + 1]
    else
        error("more than 2 recording on detected for FLIR camera")
    end

    if list_nir_off[end] < length(di_nir) - 500
        s_nir_stop = list_nir_off[end]
    elseif sum(nir_record_off) <= 2
        s_nir_stop = list_nir_off[findlast(diff(list_nir_off) .> 500)] 
    else
        error("more than 2 recording off detected for FLIR camera")
    end

    list_nir_on = filter(x -> s_nir_start - 5 .< x .< s_nir_stop .+ 5, list_nir_on)
    list_nir_off = filter(x -> s_nir_start - 5 .< x .< s_nir_stop .+ 5, list_nir_off)

    if length(list_nir_on) != length(list_nir_off)
        error("length(list_nir_on) != length(list_nir_off)")
    end

    img_id_diff = diff(img_id)
    prepend!(img_id_diff, 1)
    if abs(length(list_nir_on) - sum(diff(img_id))) > 3
        error("the detected trigger count is different from the image id data by more than 3")
    else
        img_id_diff[end] += length(list_nir_on) - sum(diff(img_id)) - 1
    end

    idx_nir_save = Bool[]
    for (Δn, q_save) = zip(img_id_diff, q_iter_save)
        if Δn == 1
            push!(idx_nir_save, q_save)
        else
            push!(idx_nir_save, q_save)
            for i = 1:Δn-1
                push!(idx_nir_save, false)
            end
        end
    end

    if sum(idx_nir_save) != n_img_nir
        error("detected number of NIR frames != saved NIR frames")
    end

    hcat(list_nir_on, list_nir_off)[idx_nir_save,:]
end

function detect_nir_timing(path_h5)
    n_img_nir, daqmx_di, img_metadata = h5open(path_h5, "r") do h5f
        n_img_nir = size(h5f["img_nir"])[3]
        daqmx_di = read(h5f, "daqmx_di")
        img_metadata = read(h5f, "img_metadata")
        n_img_nir, daqmx_di, img_metadata
    end
    di_nir = Float32.(daqmx_di[:,2])
    img_timestamp = img_metadata["img_timestamp"]
    img_id = img_metadata["img_id"]
    q_iter_save = img_metadata["q_iter_save"]

    detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)
end

function detect_confocal_timing(ai_laser)
    ai_laser_bin = Int16.(ai_laser .> mean(ai_laser)) # binarize laser analog signal

    list_confocal_on = findall(diff(ai_laser_bin) .== 1) .+ 1
    list_confocal_off = findall(diff(ai_laser_bin) .== -1) .+ 1

    list_stack_start = list_confocal_on[findall(diff(list_confocal_on) .> 150) .+ 1]
    prepend!(list_stack_start, list_confocal_on[1])
    list_stack_stop = list_confocal_off[findall(diff(list_confocal_off) .> 150)]
    append!(list_stack_stop, list_confocal_off[end])

    if length(list_stack_start) != length(list_stack_stop)
        error("n(stack_off_confocal) != n(stack_on_confocal)")
    end

    list_stack_diff = list_stack_stop .- list_stack_start
    idx_vol = 1:length(list_stack_diff)
    if diff(list_stack_diff)[end] < -3
        idx_vol = 1:(length(list_stack_diff) - 1)
    end

    list_stack_start[idx_vol], list_stack_stop[idx_vol]
end

function filter_ai_laser(ai_laser, di_camera, n_rec=1)
    n_ai, n_di = length(ai_laser), length(di_camera)
    n = min(n_ai, n_di)
    ai_laser_zstack_only = Float32.(ai_laser[1:n])
    ai_laser_filter_bit = zeros(Float32, n)
    trg_state = zeros(Float64, n)

    n_y = n
    n_kernel = 100
    @simd for i = 1:n_y
        start = max(1, i - n_kernel)
        stop = min(n_y, i + n_kernel)

        trg_state[i] = maximum(di_camera[start:stop])
    end

    Δtrg_state = diff(trg_state)    
    list_idx_start = findall(Δtrg_state .== 1)
    list_idx_end = findall(Δtrg_state .== -1)

    if n_rec > length(list_idx_start)
        error("filter_ai_laser: not enough recording detected. check n_rec")
    end
    
    list_idx_rec = sortperm(list_idx_end .- list_idx_start, rev=true)[1:n_rec]
    for i = list_idx_rec
        ai_laser_filter_bit[list_idx_start[i]+1:list_idx_end[i]-1] .= 1
    end
    
    ai_laser_filter_bit .* ai_laser_zstack_only
end

function sync_timing(di_nir, ai_laser, img_id, q_iter_save, n_img_nir)
    timing_stack = hcat(detect_confocal_timing(ai_laser)...)
    timing_nir = detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)

    confocal_to_nir = []
    nir_to_confocal = zeros(size(timing_nir,1))
    for i = 1:size(timing_stack, 1)
        start_, end_ = timing_stack[i,:]

        nir_on_bit = start_ .< timing_nir[:,1] .< end_
        nir_off_bit = start_ .< timing_nir[:,2] .< end_

        idx_ = findall(nir_on_bit .& nir_off_bit)
        push!(confocal_to_nir, idx_)

        nir_to_confocal[idx_] .= i
    end

    confocal_to_nir, nir_to_confocal, timing_stack, timing_nir
end

"""
    sync_timing(path_h5, n_rec=1)

Sync NIR and confocal timing. Returns `(n_img_nir, daqmx_ai, daqmx_di, img_metadata)`

# Arguments
---------
* `path_h5::String`: path to h5 file
* `n_rec::Int`: number of recordings in the file
---------
"""
function sync_timing(path_h5, n_rec=1)
    n_img_nir, daqmx_ai, daqmx_di, img_metadata = h5open(path_h5, "r") do h5f
        n_img_nir = size(h5f["img_nir"])[3]
        daqmx_ai = read(h5f, "daqmx_ai")
        daqmx_di = read(h5f, "daqmx_di")
        img_metadata = read(h5f, "img_metadata")
        n_img_nir, daqmx_ai, daqmx_di, img_metadata
    end

    ai_laser = filter_ai_laser(daqmx_ai[:,1], daqmx_di[:,1], n_rec)
    ai_piezo = daqmx_ai[:,2]
    ai_stim = daqmx_ai[:,3]
    di_confocal = Float32.(daqmx_di[:,1])
    di_nir = Float32.(daqmx_di[:,2])
    img_timestamp = img_metadata["img_timestamp"]
    img_id = img_metadata["img_id"]
    q_iter_save = img_metadata["q_iter_save"]

    sync_timing(di_nir, ai_laser, img_id, q_iter_save, n_img_nir)
end

"""
    sync_stim(stim, timing_stack, timing_nir)

Align stim to confocal and NIR timing. Returns `(stim_to_confocal, stim_to_nir)`

# Arguments
---------
* `stim::Vector`: stim signal
* `timing_stack`: confocal timing
* `timing_nir`: NIR timing
---------
"""
function sync_stim(stim, timing_stack, timing_nir)
    stim_to_confocal = [mean(stim[timing_stack[i,1]:timing_stack[i,2]]) for i = 1:size(timing_stack,1)]
    stim_to_nir = stim[round.(Int, dropdims(mean(timing_nir, dims=2), dims=2))]

    stim_to_confocal, stim_to_nir
end

"""
    signal_stack_repeatability(signal, timing_stack; sampling_rate=5000)

Calculate signal repeatability. Returns `(signal_eta_u, signal_eta_s, list_t)`
e.g. checking z-stack repeatability

# Arguments
---------
* `signal::Vector`: signal
* `timing_stack`: confocal timing
* `sampling_rate::Int`: sampling rate (Hz)
---------
"""
function signal_stack_repeatability(signal, timing_stack; sampling_rate=5000)
    s_stack_start = timing_stack[:,1]
    s_stack_end = timing_stack[:,2]
    n_stack = size(s_stack_end, 1)
    n_stack_len = minimum(s_stack_end - s_stack_start)
    signal_eta = zeros(n_stack, n_stack_len)
    for i = 1:n_stack
        signal_eta[i,:] .= signal[s_stack_start[i]:s_stack_start[i]+n_stack_len-1]
    end

    signal_eta_u = dropdims(mean(signal_eta, dims=1), dims=1)
    signal_eta_s = dropdims(std(signal_eta, dims=1), dims=1)
    list_t = collect(1:n_stack_len) / sampling_rate

    signal_eta_u, signal_eta_s, list_t, n_stack
end

"""
    nir_vec_to_confocal(vec, confocal_to_nir, confocal_len)

Bins NIR behavioral data to match the confocal time points.

# Arguments:
- `vec`: behavioral data vector. Can be 1D or 2D; if 2D time should be on the columns
- `confocal_to_nir`: confocal to NIR time sync
- `confocal_len`: length of confocal dataset
"""
function nir_vec_to_confocal(vec, confocal_to_nir, confocal_len)
    if length(size(vec)) == 1
        new_data = [mean(vec[confocal_to_nir[t]]) for t=1:confocal_len]
    elseif length(size(vec)) == 2
        new_data = zeros(size(vec,1), confocal_len)
        for dim in 1:size(vec,1)
            new_data[dim,:] = nir_vec_to_confocal(vec[dim,:], confocal_to_nir, confocal_len)
        end
    else
        error("Vector dimension cannot be greater than 2.")
    end
    return new_data
end

"""
    unlag_vec(vec, lag)

Shifts a lagged vector `vec` to correct for the lag amount `lag`.
"""
function unlag_vec(vec, lag)
    if length(size(vec)) == 1
        unlagged_vec = fill(NaN, length(vec) + lag)
        unlagged_vec[1+lag÷2:end-lag÷2] = vec
    elseif length(size(vec)) == 2
        unlagged_vec = fill(NaN, size(vec,1), size(vec,2) + lag)
        for dim = 1:size(vec,1)
            unlagged_vec[dim,:] = unlag_vec(vec[dim,:], lag)
        end
    else
        error("Vector dimension cannot be greater than 2.")
    end
    return unlagged_vec     
end

"""
    nir_to_confocal_t(t, nir_to_confocal)

Converts NIR time point `t` to confocal time point using `nir_to_confocal` timesync variable.
"""
function nir_to_confocal_t(t, nir_to_confocal)
    for t_check = t:-1:1
        if nir_to_confocal[t_check] > 0
            return nir_to_confocal[t_check]
        end
    end
    return 1
end

"""
    get_timestamps(path_h5)

Gets NIR timestamps from the NIR data file
"""
function get_timestamps(path_h5)
    f = h5open(path_h5, "r")
    timestamps = f["img_metadata"]["img_timestamp"][:]
    saving = f["img_metadata"]["q_iter_save"][:]
    close(f)
    return timestamps[saving] ./ 1e9
end

"""
    get_timing_info!(data_dict::Dict, param::Dict, path_h5::String, h5_confocal_time_lag::Integer)

Initializes all timing and syncing variables into `data_dict::Dict` given `param::Dict`, `path_h5::String` and the `h5_confocal_time_lag::Integer`
"""
function get_timing_info!(data_dict::Dict, param::Dict, path_h5::String, h5_confocal_time_lag::Integer)
    data_dict["confocal_to_nir"], data_dict["nir_to_confocal"], data_dict["timing_stack"], data_dict["timing_nir"] = sync_timing(path_h5, param["n_rec"]);

    if h5_confocal_time_lag == 0
        data_dict["confocal_to_nir"] = data_dict["confocal_to_nir"][1:param["max_t"]]
        data_dict["timing_stack"] = data_dict["timing_stack"][1:param["max_t"]]
        data_dict["nir_to_confocal"] = [(x > param["max_t"]) ? 0.0 : x for x in data_dict["nir_to_confocal"]]
    else
        data_dict["confocal_to_nir"] = data_dict["confocal_to_nir"][h5_confocal_time_lag+1:end]
        data_dict["timing_stack"] = data_dict["timing_stack"][h5_confocal_time_lag+1:end]
        data_dict["nir_to_confocal"] = [(x < h5_confocal_time_lag + 1) ? 0.0 : x - h5_confocal_time_lag for x in data_dict["nir_to_confocal"]]
    end
    data_dict["nir_timestamps"] = get_timestamps(path_h5)
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["confocal_to_nir"], param["max_t"])

    data_dict["timestamps"] = vec_to_confocal(data_dict["nir_timestamps"]);
    data_dict["max_t_nir"] = length(data_dict["nir_to_confocal"])

    data_dict["avg_timestep"] = (data_dict["timestamps"][end] - data_dict["timestamps"][1]) / length(data_dict["timestamps"])

    data_dict["pre_nir_to_confocal"], data_dict["pre_confocal_to_nir"] = pre_confocal_timesteps(data_dict, param)
    data_dict["max_t"] = param["max_t"]
    data_dict["t_range"] = param["t_range"]

    data_dict["pre_max_t"] = length(data_dict["pre_confocal_to_nir"]);
    data_dict["pre_t_range"] = collect(1:data_dict["pre_max_t"]);


    stim = h5read(path_h5, "daqmx_ai")[:,3]
    timing_nir = BehaviorDataNIR.detect_nir_timing(path_h5)
    thresh = max(maximum(stim)/2, 0.1)
    stim_to_nir = findall(x->x>thresh, stim[round.(Int, dropdims(mean(timing_nir, dims=2), dims=2))])
    prev_len = 20
    stim_begin_nir = [t for t in stim_to_nir if !any([t-i in stim_to_nir for i=1:prev_len])]
    stim_to_confocal = [maximum(data_dict["nir_to_confocal"][1:stim_begin_nir[i]]) for i=1:length(stim_begin_nir)]
    data_dict["stim_begin_nir"] = stim_begin_nir
    data_dict["stim_begin_confocal"] = stim_to_confocal;
end

"""
    fill_timeskip(traces, timestamps; min_timeskip_length=5, timeskip_step=1, fill_val=0)

Fills in timeskips with multiple 0 datapoints for easier visualization.

# Arguments:
- `traces`: Traces matrix with timeskips
- `timestamps`: Timestamps for all data points in the traces matrix
- `min_timeskip_length` (default 5): Minimum difference (in seconds) between adjacent data points to qualify as a timeskip.
- `timeskip_step` (default 1): Number of seconds per intermediate data point generated.
"""
function fill_timeskip(traces, timestamps; min_timeskip_length=5, timeskip_step=1, fill_val=0)
    timeskips = [t for t in 1:length(timestamps)-1 if diff(timestamps)[t] >= min_timeskip_length]
    num_timeskips = length(timeskips)
    new_traces = [[] for n=1:size(traces,1)]
    new_timestamps = []
    prev_timeskip = 1
    for timeskip in timeskips
        num_steps = floor((timestamps[timeskip+1] - timestamps[timeskip]) ÷ timeskip_step)
        for n=1:size(traces,1)
            append!(new_traces[n], traces[n,prev_timeskip:timeskip])
            append!(new_traces[n], [fill_val for t=1:num_steps])
        end
        append!(new_timestamps, timestamps[prev_timeskip:timeskip])
        append!(new_timestamps, [timestamps[timeskip] + t*timeskip_step for t=1:num_steps])
    end
    for n=1:size(traces,1)
        append!(new_traces[n], traces[n,timeskips[end]+1:end])
    end
    append!(new_timestamps, timestamps[timeskips[end]+1:end])

    new_traces_matrix = zeros(size(traces,1), length(new_traces[1]))
    for n=1:size(traces,1)
        new_traces_matrix[n,:] .= new_traces[n]
    end
    return new_timestamps, new_traces_matrix
end        

function fill_timeskip_behavior(behavior, timestamps; min_timeskip_length=5, timeskip_step=1, fill_val=NaN)
    vec = zeros(1, length(behavior))
    vec[1,:] .= behavior
    new_timestamps, new_vec = fill_timeskip(vec, timestamps, min_timeskip_length=min_timeskip_length, timeskip_step=timeskip_step, fill_val=fill_val)
    return new_timestamps, new_vec[1,:]
end

"""
    pre_confocal_timesteps(data_dict::Dict, param::Dict)

Computes confocal timesteps backwards in time from the beginning of the confocal recording.
"""
function pre_confocal_timesteps(data_dict::Dict, param::Dict)
    step = data_dict["avg_timestep"]*param["FLIR_FPS"]
    idx = findall(x->x==1, data_dict["nir_to_confocal"])[1]-1
    pre_nir_to_conf = zeros(idx)
    n_prev = Int(floor(idx/step))
    pre_conf_to_nir = []
    offset = idx - Int(floor(step * n_prev))
    for t=1:n_prev
        rng = Int(floor((t-1)*step+1+offset)):Int(floor(t*step+offset))
        pre_nir_to_conf[rng] .= t
        push!(pre_conf_to_nir, collect(rng))
    end
    return pre_nir_to_conf, pre_conf_to_nir
end
