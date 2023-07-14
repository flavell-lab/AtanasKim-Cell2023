"""
    zero_stage(x::Array{<:AbstractFloat,2})

Sets the first time point to start at 0, 0

Arguments
---------
* `pos_stage`: (x,y) location, 2 by T array where T is len(time points)
"""
function zero_stage(pos_stage::Array{<:AbstractFloat,2})
    x, y = pos_stage[1,:], pos_stage[2,:]
    @assert(all(.!isnan.([x[1], y[1]])))

    x .- x[1], y .- y[1]
end

function zero_stage(x, y)
    x .- x[1], y .- y[1]
end

"""
    impute_list(x::Array{<:AbstractFloat,1})

Imputes missing data (NaN or missing) with interpolation

Arguments
---------
* `x`: 1D data to impute
"""
function impute_list(x::Array{<:AbstractFloat,1})
    imputed = Impute.locf(Impute.nocb(Impute.interp(replace(x, NaN=>missing))))
    
    convert.(eltype(x), imputed)
end

"""
    speed(Δx::Array{<:AbstractFloat,1}, Δy::Array{<:AbstractFloat,1}, Δt::AbstractFloat)

Computes speed in mm/s 

Arguments
---------
* `Δx`: discrete difference of x
* `Δy`: discrete difference of y
* `Δt`: time interval
"""
function speed(Δx::Array{<:AbstractFloat,1}, Δy::Array{<:AbstractFloat,1}, Δt::AbstractFloat)
    unit_stage_unit_to_mm(sqrt.(Δx .^2 .+ Δy .^2)) / Δt 
end

"""
    speed(x, y; lag::Int, fps=FLIR_FPS)

Computes speed using x, y coordinates

`lag` determines the interval at which to compute the discrete difference

Arguments
---------
* `x`: list of x
* `y`: list of y
* `lag`: number of time points for discerete difference interval
* `fps`: fps
"""
function speed(x, y, lag::Int; fps=FLIR_FPS)
    Δx, Δy = diff_lag.([x, y], lag)
    Δt = lag * 1 / fps
    
    speed(Δx, Δy, Δt)
end

function time_axis(list::AbstractVector, lag=0; fps=FLIR_FPS)
    num_frame = maximum(size(list))
    (2 * collect(1 + lag / 2 : num_frame - lag / 2) .+ lag / 2) / 2 * 1 / fps
end

function Δpos_angle(Δx, Δy)
    atan.(Δy ./ Δx)
end

function Δpos_angle(x, y, lag::Int)
    Δx, Δy = diff_lag.([x, y], lag) # diff_lag in util.jl
    
    Δpos_angle(Δx, Δy)
end

function angular_velocity(Δθ, Δt)
    Δθ / Δt # rad/s
end

function angular_velocity(x, y, lag::Int; fps=FLIR_FPS)
    Δθ = diff(Δpos_angle(x, y, lag))
    Δt = (1 / fps) * lag
    
    angular_velocity(Δθ, Δt) # rad/s
end

function ang_btw_vec(v1::Array{<:AbstractFloat,2}, v2::Array{<:AbstractFloat,2})
    timept = min(size(v1, 2), size(v2, 2))
    vec_ang_lst = zeros(timept)
    
    for i = 1 : timept
        try
            vec_ang_lst[i] = vec_ang(v1[:, i], v2[:, i])
        catch
            @warn "error at index " * string(i) * "// v1: " * string(v1[:, i]) * "// v2: " * string(v2[:, i])
        end
    end
    
    vec_ang_lst
end

# take in 2 x num_vec array as iput and return 1 x num_vec array of magnitudes
function magnitude_vec(list_v::Array{<:AbstractFloat,2})
    sqrt.(list_v[1, :] .^ 2 .+ list_v[2, :] .^ 2)
end

"""
    offset_xy(x_imp, y_imp, x_array, y_array, segment_end_matrix, seg_range; unbin_fn=x->unit_bfs_pix_to_stage_unit(2*x+3))

Corrects x and y position variables to correspond to the worm centroid, rather than the worm pharynx.

# Arguments:

- `x_imp`: Pharynx `x` position
- `y_imp`: Pharynx `y` position
- `x_array`: Segmentation array in the `x`-dimension
- `y_array`: Segmentation array in the `y`-dimension
- `segment_end_matrix`: Matrix of consistently-spaced segmentation locations across time points
- `seg_range`: Set of segmentation locations to use to compute the centroid (which will be the average of them)
- `unbin_fn`: Function to undo binning used to make the segmentation matrix. Default `x->unit_bfs_pix_to_stage_unit(2*x+3)`
"""
function offset_xy(x_imp, y_imp, x_array, y_array, segment_end_matrix, seg_range; unbin_fn=x->unit_bfs_pix_to_stage_unit(2*x+3))
    x_imp_offset = []
    y_imp_offset = []
    err_timepts = []
    last_x_offset = 0
    last_y_offset = 0
    for t=1:length(x_imp)
        if(length(segment_end_matrix[t]) >= maximum(seg_range) + 2)
            last_x_offset = mean([unbin_fn(x_array[t,segment_end_matrix[t][i]]) for i in seg_range])
            last_y_offset = mean([unbin_fn(y_array[t,segment_end_matrix[t][i]]) for i in seg_range])
        else
            push!(err_timepts, t)
        end
        push!(x_imp_offset, x_imp[t] - last_x_offset)
        push!(y_imp_offset, y_imp[t] - last_y_offset)
    end
    return (x_imp_offset, y_imp_offset, err_timepts)
end

function offset_xy(x_imp, y_imp, pos_med; unbin_fn=x->unit_bfs_pix_to_stage_unit(2*x+3))
    x_imp_offset = []
    y_imp_offset = []
    err_timepts = []
    last_x_offset = unbin_fn(pos_med[1,1])
    last_y_offset = unbin_fn(pos_med[2,1])
    for t=1:length(x_imp)
        last_x_offset = unbin_fn(pos_med[1,t]) * pos_med[3,t] + last_x_offset * (1 - pos_med[3,t])
        last_y_offset = unbin_fn(pos_med[2,t]) * pos_med[3,t] + last_y_offset * (1 - pos_med[3,t])
        push!(x_imp_offset, x_imp[t] - last_x_offset)
        push!(y_imp_offset, y_imp[t] - last_y_offset)
    end
    return (x_imp_offset, y_imp_offset)
end


"""
    get_reversal_events(param, velocity, t_range, max_t)

Finds reversal events.

# Arguments:
- `param`: Dictionary containing the following variables:
    - `rev_len_thresh`: Number of consecutive reversal time points necessary for a reversal event
    - `rev_v_thresh`: Velocity threshold below which the worm is counted as reversing
- `velocity`: Worm velocity
- `t_range`: Time range over which to compute reversal events
- `max_t`: Maximum time point
"""
function get_reversal_events(param, velocity, t_range, max_t)
    reversal_events = []
    len_thresh = param["rev_len_thresh"]
    reverse_length = 0
    v_thresh = param["rev_v_thresh"]
    t_inc = t_range
    for t in 1:max_t
        if velocity[t] < v_thresh && (t in t_inc || reverse_length > 0)
            reverse_length += 1
        elseif reverse_length >= len_thresh
            push!(reversal_events, t-reverse_length:t-1)
            reverse_length = 0
        else
            reverse_length = 0
        end
    end
    all_rev = []
    for x in reversal_events
        if mean(velocity[x]) > v_thresh
            continue
        end
        for y in x
            if !(y in all_rev)
                append!(all_rev, y)
            end
        end
    end
    return (reversal_events, all_rev)
end

"""
    compute_reversal_times(reversals, max_t)

Computes the duration of each reversal event.

# Arguments:
- `reversals`: List of all time points where the worm is reversing
- `max_t`: Maximum time point in dataset
"""
function compute_reversal_times(reversals, max_t)
    rev_num = 0
    rev_times = []
    for t=1:max_t
        if t in reversals 
            if rev_num == 0
                non_rev_vals = [T for T in t:max_t if !(T in reversals)]
                if length(non_rev_vals) == 0
                    rev_num = max_t - t
                else
                    rev_num = minimum(non_rev_vals) - t
                end
            end
            append!(rev_times, rev_num)
        else
            append!(rev_times, 0)
            rev_num = 0
        end
    end
    return rev_times
end
