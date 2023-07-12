function diff_lag(A::AbstractVector, lag::Int)  
    A[lag+1:end] .- A[1:end-lag]
end

function vec_ang(v1, v2)
    acos(clamp(dot(v1, v2) / (norm(v1, 2) * norm(v2, 2)), -1, 1))
end

function read_pos_feature(h5f::HDF5.File)
    pos_feature = read(h5f, "pos_feature")
    pos_feature_unet = copy(pos_feature)
    pos_feature_unet[:,1:2,:] = round.((pos_feature_unet[:,1:2,:] .- 4) ./ 2)
   
    pos_feature, pos_feature_unet
end

function read_pos_feature(path_h5::String)
    h5open(path_h5, "r") do h5f
        read_pos_feature(h5f)
    end
end

function read_stage(path_h5::String)
    h5open(path_h5, "r") do h5f
        pos_stage = read(h5f, "pos_stage")
        return pos_stage
    end
end


function read_h5(path_h5::String)
    h5open(path_h5, "r") do h5f
        pos_feature, pos_feature_unet = read_pos_feature(h5f)
        pos_stage = read(h5f, "pos_stage")
        img_nir = read(h5f, "img_nir")
        
        img_nir, pos_stage, pos_feature, pos_feature_unet
    end
end

"""
Recenters `angle` to be within `pi` of a reference angle `ref` (optional, default 0)
"""
function recenter_angle(angle; ref=0)
    return angle - round((angle-ref)/(2*pi))*2*pi
end

"""
Recenters `angles` to be continuous. `delta` (default 10) is the timespan of reference angles.
"""
function local_recenter_angle(angles; delta=10)
    new_angles = [angles[1]]
    for i=2:length(angles)
        if isnan(angles[i])
            push!(new_angles, NaN)
        else
            ref_angles = new_angles[max(1,i-delta):i-1]
            if all(isnan.(ref_angles))
                push!(new_angles, angles[i])
            else
                push!(new_angles, recenter_angle(angles[i], ref=mean([x for x in ref_angles if !isnan(x)])))
            end
        end
    end
    return new_angles
end

"""
Converts a vector into an angle in the lab `xy`-coordinate space.
"""
function vec_to_angle(vec)
    return recenter_angle.([vec[1,i] < 0 ? atan(vec[2,i]/vec[1,i]) + pi : atan(vec[2,i]/vec[1,i]) for i=1:size(vec,2)])
end

"""
Creates a vector out of `x` and `y` position variables.
"""
function make_vec(x::Array{<:AbstractFloat,1}, y::Array{<:AbstractFloat,1})
    convert(Array{Float64,2}, (hcat(x, y))') # can't do typeof(x) since it's 1D
end

"""
Computes least squares error of a fit.
"""
function get_lsqerr(fit, raw)
    return sum(fit.resid .^ 2) / sum(raw .^ 2)
end

"""
Filters data using the Savitzky-Golay algorithm, either for smoothing or differentiating purposes.

# Arguments:
- `data`: Data to filter.
- `lag`: Interval on each side to use for filtering.
- `is_derivative::Bool` (optional, default `false`): Whether to differentiate the data (vs just smoothing)
- `has_inflection::Bool` (optional, default `true`): Whether the data has inflection points within smoothing interval.
    If set to `true`, apply higher-order smoothing function.
- `is_angle::Bool` (optional, default `false`): Whether the data is an angle (ie -pi=pi)
"""
function savitzky_golay_filter(data, lag; is_derivative::Bool=false, has_inflection::Bool=true, is_angle=false)
    m = 2*lag+1
    if has_inflection
        if is_derivative
            smoothing_array = [15 * (5*(3*m^4-18*m^2+31)*i - 28*(3*m^2-7)*i^3) / (m*(m^2-1)*(3*m^4-39*m^2+108)) for i=(1-m)/2:(m-1)/2]
        else
            smoothing_array = [3/4 * (3*m^2-7-20*i^2) / (m*(m^2-4)) for i=(1-m)/2:(m-1)/2]
        end
    else
        if is_derivative
            smoothing_array = [12 * i / (m*(m^2-1)) for i=(1-m)/2:(m-1)/2]
        else
            smoothing_array = [1 / m for i=(1-m)/2:(m-1)/2]
        end
    end
    smoothed_data = fill(0.0, length(data))
    for i=lag+1:length(data)-lag
        if is_angle
            smoothed_data[i] = sum(recenter_angle.(data[i-lag:i+lag], ref=data[i]) .* smoothing_array)
        else
            smoothed_data[i] = sum(data[i-lag:i+lag] .* smoothing_array)
        end
    end
    return smoothed_data
end

euclidean_dist(x1, x2) = norm(x1 .- x2, 2)
