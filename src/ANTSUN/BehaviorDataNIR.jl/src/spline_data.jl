"""
Gets the vector the worm is facing, in the lab `xy`-coordinate system.

# Arguments:
- `x_array`: Segmentation array in the `x`-dimension
- `y_array`: Segmentation array in the `y`-dimension
- `segment_end_matrix`: Matrix of consistently-spaced segmentation locations across time points
- `seg_range`: Set of segmentation locations to use to compute the centroid (which will be the average of them)
"""
function get_worm_vector(x_array, y_array, segment_end_matrix, seg_range)
    vec = zeros(2,size(x_array,1))
    for t=1:size(x_array,1)
        if(length(segment_end_matrix[t]) >= maximum(seg_range) + 2)
            vec[1,t] = x_array[t,segment_end_matrix[t][maximum(seg_range)]]-x_array[t,segment_end_matrix[t][minimum(seg_range)]]
            vec[2,t] = y_array[t,segment_end_matrix[t][maximum(seg_range)]]-y_array[t,segment_end_matrix[t][minimum(seg_range)]]
        end
    end
    return vec
end

"""
Gets the body angle of a worm between the given points in the spline.

# Arguments:
- `x_array`: Segmentation array in the `x`-dimension
- `y_array`: Segmentation array in the `y`-dimension
- `segment_end_matrix`: Matrix of consistently-spaced segmentation locations across time points
- `pts`: A list of three points in the spline to get the body angle between.
"""
function get_worm_body_angle(x_array, y_array, segment_end_matrix, pts)
    vec = [zeros(2,size(x_array,1)), zeros(2,size(x_array,1))]
    for t=1:size(x_array,1)
        if(length(segment_end_matrix[t]) >= maximum(pts) + 2)
            for i=1:2
                vec[i][1,t] = x_array[t,segment_end_matrix[t][pts[i+1]]]-x_array[t,segment_end_matrix[t][pts[i]]]
                vec[i][2,t] = y_array[t,segment_end_matrix[t][pts[i+1]]]-y_array[t,segment_end_matrix[t][pts[i]]]
            end
        end
    end
    return recenter_angle.(vec_to_angle(vec[2]) .- vec_to_angle(vec[1]))
end

"""
Computes total worm curvature

# Arguments:
- `body_angle`: Array of worm body angles
- `min_len`: If there aren't this many angles at a given time point, interpolate that time point instead of computing it
- `directional::Bool` (default `false`): Use directional curvature.
"""
function get_tot_worm_curvature(body_angle, min_len; directional::Bool=false)
    worm_curvature = zeros(size(body_angle,2))
    for t=1:size(body_angle,2)
        all_angles = [body_angle[i,t] for i=1:size(body_angle,1) if !isnan(body_angle[i,t])]
        if length(all_angles) < min_len
            worm_curvature[t] = NaN
        else
            all_angles = local_recenter_angle(all_angles)
            if directional
                worm_curvature[t] = (all_angles[1]-all_angles[min_len])/length(all_angles)
            else
                worm_curvature[t] = std(all_angles)
            end
        end
    end
    return impute_list(worm_curvature)
end



""" Gets the smallest ratio between distance between two points in space and distance between them along the worm's curvature.
A value of 0 means that the worm is intersecting itself, while a value of 1 means the worm is a straight line.

# Arguments:
- `spline_x`: x positions in worm spline
- `spline_y`: y positions in worm spline
- `segment_end`: equally-spaced segments along worm spline
- `max_i` (default `1`): Maximum location along the medium axis to try. For nose curling, use 1.
"""
function nose_curling(spline_x, spline_y, segment_end; max_i=1)
    ratio = Inf
    for i=1:length(segment_end)
        if i > max_i
            return 1/ratio
        end
        pos1 = segment_end[i]
        for j=i+1:length(segment_end)
            pos2 = segment_end[j]
            ratio = min(ratio, euclidean_dist((spline_x[pos1], spline_y[pos1]), (spline_x[pos2], spline_y[pos2]))/(j-i))
        end
    end
    return 1/ratio
end
