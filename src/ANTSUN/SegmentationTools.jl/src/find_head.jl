"""
    in_conv_hull(point, centroids, max_d::Real)

Computes whether a point is in the local convex hull of a collection of 2D points.

# Arguments:
- `point`: a point in 2D space.
- `centroids`: a cloud of points in 2D space.
- `max_d::Real`: the farthest a point in `centroids` can be from `point` and still count for the convex hull
"""
function in_conv_hull(point, centroids, max_d::Real)
    # centroids within max_d distance of our point
    centroids_zeroed = filter(x->sum(x.*x)<max_d^2, map(x->x.-point, centroids))
#     println(centroids_zeroed)
    # need three points to form a convex hull
    if length(centroids_zeroed) < 3
        return false
    end
    for pt1 in centroids_zeroed
        for pt2 in centroids_zeroed
            if (pt2 == pt1) || (sum(pt2 .* pt1) > 0)
                continue
            end
            diff2 = ((pt1 .- pt2)[2], -(pt1 .- pt2)[1])
            diff2_dot = sum(pt1 .* diff2)
            for pt3 in centroids_zeroed
                if (pt3 == pt2) || (pt3 == pt1)
                        continue
                    end
                if sum(pt3 .* diff2) * diff2_dot < 0
                    return true
                end
            end
        end
    end
    return false
end

"""
    remove_outlier_neurons(centroids, max_d::Real, threshold::Integer)

Removes neurons from `centroids` in a radius `max_d::Real` around them if there are less than `threshold::Integer` neurons.
"""
function remove_outlier_neurons(centroids, max_d::Real, threshold::Integer)
    return filter(x->length(filter(y->sum((y.-x).*(y.-x)) < max_d^2, centroids)) >= threshold, centroids)
end
    
"""
    local_conv_hull(centroids, max_d::Real, imsize, threshold::Integer)

Computes local convex hull (2D), removing outlier neurons first.
# Arguments
- `centroids`: the locations of the neuron centroids
- `max_d::Real`: maximum distance between neurons to be counted in local convex hull
- `imsize`: the image size in the X-Y plane
- `threshold::Integer`: number of neighboring neurons needed for a point to be counted in the local convex hull
"""
function local_conv_hull(centroids, max_d::Real, imsize, threshold::Integer)
    centroids_2D = remove_outlier_neurons(map(x->x[1:2], centroids), max_d, threshold)
    output = zeros(imsize)
    return map(x->in_conv_hull(Tuple(x),centroids_2D,max_d), CartesianIndices(imsize))
end

""" 
    is_in_focus(centroids, imsize; max_d::Real=50, threshold::Integer=2, z_cutoff::Integer=7)

Detects whether the current frame is in focus, and also outputs z-cropping parameters.
# Arguments
- `centroids`: the locations of the neuron centroids
- `imsize`: the image size in the X-Z plane
## Optional keyword arguments
- `z_cutoff::Integer` (defualt 7): required distance from the Z-end of the frame (in pixels) for a worm to not be too close to the end
- `max_d::Real` (default 50): maximum distance between neurons to be counted in local convex hull
- `threshold::Integer` (default 2): number of neighboring neurons needed for a point to be counted in the local convex hull
"""
function is_in_focus(centroids, imsize; max_d::Real=50, threshold::Integer=2, z_cutoff::Integer=7)
    lch = local_conv_hull(map(x->[x[1], imsize[1]/imsize[2] * x[3]], centroids), max_d, (imsize[1], imsize[1]), threshold);
    lch_nonzero = map(x->collect(Tuple(x)), filter(x->lch[x]!=0, CartesianIndices(lch)))
    z_coords = [lch_nonzero[x][2] for x in 1:length(lch_nonzero)]
    min_z = Int64(floor(minimum(z_coords) * imsize[2]/imsize[1]))
    max_z = Int64(ceil(maximum(z_coords) * imsize[2]/imsize[1]))
    centroid = reduce((x,y)->x.+y, lch_nonzero) ./ length(lch_nonzero)
    worm_z = centroid[2] * imsize[2] / imsize[1]
    if worm_z < z_cutoff || (imsize[2] - worm_z) < z_cutoff
        return (false, (min_z, max_z))
    end
    return (true, (min_z, max_z))
end

"""
    find_head(
        centroids, imsize; tf=[10,10,30], max_d=[30,50,50], hd_threshold::Integer=100, 
        vc_threshold::Integer=300, num_centroids_threshold::Integer=90, edge_threshold::Integer=5, manual_override=false
    )

Finds the tip of the nose of the worm in each time point, and warns of bad time points.
Uses a series of blob-approximations of the worm with different sensitivities, by using local convex hull.
The convex hulls should be set up in increasing order (so the last convex hull is the most generous).
The difference between the first two convex hulls is used to determine the direction of the worm's head.
The third convex hull is used to find the tip of the worm's head.

# Arguments
- `centroids`: the locations of the neuron centroids
- `imsize`: the image size

## Optional keyword arguments
- `tf` (default [10,10,30]): threshold for required neuron density for convex hull `i` is (number of centroids) / `tf[i]`
- `max_d` (default [30,50,50]): the maximum distance for a neuron to be counted as part of convex hull `i` is `max_d[i]`
- `hd_threshold::Integer` (default 100): if convex hulls 2 and 3 give head locations farther apart than this many pixels, set error flag.
- `vc_threshold::Integer` (default 300): if convex hulls 2 and 3 give tail locations farther apart than this many pixels, set error flag.
- `num_centroids_threshold::Integer` (default 90): if there are fewer than this many centroids, set error flag.
- `edge_threshold::Integer` (default 5): if the boundary of the worm is closer than this to the edge of the frame, set error flag.
- `manual_override`: In case the algorithm finds the worm's ventral cord instead of its head, set this variable to a list of all time points where
the algorithm was wrong.

# Outputs a tuple `(head_pos, q_flag, crop_x, crop_y, crop_z, theta, centroid)`
- `head_pos`: position of the worm's head.
- `q_flag`: array of error flags (empty means no errors were detected)
- `crop_x, crop_y, crop_z`: cropping parameters
- `theta`: amount by which to rotate the image to align it in the x-y plane
- `centroid`: centroid of the worm
"""
function find_head(centroids, imsize; tf=[10,10,30], max_d=[30,50,50], hd_threshold::Integer=100, 
        vc_threshold::Integer=300, num_centroids_threshold::Integer=90, edge_threshold::Integer=5, manual_override=false)
    len = length(max_d)
    threshold = [Int64(ceil(length(centroids)/tf[i])) for i in 1:len]
    # Create three local convex hulls with different thresholds
    # Use the first and second to determine which direction the head is in
    # by exploiting asymmetry of neuron distribution
    # Use the third to find the tip of the head (which might be filtered out in the second)
    lch = [local_conv_hull(centroids, max_d[i], imsize[1:2], threshold[i]) for i in 1:len]
    # Get only worm points
    lch_nonzero = [map(x->collect(Tuple(x)), filter(x->lch_i[x]!=0, CartesianIndices(lch_i))) for lch_i in lch]
    # Get center of the worm in each threshold
    worm_centroid = [reduce((x,y)->x.+y, lch_nonzero_i) ./ length(lch_nonzero_i) for lch_nonzero_i in lch_nonzero]
    # Get pca
    deltas = [map(x->collect(x.-worm_centroid[ch]), lch_nonzero[ch]) for ch in 1:len]
    cov_mat = [cov(deltas[i]) for i in 1:len]
    mat_eigvals = [eigvals(cov_mat[i]) for i in 1:len]
    mat_eigvecs = [eigvecs(cov_mat[i]) for i in 1:len]
    eigvals_order = [sortperm(mat_eigvals[i], rev=true) for i in 1:len] # sort order descending

    # get worm axis direction (still from second convex hull)
    long_axis = [mat_eigvecs[i][eigvals_order[i][1],:] for i in 1:len]
    short_axis = [mat_eigvecs[i][eigvals_order[i][2],:] for i in 1:len]
    
    # get sign of difference in centroids - the head is closer to centroid 1
    sign_of_head = sum((worm_centroid[1].-worm_centroid[2]).*long_axis[2]) > 0
    # sort worm points by their worm axis distance
    distances = [map(x->sum(x.*long_axis[2]), deltas[i]) for i in 1:len]
    distances_order = [sortperm(distances[i]) for i in 1:len]
    focus, crop_z = is_in_focus(centroids, [imsize[1], imsize[3]])
    
    q_flag = []
    # Find head according to all three convex hulls, using sign computed above, but always using the second PCA
    if sign_of_head * (!manual_override) > 0
        head = [lch_nonzero[i][distances_order[i][end]] for i in 1:len]
        vc = [lch_nonzero[i][distances_order[i][1]] for i in 1:len]
    else
        head = [lch_nonzero[i][distances_order[i][1]] for i in 1:len]
        vc = [lch_nonzero[i][distances_order[i][end]] for i in 1:len]
    end
    
    diff = (head[2] .- head[3])
    # Hulls 2 and 3 give very different values for head location - check for contamination by tail
    if sqrt(sum(diff .* diff)) > hd_threshold
        push!(q_flag, "TAIL_NEAR_HEAD")
    end
    # The head is very close to the edge of the screen - check that it is not out of the FOV
    if length(filter(x->x<edge_threshold, head[3])) > 0 || length(filter(x->x<edge_threshold, imsize[1:2] .- head[3])) > 0
        push!(q_flag, "HEAD_OUT_OF_VIEW")
    end

    diff = (vc[2] .- vc[3])
    # Hulls 2 and 3 give very different values for ventral cord location - check for contamination by tail
    if sqrt(sum(diff .* diff)) > vc_threshold
        push!(q_flag, "TAIL_NEAR_VC")
    end
    # Part of the central worm is off the screen - check that enough neurons are still present
    if length(filter(x->x<edge_threshold, vc[2])) > 0 || length(filter(x->x<edge_threshold, imsize[1:2] .- vc[2])) > 0
        push!(q_flag, "VC_OUT_OF_VIEW")
    end
    if !focus
        push!(q_flag, "OUT_OF_FOCUS")
    end
    if length(centroids) < num_centroids_threshold
        push!(q_flag, "NOT_ENOUGH_NEURONS")
    end
    # Return head from third (most generous) hull, plus flags, plus cropping parameters
    return (head[3], q_flag)
end

"""
    find_head_unet(param_path, param, dict_param_crop_rot, model, img_size; nrrd_dir="path_dir_nrrd_shearcorrect", crop=true)

Finds the head using the head-detection UNet.
Automatically crops the images to 1:322,1:210, downsamples them by 2x, and takes a maximum-intensity projection.

# Arguments:
- `param_path`: Dictionary containing the keys:
    - `path_dir_nrrd_shearcorrect`, a path to shear-corrected NRRD files
    - `get_basename`, a function that takes time and channel as inputs and outputs NRRD filename
- `param`: Dictionary containing the keys:
    - `t_range`: Time points to compute head
    - `head_threshold`: UNet detection threshold
- `dict_param_crop_rot`: Dictionary of cropping parameters
- `model`: UNet model
- `img_size`: Raw image size.
- `nrrd_dir` (optional, default `path_dir_nrrd_shearcorrect`): Path to NRRD files.
- `crop` (optional, default `true`): Whether to crop the head position.
"""
function find_head_unet(param_path, param, dict_param_crop_rot, model, img_size; nrrd_dir="path_dir_nrrd_shearcorrect", crop=true)
    head_pos = Dict()
    head_errs = Dict()
    @showprogress for t in param["t_range"]
        path_nrrd = joinpath(param_path[nrrd_dir],
            param_path["get_basename"](t,2) * ".nrrd")
        img = maxprj(read_img(NRRD(path_nrrd)), dims=3)
        img_raw = UNet2D.standardize(Float32.(resample_img(img[1:322,1:210], [2,2])))
        img_pred = resample_img(eval_model(img_raw, model), [0.5, 0.5], dtype="weight")

        img_pred_reshape = zeros(img_size)
        for z=1:img_size[3]
            img_pred_reshape[1:322,1:210,z] .= img_pred
        end

        crop_x, crop_y, crop_z = dict_param_crop_rot[t]["crop"]
        θ = dict_param_crop_rot[t]["θ"]
        worm_centroid = dict_param_crop_rot[t]["worm_centroid"]

        if crop
            img_pred_crop = maxprj(crop_rotate(img_pred_reshape, crop_x, crop_y, crop_z,
                    θ, worm_centroid)[1], dims=3)
        else
            img_pred_crop = maxprj(img_pred_reshape, dims=3)
        end
        img_pred_thresh = instance_segmentation(img_pred_crop .> param["head_threshold"],
            min_neuron_size=0)
        img_pred_thresh[img_pred_crop .<= param["head_threshold"]] .= 0

        try
            centroids = get_centroids_round(img_pred_thresh)
            idx = 1
            if length(centroids) > 1
                idx = findmax([sum(img_pred_crop[img_pred_thresh .== i]) for i=1:length(centroids)])[2]
            end

            head_pos[t] = Int64.(collect(centroids[idx]))
        catch e
            head_errs[t] = e
        end
    end
    return (head_pos, head_errs)
end

"""
    find_head(param::Dict, param_path::Dict, t_range, f_basename::Function; manual_override=[])

Finds the tip of the nose of the worm in each time point, and warns of bad time points.
Uses a series of blob-approximations of the worm with different sensitivities, by using local convex hull.
The convex hulls should be set up in increasing order (so the last convex hull is the most generous).
The difference between the first two convex hulls is used to determine the direction of the worm's head.
The third convex hull is used to find the tip of the worm's head.

# Arguments:

- `param_path::Dict`: Dictionary of paths including the keys:
  - `path_head_pos`: Path to head position output file
  - `path_dir_nrrd_crop`: Path to NRRD input files
  - `path_dir_centroid`: Path to centroid input files
- `param::Dict`: Dictionary of parameter settings including the keys:
  - `head_threshold`: threshold for required neuron density for convex hull `i`
    is (number of centroids) / `param["head_threshold"][i]`
  - `head_max_distance`: the maximum distance for a neuron to be counted as part of
    convex hull `i` is `max_d[i]`
  - `head_err_threshold`: if convex hulls 2 and 3 give head locations farther apart than 
    this many pixels, set error flag.
  - `head_vc_err_threshold`: if convex hulls 2 and 3 give tail locations farther apart than 
    this many pixels, set error flag.
  - `head_edge_err_threshold`: if the boundary of the worm is closer than this to the edge of 
    the frame, set error flag.
- `t_range`: The time points to compute head location
- `f_basename::Function`: Function that takes as input a time point and a channel and gives the 
  basename of the corresponding NRRD file.
- `manual_override` (optional): In case the algorithm finds the worm's ventral cord instead of
  its head, set this variable to a list of all time points where
  the algorithm was wrong.
"""
function find_head(param::Dict, param_path::Dict, t_range, f_basename::Function; manual_override=[])
    path_head_pos = param_path["path_head_pos"]
    path_dir_nrrd = param_path["path_dir_nrrd_crop"]
    path_dir_centroid = param_path["path_dir_centroid"]
    
    head_threshold = param["head_threshold"]
    head_max_distance = param["head_max_distance"]
    head_err_threshold = param["head_err_threshold"]
    head_vc_err_threshold = param["head_vc_err_threshold"] 
    head_edge_err_threshold = param["head_edge_err_threshold"]

    dict_qc_flag = Dict{Int,Any}()
    dict_head_pos = Dict{Int,Any}()
    dict_error = Dict{Int,Exception}()
    
    @showprogress for t = t_range
        try
            centroids = read_centroids_roi(joinpath(path_dir_centroid, "$(t).txt"))
            path_nrrd = joinpath(path_dir_nrrd, f_basename(t, param["ch_marker"]) * ".nrrd")
            img = read_img(NRRD(path_nrrd))
            head_pos, qc_flag = find_head(centroids, size(img), tf=head_threshold,
                max_d=head_max_distance, hd_threshold=head_err_threshold,
                vc_threshold=head_vc_err_threshold, edge_threshold=head_edge_err_threshold,
                manual_override=(t in manual_override))
            
            dict_qc_flag[t] = qc_flag
            dict_head_pos[t] = head_pos
        catch e_
            dict_error[t] = e_
        end
    end
        
    str_head_pos = ""
    for t = sort(collect(keys(dict_head_pos)))
        head_pos = dict_head_pos[t]
        str_head_pos *= "$t    $(head_pos[1]) $(head_pos[2])\n"
    end
    write_txt(path_head_pos, str_head_pos[1:end-1])
    
    dict_qc_flag, dict_head_pos, dict_error
end
