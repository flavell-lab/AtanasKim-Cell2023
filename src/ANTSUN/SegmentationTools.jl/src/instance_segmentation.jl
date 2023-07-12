"""
    instance_segmentation_watershed(
        param::Dict, param_path::Dict, path_dir_nrrd::String, t_range,
        f_basename::Function; save_centroid::Bool=false, save_signal::Bool=false, save_roi::Bool=false
    )

Runs watershed instance segmentation on all given frames and can output to various files (centroids, activity measurements, and image ROIs).
Skips a given output method if the corresponding output directory was empty.
Returns dictionary of results and a list of error frames (most likely because the worm was not in the field of view).

# Arguments
- `param::Dict`: Dictionary containing parameters, including:
   - `seg_threshold_unet`: Confidence threshold of the UNet output for a pixel to be counted as a neuron.
   - `seg_min_neuron_size`: Minimum neuron size, in voxels
   - `seg_threshold_watershed`: Confidence thresholds of the UNet output for a pixel to be counted as a neuron in each watershed step
   - `seg_watershed_min_neuron_sizes`: Minimum neuron sizes, in voxels, in each watershed step
 - `param_path::Dict`: Dictionary containing paths to directories and a `get_basename` function that returns NRRD file names, including:
    - `path_dir_unet_data`: Path to UNet output data (input to the watershed algorithm)
    - `path_dir_roi`: Path to non-watershed ROI ouptut data
    - `path_dir_roi_watershed`: Path to watershed ROI output data
    - `path_dir_marker_signal`: Path to marker channel signal output data
    - `path_dir_centroid`: Path to centroid output data
 - `path_dir_nrrd::String`: Path to NRRD files
 - `t_range`: Time points to watershed
 - `f_basename::Function`: Function that returns the name of NRRD files
 - `save_centroid::Bool` (optional): Whether to save centroids. Default `false`
 - `save_signal::Bool` (optional): Whether to save marker signal. Default `false`
 - `save_roi::Bool` (optional): Whether to save ROIs before watershedding. Default `false`
"""
function instance_segmentation_watershed(param::Dict, param_path::Dict, path_dir_nrrd::String, t_range,
        f_basename::Function; save_centroid::Bool=false, save_signal::Bool=false, save_roi::Bool=false)

    path_dir_unet_data = param_path["path_dir_unet_data"]    
    path_dir_roi = param_path["path_dir_roi"]
    path_dir_roi_watershed = param_path["path_dir_roi_watershed"]
    path_dir_activity = param_path["path_dir_marker_signal"]
    path_dir_centroid = param_path["path_dir_centroid"]
    threshold_unet = param["seg_threshold_unet"]
    min_neuron_size = param["seg_min_neuron_size"]

    save_centroid && create_dir(path_dir_centroid)
    save_signal && create_dir(path_dir_activity)
    save_roi && create_dir(path_dir_roi_watershed)
    save_roi && create_dir(path_dir_roi)

    watershed_thresholds = param["seg_threshold_watershed"]
    watershed_min_neuron_sizes = param["seg_watershed_min_neuron_sizes"]

    errors = Vector{Exception}(undef, param["max_t"])

    batch_size = Threads.nthreads() * param["instance_seg_num_batches"]

    @showprogress for batch = 1:Int(ceil(length(t_range) / batch_size))
        imgs = Vector{Array{UInt16, 3}}(undef, batch_size)
        imgs_pred = Vector{Array{Float32, 3}}(undef, batch_size)
        imgs_pred_thresh = Vector{BitArray{3}}(undef, batch_size)
        imgs_roi = Vector{Array{Int16, 3}}(undef, batch_size)
        imgs_roi_watershed = Vector{Array{Int16, 3}}(undef, batch_size)
        activities = Vector{Vector{Float64}}(undef, batch_size)
        centroids = Vector{Vector{Tuple{Float64, Float64, Float64}}}(undef, batch_size)
        
        batch_idx_rng = 1:min(batch_size, length(t_range) - (batch - 1) * batch_size)
        for batch_idx = batch_idx_rng
            t = t_range[(batch - 1) * batch_size + batch_idx]
            path_pred = joinpath(path_dir_unet_data, "$(t)_predictions.h5")
            imgs_pred[batch_idx] = load_predictions(path_pred)
            imgs_pred_thresh[batch_idx] = imgs_pred[batch_idx] .> threshold_unet
            if save_signal || save_roi
                path_nrrd = joinpath(path_dir_nrrd, f_basename(t, param["ch_marker"]) * ".nrrd")
                nrrd = NRRD(path_nrrd)
                imgs[batch_idx] = read_img(NRRD(path_nrrd))
            end
        end
        for batch_idx = batch_idx_rng
            t = t_range[(batch - 1) * batch_size + batch_idx]
            try
                imgs_roi[batch_idx] = instance_segmentation(imgs_pred_thresh[batch_idx], min_neuron_size=min_neuron_size)
                imgs_roi_watershed[batch_idx] = instance_segmentation_threshold(imgs_roi[batch_idx], imgs_pred[batch_idx],
                    thresholds=watershed_thresholds, neuron_sizes=watershed_min_neuron_sizes)
                activities[batch_idx] = Float64.(get_activity(imgs_roi_watershed[batch_idx], imgs[batch_idx]))
                centroids[batch_idx] = Tuple{Float64, Float64, Float64}.(get_centroids_round(imgs_roi_watershed[batch_idx]))
            catch e
                errors[t] = e
            end
        end
        for batch_idx = batch_idx_rng
            t = t_range[(batch - 1) * batch_size + batch_idx]
            if isassigned(errors, t)
                continue
            end
            path_nrrd = joinpath(path_dir_nrrd, f_basename(t, param["ch_marker"]) * ".nrrd")
            nrrd = NRRD(path_nrrd)
            if save_centroid
                path_centroid = joinpath(path_dir_centroid, "$(t).txt")
                write_centroids(centroids[batch_idx], path_centroid)
            end

            if save_signal
                path_activity = joinpath(path_dir_activity, "$(t).txt")
                write_activity(activities[batch_idx], path_activity)
            end

            if save_roi
                # save image before watershedding
                path_roi = joinpath(path_dir_roi, "$(t).nrrd")
                write_nrrd(path_roi, map(x->UInt16(x), imgs_roi[batch_idx]),
                    spacing(nrrd))
                
                # save image after watershedding
                path_roi_watershed = joinpath(path_dir_roi_watershed, "$(t).nrrd")
                write_nrrd(path_roi_watershed, map(x->UInt16(x),
                        imgs_roi_watershed[batch_idx]), spacing(nrrd))
            end
        end
    end

    dict_error = Dict{Int, Exception}()
    for t = t_range
        if isassigned(errors, t)
            dict_error[t] = errors[t]
        end
    end

    return dict_error
end

"""
    volume(radius, sampling_ratio)

Computes volume from a radius and a sampling ratio
"""
volume(radius, sampling_ratio) = (4 / 3) * π * prod(radius .* sampling_ratio)


"""
    instance_segmentation(predictions; min_neuron_size::Integer=7)

Runs instance segmentation on a frame. Removes detected objects that are too small to be neurons.

# Arguments

- `predictions: UNet predictions array

# Optional keyword arguments

- `min_neuron_size::Integer`: smallest neuron size, in voxels. Default 7.
"""
function instance_segmentation(predictions; min_neuron_size::Integer=7)
    return consolidate_labeled_img(labels_map(fast_scanning(predictions, 0.5)), min_neuron_size);
end


""" 
    consolidate_labeled_img(labeled_img, min_neuron_size)

Converts an instance-segmentation image `labeled_img` to a ROI image. Ignores ROIs smaller than the minimum size `min_neuron_size`. """
function consolidate_labeled_img(labeled_img, min_neuron_size)
    labels = []
    # background label will have the majority of pixels
    bkg_label = 0
    max_sum = 0

    counts = Dict()
    for i in labeled_img
        if i in keys(counts)
            counts[i] += 1
        else
            counts[i] = 1
        end

        if counts[i] > max_sum
            max_sum = counts[i]
            bkg_label = i
        end
    end

    label_dict = Dict()
    label_dict[bkg_label] = UInt16(0)
    roi = 1
    for i in keys(counts)
        if (i != bkg_label) && (counts[i] >= min_neuron_size)
            label_dict[i] = UInt16(roi)
            roi = roi + 1
        end
    end
    return map(x->(x in keys(label_dict) ? label_dict[x] : UInt16(0)), labeled_img)
end

"""
    get_activity(img_roi, img; activity_func=mean)

Returns the average activity of all ROIs in an image.

# Arguments
- `img_roi`: ROI-labeled image
- `img`: raw image
- `activity_func` (optional, default `mean`): function to apply to pixel intensities of the ROI tocompute the activity for the entire ROI.
"""
function get_activity(img_roi, img; activity_func=mean)
    activity = []
    for i=1:maximum(img_roi)
        total = sum(img_roi .== i)
        if total == 0
            push!(activity, 0)
            continue
        end
        push!(activity, map(x->round(x), activity_func(img[img_roi .== i])))
    end
    return activity
end

"""
    find_convex_hull(points)

Finds the convex hull of a set of `points`, counting how many lines intersect each point.
"""
function find_convex_hull(points)
    hull = Dict()
    
    for p1 in points
        for p2 in points
            if p1 == p2
                continue
            end
            d = floor(distance(p1,p2,zscale=1))
            for t=0:d
                pt = Tuple(map(x->Int32(round(x)),p2 .+ t.*(p1.-p2)./d))
                if pt in keys(hull)
                    hull[pt][2] = hull[pt][2] + 1
                    continue
                end
                if pt in points
                    hull[pt] = [1,1]
                else
                    hull[pt] = [0,1]
                end
            end
        end
    end
    return hull
end

"""
    get_points(img_roi, roi)

Returns all points from `img_roi` corresponding to region `roi`.
"""
function get_points(img_roi, roi)
    return Tuple.(findall(img_roi .== roi))
end

"""
    distance(p1, p2; zscale=1)

Computes the distance between two points `p1` and `p2`, with the z-axis scaled by `zscale` (default 1).
"""
function distance(p1, p2; zscale=1)
    dim_dist = collect(Float64, (p1 .- p2) .* (p1 .- p2))
    dim_dist[3] *= zscale^2
    return sqrt(sum(dim_dist))
end

"""
    hull_watershed(points, hull; zscale=1, init_scale=0.7)

Does watershed segmentation on a set of `points` and their convex `hull`, with the intent of splitting concave neurons.
Scales z-axis by `zscale` (default 1) and expands first segmented neuron by `init_scale` (default 0.7) to determine second neuron location.
"""
function hull_watershed(points, hull; zscale=1, init_scale=0.7)
    weight = sum([(1 - hull[pt][1]) * hull[pt][2] for pt in keys(hull)])
    # get the farthest point from the concave points and declare it the center of the first neuron
    dist_1 = [sum([distance(p1, p2, zscale=zscale) * (1 - hull[p2][1]) * hull[p2][2] for p2 in keys(hull)])/weight for p1 in points]
    farthest_1 = argmax(dist_1)
    neuron_1_init_hull = Dict()
    for p in keys(hull)
        if distance(p, points[farthest_1], zscale=zscale) < dist_1[farthest_1] * init_scale
            neuron_1_init_hull[p] = 0
        else
            neuron_1_init_hull[p] = 1
        end
    end
    
    # get the farthest point from the first neuron and concave points and declare it the center of the second neuron
    dist_2 = [sum([distance(p1, p2, zscale=zscale) * (1 - neuron_1_init_hull[p2]) for p2 in keys(hull)])/sum([1 - neuron_1_init_hull[pt] for pt in keys(hull)]) for p1 in points]
    farthest_2 = argmax(dist_2)
    
    # sort points based on which center they're closer to
    min_dim = [minimum([points[i][l] for i=1:length(points)]) - 1 for l = 1:length(points[1])]
    max_dim = [maximum([points[i][l] for i=1:length(points)]) for l = 1:length(points[1])]
    watershed_img = zeros(Tuple(max_dim[l] .- min_dim[l] for l = 1:length(min_dim)))
    for idx in CartesianIndices(watershed_img)
        point = collect(Tuple(idx)) .+ min_dim
        watershed_img[idx] = -sum([distance(point, p2, zscale=zscale) * (1 - hull[p2][1]) * hull[p2][2] for p2 in keys(hull)])/weight
    end
    
    watershed_markers = zeros(Int64, size(watershed_img))
    watershed_markers[CartesianIndex(Tuple(collect(points[farthest_1]) .- min_dim))] = 1
    watershed_markers[CartesianIndex(Tuple(collect(points[farthest_2]) .- min_dim))] = 2
    
    results = labels_map(watershed(watershed_img, watershed_markers))
    
    neuron_1 = [p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == 1]
    neuron_2 = [p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == 2]
    
    return (neuron_1, neuron_2)
end

"""
    watershed_threshold(points, centroid_matches, predictions)

Watersheds an ROI, taking as input its peaks (found previously via thresholding) and the UNet raw output.

# Arguments:

- `points`: Set of points in the ROI to watershed.
- `centroid_matches`: Set of centroids in the points in question - each centroid will spawn a new ROI via watershed.
- `predictions`: UNet raw output (*not* thresholded)
"""
function watershed_threshold(points, centroid_matches, predictions)
    min_dim = [minimum([points[i][l] for i=1:length(points)]) - 1 for l = 1:length(points[1])]
    max_dim = [maximum([points[i][l] for i=1:length(points)]) for l = 1:length(points[1])]
    watershed_img = zeros(Tuple(max_dim[l] .- min_dim[l] for l = 1:length(min_dim)))
    for idx in CartesianIndices(watershed_img)
        point = collect(Tuple(idx)) .+ min_dim
        watershed_img[idx] = -predictions[CartesianIndex(Tuple(point))]
    end
    watershed_markers = zeros(Int64, size(watershed_img))
    for (i, centroid) in enumerate(centroid_matches)
        watershed_markers[CartesianIndex(Tuple(collect(map(x->Int32(x), centroid)) .- min_dim))] = i
    end
    
    results = labels_map(watershed(watershed_img, watershed_markers))
    
    neurons = [[p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == i] for i=1:length(centroid_matches)]
    
    return neurons
end

"""
    detect_incorrect_merges(img_roi, predictions, thresholds, neuron_sizes)

Detects incorrectly merged ROIs via thresholding. Thresholds the UNet raw output multiple times, checking if 
an ROI gets split into multiple, smaller ROIs at higher threshold values.

# Arguments: 

- `img_roi`: Current ROIs for the image - the method checks each ROI for incorrect merging
- `predictions`: UNet raw output (*not* thresholded)
- `thresholds`: Array of threshold values - at each value, check if an ROI was split
- `neuron_sizes`: Array of neuron size values, one per threshold.
    Neurons that were found in a threshold that are smaller than the corresponding value are discarded and not counted for ROI split evidence.
"""
function detect_incorrect_merges(img_roi, predictions, thresholds, neuron_sizes)
    bad_rois = Dict()
    for t in 1:length(thresholds)
        img_roi_t = instance_segmentation(predictions .> thresholds[t], min_neuron_size=neuron_sizes[t])
        centroids = get_centroids_round(img_roi_t)
        for i=1:maximum(img_roi)
            points = get_points(img_roi, i)
            centroid_matches = []
            for centroid in centroids
                if centroid in points
                    push!(centroid_matches, centroid)
                end
            end
            if length(centroid_matches) >= length(get(bad_rois, i, [0,0]))
                bad_rois[i] = centroid_matches
            end
        end
    end
    return bad_rois
end

"""
    instance_segmentation_threshold(img_roi, predictions; thresholds=[0.7, 0.8, 0.9], neuron_sizes=[5,4,4])

Further instance segments a preliminary ROI image by thresholding UNet predictions and checking if ROIs split during thresholding.

# Arguments:

- `img_roi`: image that maps points to their current ROIs
- `predictions`: UNet raw predictions

# Optional keyword arguments
- `thresholds`: Array of threshold values - at each value, check if an ROI was split. Default [0.7, 0.8, 0.9]
- `neuron_sizes`: Array of neuron size values, one per threshold.
    Neurons that were found in a threshold that are smaller than the corresponding value are discarded and not counted for ROI split evidence.
    Default [5,4,4]
"""
function instance_segmentation_threshold(img_roi, predictions; thresholds=[0.7, 0.8, 0.9], neuron_sizes=[5,4,4])
    img_roi_new = copy(img_roi)
    bad_rois = detect_incorrect_merges(img_roi, predictions, thresholds, neuron_sizes)
    idx = maximum(img_roi) + 1
    for roi in keys(bad_rois)
        neurons = watershed_threshold(get_points(img_roi, roi), bad_rois[roi], predictions)
        for i in 1:length(neurons)
            if i == 1
                continue
            end
            neuron = neurons[i]
            for pt in neuron
                img_roi_new[CartesianIndex(pt)] = idx
            end
            idx = idx + 1
        end
    end
    return img_roi_new
end



"""
    instance_segment_hull(img_roi, points, hull; min_neuron_size=10, zscale=1, init_scale=0.7)

Instance segments an image `img_roi` given set of `points` with a given convex `hull` via watershedding.
Discards neurons with size less than `min_neuron_size` (default 10), scales z-axis by `zscale` (default 1),
and expands first segmented neuron by `init_scale` (default 0.7) to determine second neuron location.
"""
function instance_segment_hull(img_roi, points, hull; min_neuron_size=10, zscale=1, init_scale=0.7)
    neuron_1, neuron_2 = hull_watershed(points, hull, zscale=zscale, init_scale=init_scale)
    failed_flag = (length(neuron_1) < min_neuron_size || length(neuron_2) < min_neuron_size)
    if failed_flag
        return (img_roi, true)
    end
    m = maximum(img_roi)
    img_roi_new = copy(img_roi)
    for pt in neuron_2
        img_roi_new[CartesianIndex(pt)] = m+1
    end
    return (img_roi_new, false)
end

"""
    concave_score(points, hull)

Generates a score for how concave a set of `points` with a given convex `hull` is.
"""
function concave_score(points, hull)
    return sum([(1-x[1])*x[2] for x in values(hull)]) * length(hull) / sum([x[2] for x in values(hull)]) * log(length(values(hull))) / length(values(hull))
end

"""
    is_concave(points, hull; threshold_scale=0.3)

Detects if a given set of `points` is concave given its `hull`, by comparing its concavity score with `threshold_scale`.
"""
function is_concave(points, hull; threshold_scale=0.3)
    return concave_score(points, hull) > threshold_scale
end

"""
    find_concave_neurons(img_roi; num_neurons=10, threshold_scale=0.3)

Finds the `num_neurons` (default 10) most concave neurons in an image `img_roi`.
Concave neurons must also meet the `threshold_scale` (default 0.3).
"""
function find_concave_neurons(img_roi; num_neurons=10, threshold_scale=0.3)
    hull_points = Dict()
    top = [[0.0,0.0] for i=1:num_neurons]
    for roi in 1:maximum(img_roi)
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        score = concave_score(points, hull)
        if score > max(top[1][2], threshold_scale)
            top[1][1] = roi
            top[1][2] = score
            top = sort(top, by=x->x[2])
        end
        hull_points[roi] = (points, hull)
    end
    concave = Dict()
    best_score = 0
    for t in top
        if t[1] != 0
            if best_score == 0
                best_score = t[2]
            end
            concave[t[1]] = hull_points[t[1]]
        end
    end
    return (concave, best_score)
end

"""
    instance_segment_concave(
        img_roi; threshold_scale::Real=0.3, init_scale::Real=0.7,
        zscale::Real=1, min_neuron_size::Integer=10, scale_recurse_multiply::Real=1.5, num_neurons::Integer=10
    )

Recursively segments all concave neurons in an image. 

# Arguments
- `img_roi`: Image to segment

# Optional keyword arguments
- `threshold_scale::Real`: Neurons less concave than this won't be segmented. Default 0.3
- `num_neurons::Real`: Maximum number of concave neurons per frame. Defaul 10.
- `zscale::Real`: Scale of z-axis relative to xy plane. Default 1.
- `min_neuron_size::Integer`: Minimum size of a neuron (in pixels). Default 10.
- `scale_recurse_multiply::Real`: Factor to increase the concavity threshold for recursive segmentation. Default 1.5.
- `init_scale::Real`: Amount to expand first neuron before computing location of second neuron. Default 0.7. 
"""
function instance_segment_concave(img_roi; threshold_scale::Real=0.3, init_scale::Real=0.7, zscale::Real=1, min_neuron_size::Integer=10, scale_recurse_multiply::Real=1.5, num_neurons::Integer=10)
    concave, score = find_concave_neurons(img_roi, threshold_scale=threshold_scale, num_neurons=num_neurons)
    threshold_scale_recurse = score * scale_recurse_multiply
    concave_queue = Queue{Int64}()
    errors = []
    for roi in keys(concave)
        enqueue!(concave_queue, roi)
    end
    while !isempty(concave_queue)
        roi = dequeue!(concave_queue)
        img_roi, error = instance_segment_hull(img_roi, concave[roi][1], concave[roi][2], zscale=zscale, min_neuron_size=min_neuron_size, init_scale=init_scale)
        if error
            append!(errors, roi)
            continue
        end
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        if is_concave(points, hull, threshold_scale=threshold_scale_recurse)
            concave[roi] = (points, hull)
            enqueue!(concave_queue, roi)
        end
        # newest neuron will be the highest number
        roi = maximum(img_roi)
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        if is_concave(points, hull, threshold_scale=threshold_scale_recurse)
            concave[roi] = (points, hull)
            enqueue!(concave_queue, roi)
        end
    end
    return (img_roi, errors)
end

