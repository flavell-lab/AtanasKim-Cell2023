"""
    get_neighbors_diagonal(idx, s)

Gets all neighbors of a voxel (including diagonally).

# Arguments

- `idx`: coordinate to find neighbors of
- `s`: shape of image (to avoid finding neighbors outside the image boundary).
"""
function get_neighbors_diagonal(idx, s)
    nbs = [[]]
    for elt in zip(idx, s)
        new_nbs = []
        for nb in nbs
            if elt[1] != 1
                new_nb = copy(nb)
                push!(new_nbs, append!(new_nb, elt[1] - 1))
            end
            if elt[1] != elt[2]
                new_nb = copy(nb)
                push!(new_nbs, append!(new_nb, elt[1] + 1))
            end
            new_nb = copy(nb)
            push!(new_nbs, append!(new_nb, elt[1]))
        end
        nbs = new_nbs
    end
    return filter(x->x!=idx, map(x->Tuple(x),nbs))
end

"""
    get_neighbors_cartesian(idx, s)

Gets all neighbors of a voxel (not including diagonally) within an N-dimensional image.

# Arguments

- `idx`: coordinate to find neighbors of
- `s`: shape of image (to avoid finding neighbors outside the image boundary).
"""
function get_neighbors_cartesian(idx, s)
    idx_arr = collect(Tuple(idx))
    nbs = []
    for i in 1:length(idx)
        if idx[i] != 1
            nb = copy(idx_arr)
            nb[i] = idx[i] - 1
            push!(nbs, nb)
        end
        if idx[i] != s[i]
            nb = copy(idx_arr)
            nb[i] = idx[i] + 1
            push!(nbs, nb)
        end
    end
    return map(x->CartesianIndex(Tuple(x)),nbs)
end


"""
    create_weights(
        label; scale_xy::Real=0.36, scale_z::Real=1, metric::String="taxicab", weight_foreground::Real=4,
        weight_bkg_gap::Real=24, boundary_weight=nothing, scale_bkg_gap::Bool=false
    )

Creates a weight map from a given labeled dataset.
Unlabeled data (label 0) has weight 0 and background data (label 2) far from foreground data
has weight 1. Foreground data (label 1) has a higher weight.
Background data near the foreground (also label 2) has weight exponentially decaying
down from the foreground weight. "Background-gap" data that serves to mark boundaries between foreground objects
(label 3) has the highest weight.

# Arguments

- `label`: labeled dataset to turn into weights

# Optional keyword arguments

- `scale_xy::Real`: Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 0.36.
- `scale_z::Real`: Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.
- `metric::String`: Metric to compute distance. Default taxicab (currently, no other metrics are implemented)
- `weight_foreground::Real`: weight of foreground (1) label
- `weight_bkg_gap::Real`: weight of background-gap (3) label
- `boundary_weight`: weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.
- `scale_bkg_gap::Bool`: whether to upweight background-gap pixels for each neuron pixel they border.
"""
function create_weights(label; scale_xy::Real=0.36, scale_z::Real=1, metric::String="taxicab", weight_foreground::Real=4,
        weight_bkg_gap::Real=24, boundary_weight=nothing, scale_bkg_gap::Bool=false)
    weights = zeros(size(label))
    # add flat weight to background, so regions far from neurons get counted
    # this will be overwritten with higher weights near neurons
    bkg = [label[x] == 2 for x in CartesianIndices(size(weights))]
    weights = weights .+ bkg
    # over-weight neurons and regions near neurons by neuron-to-background ratio
    if weight_foreground == "proportional"
        multiplier = sum(label .== 2) / sum(label .== 1)
    else
        multiplier = weight_foreground
    end
    to_update = Queue{Tuple}()
    visited = Dict()
    decay_z = 2.0^(-scale_z)
    decay_xy = 2.0^(-scale_xy)
    for coord in CartesianIndices(size(weights))
        if label[coord] == 1
            enqueue!(to_update, (coord, multiplier, coord))
            weights[coord] = multiplier
            visited[coord] = true
        else
            visited[coord] = false
        end
    end
    count = 0
    while !isempty(to_update)
        item = dequeue!(to_update)
        nbs = get_neighbors_cartesian(item[1], size(weights))
        for nb in nbs
            if metric == "euclidean"
                throw("Euclidean metric not fully functional.")
                #weight = multiplier * get_weight(nb, item[3], decay, scale_xy, scale_z)
            elseif metric == "taxicab"
                # nb is a neighbor of our item - check whether to use xy or z scaling
                weight = item[2] * ((item[1][3] != nb[3]) ? decay_z : decay_xy)
            else
                throw("Metric "*metric*" not implemented.")
            end
            # weight is still above background levels, so we will overwrite background
            if weight > 1 && weight > weights[nb]
                enqueue!(to_update, (nb, weight, item[3]))
                weights[nb] = weight
            end
        end
        count += 1
    end
    # don't weight unlabeled points at all (we have to calculate them above as otherwise nearby background would be weighted incorrectly)
    labeled = [label[x] != 0 for x in CartesianIndices(size(weights))]
    weights = weights .* labeled
    
    # don't weight boundary points and make neurons smaller (neural net needs more flexibility here, and their labeling is hard to get consistent)
    if boundary_weight != nothing
        for idx in CartesianIndices(size(weights))
            nbs = get_neighbors_cartesian(idx, size(weights))
            for nb in nbs
                # only reweight pixels in xy plane
                if nb[3] != idx[3]
                    continue
                end
                # only overwrite neuron pixels bordering background
                if (label[idx] == 1) && (label[nb] != label[idx]) && (weights[nb] != 0)
                    weights[idx] = boundary_weight
                end
            end
        end
    end

    for idx in CartesianIndices(size(weights))
        if !scale_bkg_gap
            weights[idx] = weight_bkg_gap
            continue
        end
        nbs = get_neighbors_cartesian(idx, size(weights))
        neuron_nbs = 0
        if label[idx] == 3 
            for nb in nbs
                if label[nb] == 1
                    neuron_nbs = neuron_nbs + 1 
                end
            end
            weights[idx] = weight_bkg_gap * (neuron_nbs + 1)
        end
    end
     
    return collect(map(x->convert(Float64, x), weights))
end


"""
    decrease_SN(img; factor::Real=10, std::Real=1)

Decreases S/N ratio of an image by the given factor.

# Arguments
- `img`: Image to reduce S/N of

# Optional keyword arguments
- `factor::Real`: Factor to reduce S/N by. Default 10.
- `std::Real`: Standard deviation of image to reduce S/N of
"""
function decrease_SN(img; factor::Real=10, std::Real=1)
    m = median(img)
    r = Normal(m, std)
    return collect(map(x->UInt16((x-m) ÷ factor + round(rand(r))), img))
end


"""
    make_unet_input_h5(
        img_raw::Array, img_label::Union{Nothing,Array}, path_h5::String; crop=nothing, transpose::Bool=false, 
        weight_strategy::String="neighbors", metric::String="taxicab", scale_xy::Real=0.36, scale_z::Real=1, 
        weight_foreground::Real=6,  weight_bkg_gap::Real=10, boundary_weight=nothing,
        bin_scale=[1,1,1], SN_reduction_factor::Real=1, SN_percent::Real=16, scale_bkg_gap::Bool=false
    )

Makes a UNet input file. This function supports making files either for training or prediction.

# Arguments
- `img_raw::Array`: Raw image
- `img_label::Union{Nothing, Array}`: Image label. If `nothing`, label and weight will not be generated.
- `path_h5::String`: Path to HDF5 output directory.
- `crop` (optional, default `nothing`): `[crop_x, crop_y, crop_z]`, where every point with coordinates not in the given ranges is cropped out.
- `transpose::Bool` (optional, default `false`): whether to transpose the x-y coordinates of the image.
- `weight_strategy::String` (optional): method to generate weights from labels. Default and recommended is `neighbors`, which weights background pixels nearby foreground higher.
   Alternative is `proportional`, which will weight foreground and background constantly at a value inversely proportional to the number of pixels in those weights.
   The `proportional` weight function will ignore labels that are not 1 or 2 (including the background-gap label 3).
   This parameter has no effect if `img_label` is the empty string.
- `metric::String` (optional): metric used to infer distance. Default (and only metric currently implemented) is `taxicab`. 
   This parameter has no effect if `img_label` is the empty string.
- `scale_xy::Real` (optional): Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 1.
    This parameter has no effect if `img_label` is the empty string.
- `scale_z::Real` (optional): Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.
    This parameter has no effect if `img_label` is the empty string.
- `weight_foreground::Real` (optional, default 6): weight of foreground (1) label
    This parameter has no effect if `img_label` is the empty string.
- `weight_bkg_gap::Real` (optional, default 10): weight of background-gap (3) label
    This parameter has no effect if `img_label` is the empty string.
- `boundary_weight` (optional): weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.
    This parameter has no effect if `img_label` is the empty string.
- `bin_scale` (optional): scale to bin image in each dimension [X,Y,Z]. Default [1,1,1] (no binning).
- `SN_reduction_factor` (optional): amount to reduce. Default 1 (no reduction)
- `SN_percent` (optional): percentile to estimate std of image from. Default 16.
- `scale_bkg_gap::Bool` (optional): whether to upweight background-gap pixels for each neuron pixel they border. Default false.
    This parameter has no effect if `img_label` is the empty string.
"""
function make_unet_input_h5(img_raw::Array, img_label::Union{Nothing,Array}, path_h5::String; crop=nothing, transpose::Bool=false, weight_strategy::String="neighbors", 
    metric::String="taxicab", scale_xy::Real=0.36, scale_z::Real=1, weight_foreground::Real=6, weight_bkg_gap::Real=10, boundary_weight=nothing,
    bin_scale=[1,1,1], SN_reduction_factor::Real=1, SN_percent::Real=16, scale_bkg_gap::Bool=false)
    
    make_label = !isnothing(img_label)

    if !isnothing(crop)
        if make_label
            img_label = img_label[crop[1], crop[2], crop[3]]
        end
        img_raw = img_raw[crop[1], crop[2], crop[3]]
    end
    
    if transpose
        if make_label
            img_label = permutedims(img_label, [2,1,3])
        end
        img_raw = permutedims(img_raw, [2,1,3])
    end
    
    std = median(img_raw) - percentile(collect(Iterators.flatten(img_raw)), SN_percent)
    if bin_scale != [1,1,1]
        if make_label
            label = resample_img(img_label, bin_scale; dtype="label")
        end
        img_raw = resample_img(img_raw, bin_scale)
    end
    
    if SN_reduction_factor != 1
        img_raw = decrease_SN(img_raw; factor=SN_reduction_factor, std=std)
    end

    if make_label
        # Don't weight unlabeled, weight all neurons and bkg equally (corrected for number of pixels)
        if weight_strategy == "proportional"
            weight_arr = [0, round(sum(img_label .== 2) / sum(img_label .== 1)), 1]
            weight_fn = x->weight_arr[Int16(x)+1]
            weight = map(weight_fn, img_label)
        # Don't weight unlabeled, weight background pixels nearby neuron pixels higher.
        elseif weight_strategy == "neighbors"
            weight = create_weights(img_label, scale_xy=scale_xy, scale_z=scale_z,
                        weight_foreground=weight_foreground, weight_bkg_gap=weight_bkg_gap,
                        metric=metric, boundary_weight=boundary_weight, scale_bkg_gap=scale_bkg_gap)
        else
            throw("Weight strategy "*weight_strategy*" not implemented.")
        end
    end
    
    # Neurons are labeled as 1, assume everything else is background, but weight only real background
    if make_label
        img_label_new = map(x->Int16(x), img_label .== 1)
    end
    
    # weighting
    # weight neuron pixels more heavily since there are fewer of them
    h5open(path_h5, "w") do h5f
        h5f["raw"] = collect(img_raw)
        if make_label
            h5f["label"] = collect(img_label_new)
            h5f["weight"] = collect(weight)
        end
    end
    
    nothing
end

"""
    make_unet_input_h5(
        path_nrrd_raw::String, path_nrrd_label::Union{Nothing, String},
        path_h5::String; crop=nothing, transpose::Bool=false, weight_strategy::String="neighbors",
        metric::String="taxicab", scale_xy::Real=1, scale_z::Real=1, weight_foreground::Real=6,
        weight_bkg_gap::Real=10, boundary_weight=nothing, bin_scale=[1,1,1],
        SN_reduction_factor::Real=1, SN_percent::Real=16, scale_bkg_gap::Bool=false
    )

Makes UNet input files from all files in a directory. This function supports making files either for training or prediction.

# Arguments
- `path_nrrd_raw::String`: Path to raw data NRRD files
- `path_nrrd_label::Union{Nothing, String}`: Path to NRRD label files. If nothing, labels and weights will not be generated.
- `path_h5::String`: Path to HDF5 output files.
- `crop` (optional, default `nothing`): `[crop_x, crop_y, crop_z]`, where every point with coordinates not in the given ranges is cropped out.
- `transpose::Bool` (optional, default `false`): whether to transpose the x-y coordinates of the image.
- `weight_strategy::String` (optional): method to generate weights from labels. Default and recommended is `neighbors`, which weights background pixels nearby foreground higher.
   Alternative is `proportional`, which will weight foreground and background constantly at a value inversely proportional to the number of pixels in those weights.
   The `proportional` weight function will ignore labels that are not 1 or 2 (including the background-gap label 3).
   This parameter has no effect if `nrrd_path` is the empty string.
- `metric::String` (optional): metric used to infer distance. Default (and only metric currently implemented) is `taxicab`. 
   This parameter has no effect if `nrrd_path` is the empty string.
- `scale_xy::Real` (optional): Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 1.
    This parameter has no effect if `nrrd_path` is the empty string.
- `scale_z::Real` (optional): Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.
    This parameter has no effect if `nrrd_path` is the empty string.
- `weight_foreground::Real` (optional, default 6): weight of foreground (1) label
    This parameter has no effect if `nrrd_path` is the empty string.
- `weight_bkg_gap::Real` (optional, default 10): weight of background-gap (3) label
    This parameter has no effect if `nrrd_path` is the empty string.
- `boundary_weight` (optional): weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.
    This parameter has no effect if `nrrd_path` is the empty string.
- `bin_scale` (optional): scale to bin image in each dimension [X,Y,Z]. Default [1,1,1] (no binning).
- `SN_reduction_factor` (optional): amount to reduce. Default 1 (no reduction)
- `SN_percent` (optional): percentile to estimate std of image from. Default 16.
- `scale_bkg_gap::Bool` (optional): whether to upweight background-gap pixels for each neuron pixel they border. Default false.
    This parameter has no effect if `nrrd_path` is the empty string.
"""
function make_unet_input_h5(path_nrrd_raw::String, path_nrrd_label::Union{Nothing, String},
    path_h5::String; crop=nothing, transpose::Bool=false, weight_strategy::String="neighbors",
    metric::String="taxicab", scale_xy::Real=1, scale_z::Real=1, weight_foreground::Real=6,
    weight_bkg_gap::Real=10, boundary_weight=nothing, bin_scale=[1,1,1],
    SN_reduction_factor::Real=1, SN_percent::Real=16, scale_bkg_gap::Bool=false)
    
    img_label = isnothing(path_nrrd_label) ? nothing : read_img(NRRD(path_nrrd_label))
    img_raw = read_img(NRRD(path_nrrd_raw))
    
    make_unet_input_h5(img_raw, img_label, path_h5, crop=crop, transpose=transpose,
        weight_strategy=weight_strategy, metric=metric, scale_xy=scale_xy, scale_z=scale_z,
        weight_foreground=weight_foreground, weight_bkg_gap=weight_bkg_gap,
        boundary_weight=boundary_weight, bin_scale=bin_scale, SN_reduction_factor=SN_reduction_factor,
        SN_percent=SN_percent, scale_bkg_gap=scale_bkg_gap)
    
    nothing
end


"""
    make_unet_input_h5(param_path::Dict, path_dir_nrrd::String, t_range, ch_marker::Int, f_basename::Function)

Generatesn HDF5 file, to be input to the UNet, out of a raw image file and a label file. Assumes 3D data.

# Arguments
 - `param_path::Dict`: Dictionary containing paths to directories and a `get_basename` function that returns NRRD file names, including:
    - `path_dir_unet_data`: Path to UNet input and output data
 - `path_dir_nrrd::String`: Path to NRRD files
 - `t_range`: Time points to watershed
 - `ch_marker::Int`: Marker channel
 - `f_basename::Function`: Function that returns the name of NRRD files
"""
function make_unet_input_h5(param_path::Dict, path_dir_nrrd::String, t_range, ch_marker::Int, f_basename::Function)
    
    path_dir_unet_data = param_path["path_dir_unet_data"]
    create_dir(path_dir_unet_data)
    
    @showprogress for t = t_range
        path_nrrd = joinpath(path_dir_nrrd, f_basename(t, ch_marker) * ".nrrd")
        make_unet_input_h5(path_nrrd, nothing, joinpath(path_dir_unet_data, "$(t).h5"))
    end
end
