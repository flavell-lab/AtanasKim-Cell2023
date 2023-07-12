"""
    crop_rotate(
        img, crop_x, crop_y, crop_z, theta, worm_centroid; fill="median",
        degree=Linear(), dtype=Int16, crop_pad=[5,5,5], min_crop_size=[210,96,51]
    )

Rotates and then crops an image, optionally along with its head and centroid locations.

# Arguments

- `img`: image to transform (3D)
- `crop_x`: crop amount in x-dimension
- `crop_y`: crop amount in y-dimension
- `crop_z`: crop amount in z-dimension
- `theta`: rotation amount in xy-plane
- `worm_centroid`: centroid of the worm (to rotate around). NOT the centroids of ROIs in the worm.

## Optional keyword arguments
- `fill`: what value to put in pixels that were rotated in from outside the original image.
  If kept at its default value "median", the median of the image will be used.
  Otherwise, it can be set to a numerical value.
- `degree`: degree of the interpolation. Default `Linear()`; can set to `Constant()` for nearest-neighbors.
- `dtype`: type of data in resulting image. Default `Int16`.
- `crop_pad`: amount to pad the cropping in each dimension. Default 5 pixels in each dimension.
- `min_crop_size`: minimum size of cropped image. Default [210,96,51], the UNet input size.

Outputs a new image that is the cropped and rotated version of `img`.
"""
function crop_rotate(img, crop_x, crop_y, crop_z, theta, worm_centroid; fill="median",
        degree=Linear(), dtype=Int16, crop_pad=[5,5,5], min_crop_size=[210,96,51])
    new_img = nothing
    if fill == "median"
        fill_val = dtype(round(median(img)))
    else
        fill_val = fill
    end
    
    imsize = size(img)
    # make sure we aren't trying to crop past image boundary after adding crop padding
    cz = [max(crop_z[1]-crop_pad[3], 1), min(crop_z[2]+crop_pad[3], imsize[3])]
    increase_crop_size!(cz, 1, imsize[3], min_crop_size[3])

    cx = nothing 
    cy = nothing 
    tfm = recenter(RotMatrix(theta), worm_centroid[1:2])

    for z=cz[1]:cz[2]
        new_img_z = warp(img[:,:,z], tfm, degree)
        # initialize x and y cropping parameters
        if isnothing(cx)
            cx = [crop_x[1]-crop_pad[1], crop_x[2]+crop_pad[1]]
            cx = [max(cx[1], new_img_z.offsets[1]+1), min(cx[2], new_img_z.offsets[1] + size(new_img_z)[1])]
            increase_crop_size!(cx, new_img_z.offsets[1] + 1, new_img_z.offsets[1] + size(new_img_z)[1], min_crop_size[1])
        end
        
        if isnothing(cy)
            cy = [crop_y[1]-crop_pad[2], crop_y[2]+crop_pad[2]]
            cy = [max(cy[1], new_img_z.offsets[2]+1), min(cy[2], new_img_z.offsets[2] + size(new_img_z)[2])]
            increase_crop_size!(cy, new_img_z.offsets[2] + 1, new_img_z.offsets[2] + size(new_img_z)[2], min_crop_size[2])
        end

        if isnothing(new_img)
            new_img = zeros(dtype, (cx[2] - cx[1] + 1, cy[2] - cy[1] + 1, cz[2] - cz[1] + 1))
        end

        for x=cx[1]:cx[2]
            for y=cy[1]:cy[2]
                val = new_img_z[x,y]
                # val is NaN
                if val != val || val == 0
                    val = fill_val
                end
                if dtype <: Integer
                    val = round(val)
                end
                new_img[x-cx[1]+1,y-cy[1]+1,z-cz[1]+1] = dtype(val)
            end
        end
    end

    return (new_img[1:cx[2]-cx[1]+1, 1:cy[2]-cy[1]+1, 1:cz[2]-cz[1]+1], cx, cy, cz)
end

"""
    uncrop_img_roi(img_roi, crop_params, img_size; degree=Constant(), dtype=Int16)

Uncrops an ROI image.

# Arguments
- `img_roi`: ROI image to uncrop
- `crop_params`: Dictionary of cropping parameters
- `img_size`: Size of uncropped image
- `degree` (optional): Interpolation method used. Default `Constant()` which results in nearest-neighbor interpolation
- `dtype` (optional): Data type of uncropped image. Default `Int16`
"""
function uncrop_img_roi(img_roi, crop_params, img_size; degree=Constant(), dtype=Int16)
    worm_centroid_cropped = [crop_params["worm_centroid"][i] - crop_params["updated_crop"][i][1] + 1 for i=1:3] 
    tfm = recenter(RotMatrix(-crop_params["θ"]), worm_centroid_cropped[1:2])
    uncropped_img = zeros(dtype, img_size)

    for z=1:size(img_roi,3)
        new_img_z = warp(img_roi[:,:,z], tfm, degree)
        for x=new_img_z.offsets[1]+1:new_img_z.offsets[1]+size(new_img_z,1)
            for y=new_img_z.offsets[2]+1:new_img_z.offsets[2]+size(new_img_z,2)
                val = new_img_z[x,y]
                # skip NaN values
                if isnan(val) || val == 0
                    continue
                end
                coord = [x,y,z]
                new_coord=[coord[i]+crop_params["updated_crop"][i][1]-1 for i=1:3]
                # out of bounds
                if any(new_coord .< [1,1,1]) || any(new_coord .> img_size)
                    continue
                end
                if dtype <: Integer
                    val = round(val)
                end
                uncropped_img[CartesianIndex(Tuple(new_coord))] = dtype(val)
            end
        end
    end
    return uncropped_img
end

"""
    uncrop_img_rois(
        param_path::Dict, param::Dict, crop_params::Dict, img_size;
        roi_cropped_key::String="path_dir_roi_watershed", roi_uncropped_key::String="path_dir_roi_watershed_uncropped"
    )

Uncrops all ROI images.

# Arguments
 - `param_path::Dict`: Dictionary containing paths to files
 - `param::Dict`: Dictionary containing pipeline parameters
 - `crop_params::Dict`: Dictionary containing cropping parameters
 - `img_size`: Size of uncropped images to generate
 - `roi_cropped_key::String` (optional): Key in `param_path` corresponding to locations of ROI images to uncrop. Default `path_dir_roi_watershed`.
 - `roi_uncropped_key::String` (optional): Key in `param_path` corresponding to location to put uncropped ROI images. Default `path_dir_roi_watershed_uncropped`.
"""
function uncrop_img_rois(param_path::Dict, param::Dict, crop_params::Dict, img_size;
        roi_cropped_key::String="path_dir_roi_watershed", roi_uncropped_key::String="path_dir_roi_watershed_uncropped")
    create_dir(param_path[roi_uncropped_key])
    @showprogress for t in param["t_range"]
        img_roi_nrrd = NRRD(joinpath(param_path[roi_cropped_key], "$(t).nrrd"))
        img_roi = read_img(img_roi_nrrd)
        img_roi_uncropped = uncrop_img_roi(img_roi, crop_params[t], img_size)
        write_nrrd(joinpath(param_path[roi_uncropped_key], "$(t).nrrd"), img_roi_uncropped, spacing(img_roi_nrrd))
    end
end

"""
    increase_crop_size!(crop, min_ind, max_ind, crop_size)

Increases crop size of a `crop`. Requires the min amd max indices of the image `min_ind` and `max_ind`, and the desired minimum crop size `crop_size`.
"""
function increase_crop_size!(crop, min_ind, max_ind, crop_size)
    # crop size is larger than image size
    if crop_size > max_ind - min_ind + 1
        throw("Crop size cannot be larger than image size")
    end
    d = 1
    while crop[2] - crop[1] < crop_size
        if crop[2] >= max_ind
            crop[1] -= 1
        elseif crop[1] <= min_ind
            crop[2] += 1
        else
            d = 3 - d
            crop[d] += 2*d - 3
        end
    end
    nothing
end

"""
    get_crop_rotate_param(img; threshold_intensity::Real=3, threshold_size::Int=10)

Generates cropping and rotation parameters from a frame by detecting the worm's location with thresholding and noise removal.
The cropping parameters are designed to remove the maximum fraction of non-worm pixels.

# Arguments
- `img`: Image to crop

# Optional keyword arguments
- `threshold_intensity`: Number of standard deviations above mean for a pixel to be considered part of the worm. Default 3.
- `threshold_size`: Number of adjacent pixels that must meet the threshold to be counted. Default 10.
"""
function get_crop_rotate_param(img; threshold_intensity::Real=3, threshold_size::Int=10)
    # threshold image to detect worm
    thresh_img = ski_morphology.remove_small_objects(img .> mean(img) + threshold_intensity * std(img), threshold_size)
    # extract worm points
    frame_worm_nonzero = map(x->collect(Tuple(x)), filter(x->thresh_img[x]!=0, CartesianIndices(thresh_img)))
    # get center of worm
    worm_centroid = reduce((x,y)->x.+y, frame_worm_nonzero) ./ length(frame_worm_nonzero)
    # get axis of worm
    deltas = map(x->collect(x.-worm_centroid), frame_worm_nonzero)
    cov_mat = cov(deltas)
    mat_eigvals = eigvals(cov_mat)
    mat_eigvecs = eigvecs(cov_mat)
    eigvals_order = sortperm(mat_eigvals, rev=true)
    # PC1 will be the long dimension of the worm
    # We only want to rotate in xy plane, so project to that plane
    long_axis = mat_eigvecs[:,eigvals_order[1]]
    long_axis[3] = 0
    long_axis = long_axis ./ sqrt(sum(long_axis .^ 2))
    short_axis = [-long_axis[2], long_axis[1], 0]
    theta = atan(long_axis[2]/long_axis[1])
    if long_axis[1] < 0
        theta += pi
    end

    
    # get coordinates of points in worm axis-dimension
    distances = map(x->sum(x.*long_axis), deltas)
    distances_short = map(x->sum(x.*short_axis), deltas)
    distances_z = map(x->sum(x.*[0,0,1]), deltas)


    # get cropping parameters
    crop_x = (Int64(floor(minimum(distances) + worm_centroid[1])), Int64(ceil(maximum(distances) + worm_centroid[1])))
    crop_y = (Int64(floor(minimum(distances_short) + worm_centroid[2])), Int64(ceil(maximum(distances_short) + worm_centroid[2])))
    crop_z = (Int64(floor(minimum(distances_z) + worm_centroid[3])), Int64(ceil(maximum(distances_z) + worm_centroid[3])))

    return (crop_x, crop_y, crop_z, theta, worm_centroid)
end

"""
    crop_rotate!(
        path_dir_nrrd::String, path_dir_nrrd_crop::String, path_dir_MIP_crop::String, t_range, 
        ch_list, dict_crop_rot_param::Dict, threshold_size::Int, threshold_intensity::AbstractFloat, 
        spacing_axi::AbstractFloat, spacing_lat::AbstractFloat, f_basename::Function, save_MIP::Bool
    )

Generates cropped and rotated images from a set of input images.

# Arguments
- `path_dir_nrrd`: Path to the directory containing the input images.
- `path_dir_nrrd_crop`: Path to the directory where the cropped images will be saved.
- `path_dir_MIP_crop`: Path to the directory where the maximum intensity projection (MIP) images will be saved.
- `t_range`: Range of time points to process.
- `ch_list`: List of channels to process.
- `dict_crop_rot_param`: Dictionary containing the cropping and rotation parameters for each time point.
- `threshold_size`: Number of adjacent pixels that must meet the threshold to be counted.
- `threshold_intensity`: Number of standard deviations above mean for a pixel to be considered part of the worm.
- `spacing_axi`: Axial spacing of the input images.
- `spacing_lat`: Lateral spacing of the input images.
- `f_basename`: Function that returns the base name of the input image file.
- `save_MIP`: Boolean indicating whether to save the MIP images.

# Output
- Cropped and rotated images are saved in `path_dir_nrrd_crop`.
- Maximum intensity projection (MIP) images are saved in `path_dir_MIP_crop`.
- In-place modification of the `dict_crop_rot_param` dictionary with the cropping and rotation parameters for each time point.
- Returns a tuple containing errors and time points where the worm might have been out of focus.
"""
function crop_rotate!(path_dir_nrrd::String, path_dir_nrrd_crop::String, path_dir_MIP_crop::String, t_range, ch_list, dict_crop_rot_param::Dict,
        threshold_size::Int, threshold_intensity::AbstractFloat, spacing_axi::AbstractFloat, spacing_lat::AbstractFloat, f_basename::Function, save_MIP::Bool)
    create_dir.([path_dir_nrrd_crop, path_dir_MIP_crop])

    dict_error = Dict{Int, Any}()
    focus_issues = []
    
    @showprogress for t = t_range
        try
            img = read_img(NRRD(joinpath(path_dir_nrrd, f_basename(t, ch_list[1]) * ".nrrd")))
            if haskey(dict_crop_rot_param, t)
                crop_x, crop_y, crop_z = dict_crop_rot_param[t]["crop"]
                θ_ = dict_crop_rot_param[t]["θ"]
                worm_centroid = dict_crop_rot_param[t]["worm_centroid"]
            else
                dict_crop_rot_param[t] = Dict()
                crop_x, crop_y, crop_z, θ_, worm_centroid = get_crop_rotate_param(img,
                    threshold_intensity=threshold_intensity, threshold_size=threshold_size)
                dict_crop_rot_param[t] = Dict()
                dict_crop_rot_param[t]["crop"] = [crop_x, crop_y, crop_z]
                dict_crop_rot_param[t]["θ"] = θ_
                dict_crop_rot_param[t]["worm_centroid"] = worm_centroid
            end

            for ch = ch_list
                bname = f_basename(t, ch)
                img = read_img(NRRD(joinpath(path_dir_nrrd, bname * ".nrrd")))
                img_crop, cx, cy, cz = crop_rotate(img, crop_x, crop_y, crop_z, θ_, worm_centroid)

                # these parameters are not dependent on channel
                if !haskey(dict_crop_rot_param[t], "updated_crop")
                    dict_crop_rot_param[t]["updated_crop"] = [cx, cy, cz]
                end


                path_base = joinpath(path_dir_nrrd_crop, bname)
                path_nrrd = path_base *".nrrd"
                path_png = joinpath(path_dir_MIP_crop, bname *".png")
                
                write_nrrd(path_nrrd, img_crop, (spacing_lat, spacing_lat, spacing_axi))
                save(path_png, clamp01nan.(maxprj(img_crop, dims=3) ./ 1000))
            end

            # still crop out-of-focus data but give error message
            if crop_z[1] <= 1 || crop_z[2] >= size(img)[3]
                append!(focus_issues, t)
            end
        catch e_
            dict_error[t] = e_
        end
    end

    if length(keys(dict_error)) != 0
        @warn "Worm could not be detected or cropped at some time points."
    end

    if length(focus_issues) > 0
        @warn "Worm potentially out of focus in $(length(focus_issues))/$(length(t_range)) time points"
    end
    
    return dict_error, focus_issues
end

"""
    crop_rotate!(
        param_path::Dict, param::Dict, t_range, ch_list, dict_crop_rot_param::Dict; save_MIP::Bool=true,   
        nrrd_dir_key::String="path_dir_nrrd_shearcorrect", 
        nrrd_crop_dir_key::String="path_dir_nrrd_crop", mip_crop_dir_key::String="path_dir_MIP_crop"
    )

Crops and rotates a set of images.

# Arguments

 - `param_path::Dict`: Dictionary containing paths to directories and a `get_basename` function that returns NRRD file names.
 - `param::Dict`: Dictionary containing parameters, including:
    - `crop_threshold_intensity`: minimum number of standard deviations above the mean that a pixel must be for it to be categorized as part of a feature
    - `crop_threshold_size`: minimum size of a feature for it to be categorized as part of the worm
    - `spacing_axi`: Axial (z) spacing of the pixels, in um
    - `spacing_lat`: Lateral (xy) spacing of the pixels, in um
 - `t_range`: Time points to crop
 - `ch_list`: Channels to crop
 - `dict_crop_rot_param::Dict`: For each time point, the cropping parameters to use for that time point.
    If the cropping parameters at a time point are not found, they will be stored in the dictionary, modifying it.
 - `save_MIP::Bool` (optional): Whether to save png files. Default `true`
 - `nrrd_dir_key::String` (optional): Key in `param_path` to directory to input NRRD files. Default `path_dir_nrrd_shearcorrect`
 - `nrrd_crop_dir_key::String` (optional): Key in `param_path` to directory to output NRRD files. Default `path_dir_nrrd_crop`
 - `mip_crop_dir_key::String` (optional): Key in `param_path` to directory to output MIP files. Default `path_dir_MIP_crop`
"""
function crop_rotate!(param_path::Dict, param::Dict, t_range, ch_list, dict_crop_rot_param::Dict; save_MIP::Bool=true,
        nrrd_dir_key::String="path_dir_nrrd_shearcorrect", nrrd_crop_dir_key::String="path_dir_nrrd_crop", mip_crop_dir_key::String="path_dir_MIP_crop")
    path_dir_nrrd = param_path[nrrd_dir_key]
    path_dir_nrrd_crop = param_path[nrrd_crop_dir_key]
    path_dir_MIP_crop = param_path[mip_crop_dir_key]
    threshold_size = param["crop_threshold_size"]
    threshold_intensity = param["crop_threshold_intensity"]
    spacing_axi = param["spacing_axi"]
    spacing_lat = param["spacing_lat"]
    f_basename = param_path["get_basename"]

    crop_rotate!(path_dir_nrrd, path_dir_nrrd_crop, path_dir_MIP_crop, t_range, ch_list, dict_crop_rot_param,
        threshold_size, threshold_intensity, spacing_axi, spacing_lat, f_basename, save_MIP)
end
