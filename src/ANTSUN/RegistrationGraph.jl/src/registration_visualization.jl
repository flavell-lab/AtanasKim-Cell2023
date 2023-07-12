"""
    plot_centroid_match(fixed_img, matches, centroids_actual, centroids_inferred)

Plots fixed, and inferred moving, centroids over the fixed worm image.
Additionally, draws lines between matched centroids.

# Arguments

- `fixed_image`: fixed image of the worm (2D)
- `matches`: List of pairs of centroids (which are tuples of (x,y) coordinates) that match.
- `centroids_actual`: centroids determined directly from the fixed image
- `centroids_inferred`: centroids determined from the moving image, and then mapped onto the fixed image via registration.
"""
function plot_centroid_match(fixed_img, matches, centroids_actual, centroids_inferred)
    Plots.heatmap(fixed_img, fillcolor=:viridis, aspect_ratio=1)
    x1 = [cent[1] for cent in centroids_inferred]
    x2 = [cent[1] for cent in centroids_actual]
    y1 = [cent[2] for cent in centroids_inferred]
    y2 = [cent[2] for cent in centroids_actual]
    Plots.scatter!(x1, y1, color="blue", label="Inferred", seriesalpha=0.5)
    Plots.scatter!(x2, y2, color="red", label="Actual", seriesalpha=0.5)
    match_x = []
    match_y = []
    for match in matches
        if match[1][1] == match[2][1] && match[1][2] == match[2][2]
            push!(match_x, match[1][1])
            push!(match_y, match[1][2])
        else
            Plots.plot!(Plots.Shape([(match[1][1], match[1][2]), (match[2][1], match[2][2])]), linecolor="green", label=false)
        end
    end
    Plots.scatter!(match_x, match_y, color="red", seriesalpha=0, label="Matched")
end


"""
    view_roi_regmap(img_roi, img_roi_regmap; color_brightness::Real=0.3, plot_size=(600,600))

Plots instance segmentation image and registration-mapped instance segmentation
on the same plot, where each object is given a different color, the original image is shades of red,
and the registration-mapped image is shades of green.

# Arguments

- `img_roi`: 3D instance segmentation image
- `img_roi_regmap`: 3D instance segmentation image from another frame, mapped via registration.

# Optional keyword arguments

- `color_brightness::Real`: minimum RGB value (out of 1) that an object will be plotted with. Default 0.3
- `plot_size`: size of the plot. Default (600,600)
"""
function view_roi_regmap(img_roi, img_roi_regmap; color_brightness::Real=0.3, plot_size=(600,600))
    img_plot = gen_regmap_rgb(img_roi, img_roi_regmap; color_brightness=color_brightness)
    s = size(img_plot)
    @manipulate for z=1:s[3]
        Plots.plot(img_plot[:,:,z], size=plot_size)
    end
end

"""
    gen_regmap_rgb(img_roi, img_roi_regmap; color_brightness::Real=1)

Generates a colormap that encodes the difference between the ROI image and the registration-mapped version using shades of red and green.

# Arguments

- `img_roi`: ROI image in the fixed frame
- `img_roi_regmap`: moving frame ROI image registration-mapped to the fixed frame

# Optional keyword arguments

 - `color_brightness::Real`: maximum brightness of ROIs. Default 1.
"""
function gen_regmap_rgb(img_roi, img_roi_regmap; color_brightness::Real=1)
    a = [min(size(img_roi)[i], size(img_roi_regmap)[i]) for i=1:3]
    s = (a[1], a[2], a[3])
    num = maximum(img_roi)+1
    red = [color_brightness+(1-color_brightness)*rand() for i=1:num]
    red[1] = 0
    num = maximum(img_roi_regmap)+1
    green = [color_brightness+(1-color_brightness)*rand() for i=1:num]
    green[1] = 0
    return map(x->RGB.(red[img_roi[x]+1], green[img_roi_regmap[x]+1], 0), CartesianIndices(s))
end

"""
    visualize_roi_predictions(
        img_roi, img_roi_regmap, img, img_regmap;
        color_brightness::Real=0.3, plot_size=(600,600), roi_match=Dict(), unmatched_color=nothing, 
        make_rgb=false, highlight_rois=[], highlight_regmap_rois=[], highlight_color=RGB.(0,0,1), 
        contrast::Real=2, semantic::Bool=false, z_offset::Integer=0
    )

Visualizes a comparison between an ROI image and a registration-mapped version.

# Arguments

 - `img_roi`: ROI image in the fixed frame
 - `img_roi_regmap`: moving frame ROI image registration-mapped to the fixed frame
 - `img`: raw image in the fixed frame
 - `img_regmap`: raw image in the moving frame

# Optional keyword arguments

 - `color_brightness::Real`: brightness of ROI colors. Default 0.3.
 - `plot_size`: size of plot. Default (600, 600)
 - `roi_match`: matches between ROIs in the two frames, as a dictionary whose keys are ROIs in the moving frame
        and whose values are the corresponding ROIs in the fixed frame.
 - `unmatched_color`: If set, all ROIs in the moving frame without a corresponding match in the fixed frame will be this color.
 - `make_rgb`: If set, will generate red vs green display of ROI locations.
 - `highlight_rois`: A list of ROIs in the fixed frame to be highlighted.
 - `highlight_regmap_rois`: A list of ROIs in the moving frame to be highlighted.
 - `highlight_color`: Highlight color, as an RGB. Default blue, aka `RGB.(0,0,1)`
 - `contrast::Real`: Contrast of raw images. Default 2.
 - `semantic::Bool`: If set to true, `img_regmap` should instead be a semantic segmentation of the fixed frame image. Default false.
 - `z_offset::Integer`: z-offset of the moving image relative to the fixed image; the moving image will be shifted towards z=0 by this amount.
"""
function visualize_roi_predictions(img_roi, img_roi_regmap, img, img_regmap;
    color_brightness::Real=0.3, plot_size=(600,600), roi_match=Dict(), unmatched_color=nothing, make_rgb=false,
    highlight_rois=[], highlight_regmap_rois=[], highlight_color=RGB.(0,0,1), contrast::Real=2, semantic::Bool=false, z_offset::Integer=0)
    plot_imgs = []

    max_img = maximum(img)
    push!(plot_imgs, map(x->RGB.(contrast*x/max_img, contrast*x/max_img, contrast*x/max_img), img))

    if semantic
        push!(plot_imgs, map(x->RGB.(x,x,x), img_regmap))
    else
        max_img = maximum(img_regmap)
        push!(plot_imgs, map(x->RGB.(contrast*x/max_img, contrast*x/max_img, contrast*x/max_img), img_regmap))
    end
        

    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    if unmatched_color != nothing
        for r=2:num
            if !(r-1 in values(roi_match))
                colors[r] = unmatched_color
            end
        end
    end
    for roi in highlight_rois
        colors[roi+1] = highlight_color
    end
    push!(plot_imgs, map(x->colors[x+1], img_roi))

    num = maximum(img_roi_regmap)+1
    colors_regmap = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors_regmap[1] = RGB.(0,0,0)
    for (roi_regmap, roi) in roi_match
        colors_regmap[roi_regmap+1] = colors[roi+1]
    end
    if unmatched_color != nothing
        for r=2:num
            if !(r-1 in keys(roi_match))
                colors_regmap[r] = unmatched_color
            end
        end
    end
    for roi in highlight_regmap_rois
        colors_regmap[roi+1] = highlight_color
    end
    push!(plot_imgs, map(x->colors_regmap[x+1], img_roi_regmap))

    n_imgs = 4
    if make_rgb
        push!(plot_imgs, gen_regmap_rgb(img_roi, img_roi_regmap; color_brightness=1))
        n_imgs = 5
    end

    m = size(plot_imgs[1])[3]
    @manipulate for z=1:m
        make_plot_grid([(i == 1) || (i == 3) ? plot_imgs[i][:,:,min(max(z-z_offset,1),m)] : plot_imgs[i][:,:,z] for i in 1:length(plot_imgs)], n_imgs, plot_size)
    end
end


"""
    make_rgb_arr(red, green, blue)

Makes a `PyPlot`-compatible RGB array out of `red`, `green`, and `blue` channels.
"""
function make_rgb_arr(red, green, blue)
    s = size(red)
    @assert(s == size(green))
    @assert(s == size(blue))
    @assert(maximum(red) <= 1)
    @assert(maximum(green) <= 1)
    @assert(maximum(blue) <= 1)
    rgb_stack = zeros(append!(collect(s), 3)...)
    for x=1:s[1]
        for y=1:s[2]
            rgb_stack[x,y,1] = red[x,y]
            rgb_stack[x,y,2] = green[x,y]
            rgb_stack[x,y,3] = blue[x,y]
        end
    end
    return rgb_stack
end

"""
    make_diff_pngs(param_path::Dict, param::Dict, get_basename::Function,
        fixed::Integer, moving::Integer; proj_dim::Integer=3,
        fixed_ch_key::String="ch_marker", regdir_key::String="path_dir_reg",
        nrrd_key="path_dir_nrrd_filt", result::Union{Nothing,String}=nothing,
        contrast_f::Real=1, contrast_m::Real=1, swap_colors::Bool=false
    )

Makes PNG files to visualize how well the registration worked, by overlaying fixed and moving images.
The fixed image will be red and the moving image will be green, so yellow indicates good registration.

# Arguments
- `param_path::Dict`: Dictionary containing paths to files
- `param::Dict`: Dictionary containing parameter setting `reg_n_resolution`, an array of registration resolutions for each parameter file (eg for affine regstration with 3 resolutions
    and bspline registration with 4 resolutions, this would be `[3,4]`)
- `get_basename::Function`: Function mapping two timepoints to the base NRRD filename corresponding to them.
- `fixed::Integer`: timestamp (frame number) of fixed image
- `moving::Integer`: timestamp (frame number) of moving image

# Optional keyword arguments
- `proj_dim::Integer`: Dimension to project data. Default 3 (z-dimension)
- `fixed_ch_key::Integer`: Key in `param` to channel for the fixed image. Default `ch_marker`. (The moving image is the image automatically generated from the registration.)
- `regdir_key::String`: Key in `param_path` corresponding to the registration directory. Default `path_dir_reg`.
- `nrrd_key::String`: Key in `param_path` corresponding to the NRRD directory. Default `path_dir_nrrd_filt`.
- `result::String`: Name of resulting file. If left as default it the same as the corresponding NRRD file.
- `contrast_f::Real`: Contrast of fixed image portion of PNG. Default 1.
- `contrast_m::Real`: Contrast of moving image portion of PNG. Default 1.
- `swap_colors::Bool`: If set to `true`, fixed image will be green and moving image will be red.
"""
function make_diff_pngs(param_path::Dict, param::Dict, get_basename::Function,
    fixed::Integer, moving::Integer; proj_dim::Integer=3,
    fixed_ch_key::String="ch_marker", regdir_key::String="path_dir_reg",
    nrrd_key="path_dir_nrrd_filt", result::Union{Nothing,String}=nothing,
    contrast_f::Real=1, contrast_m::Real=1, swap_colors::Bool=false)
    fixed_ch = param[fixed_ch_key]
    reg_path = joinpath(param_path[regdir_key], string(moving) * "to" * string(fixed))
    path_fixed_img = joinpath(param_path[nrrd_key],
        get_basename(fixed, fixed_ch)*".nrrd")
    fixed_stack = dropdims(maximum(read_img(NRRD(path_fixed_img)),
            dims=proj_dim), dims=proj_dim)
    iters = param["reg_n_resolution"]
    error = false
    for i=1:length(iters)
        for j=1:iters[i]
            if isnothing(result)    
                img_path = reg_path * "/result." * string(i-1) * ".R" * string(j-1)
            else
                img_path = reg_path * "/" * result
            end
            path_nrrd = img_path * ".nrrd"
            path_png = img_path * "_"*string(proj_dim)*".png"
            try
                moving_stack = dropdims(maximum(read_img(NRRD(path_nrrd)), dims=proj_dim), dims=proj_dim)
                blank_stack = zeros(size(fixed_stack))
                fs_new = map(x->min(x,1), fixed_stack / maximum(fixed_stack) * contrast_f)
                ms_new = map(x->min(x,1), moving_stack / maximum(moving_stack) * contrast_m)
                if swap_colors
                    rgb_stack = make_rgb_arr(ms_new, fs_new, blank_stack)
                else
                    rgb_stack = make_rgb_arr(fs_new, ms_new, blank_stack)
                end
                imsave(path_png, rgb_stack, vmin=0);
            catch e
                error = true
            end
        end
    end
    return error
end

"""
    make_diff_pngs_base(
        param_path::Dict, param::Dict, get_basename::Function,
        fixed::Integer, moving::Integer; proj_dim::Integer=3,
        regdir_key::String="path_dir_reg", nrrd_key="path_dir_nrrd_filt",
        moving_ch_key::String="ch_marker", fixed_ch_key::String="ch_marker",
        contrast_f::Real=1, contrast_m::Real=1, swap_colors::Bool=false, png_name="noreg.png"
    )

Makes PNG files before registration, directly comparing two frames.
The fixed image will be red and the moving image will be green, so yellow indicates good registration.

# Arguments
- `param_path::Dict`: Dictionary containing paths to files
- `param::Dict`: Dictionary containing parameter settings
- `get_basename::Function`: Function mapping two timepoints to the base NRRD filename corresponding to them.
- `fixed::Integer`: timestamp (frame number) of fixed image
- `moving::Integer`: timestamp (frame number) of moving image

# Optional keyword arguments
- `proj_dim::Integer`: Dimension to project data. Default 3 (z-dimension)
- `regdir_key::String`: Key in `param_path` corresponding to the registration directory. Default `path_dir_reg`.
- `nrrd_key::String`: Key in `param_path` corresponding to the NRRD directory. Default `path_dir_nrrd_filt`.
- `moving_ch_key::Integer`: Key in `param` corresponding to channel for the moving image. Default `ch_marker`.
- `fixed_ch_key::Integer`: Key in `param` corresponding to channel for the fixed image. Default `ch_marker`.
- `contrast_f::Real`: Contrast of fixed image portion of PNG. Default 1.
- `contrast_m::Real`: Contrast of moving image portion of PNG. Default 1.
- `swap_colors::Bool`: If set to `true`, fixed image will be green and moving image will be red.
- `png_name::String`: Name of output file. Default `noreg.png`
"""
function make_diff_pngs_base(param_path::Dict, param::Dict, get_basename::Function,
    fixed::Integer, moving::Integer; proj_dim::Integer=3,
    regdir_key::String="path_dir_reg", nrrd_key="path_dir_nrrd_filt",
    moving_ch_key::String="ch_marker", fixed_ch_key::String="ch_marker",
    contrast_f::Real=1, contrast_m::Real=1, swap_colors::Bool=false, png_name="noreg.png")
    fixed_ch = param[fixed_ch_key]
    moving_ch = param[moving_ch_key]
    
    path_moving_img = joinpath(param_path[nrrd_key], get_basename(moving, moving_ch)*".nrrd")
    path_fixed_img = joinpath(param_path[nrrd_key], get_basename(fixed, fixed_ch)*".nrrd")
    path_ong = joinpath(param_path[regdir_key], string(moving) * "to" * string(fixed), png_name)

    fixed_stack = dropdims(maximum(read_img(NRRD(fixed_img_path)),
            dims=proj_dim), dims=proj_dim)
    moving_stack = dropdims(maximum(read_img(NRRD(moving_img_path)),
            dims=proj_dim), dims=proj_dim)
    s = size(fixed_stack)
    m = size(moving_stack)
    
    fixed_stack = map(x->min(x,1), fixed_stack[1:min(s[1],m[1]),1:min(s[2],m[2])] /
        maximum(fixed_stack) * contrast_f)
    moving_stack = map(x->min(x,1), moving_stack[1:min(s[1],m[1]),1:min(s[2],m[2])] /
        maximum(moving_stack) * contrast_m)
    blank_stack = zeros((min(s[1],m[1]),min(s[2],m[2])))
    if swap_colors
        rgb_stack = make_rgb_arr(moving_stack / maximum(moving_stack), fixed_stack /
            maximum(fixed_stack), blank_stack)
    else
        rgb_stack = make_rgb_arr(fixed_stack / maximum(fixed_stack), moving_stack /
            maximum(moving_stack), blank_stack)
    end
    imsave(png_path, rgb_stack, vmin=0);
end

"""
    nrrd_to_png(nrrd_path, png_path; proj_dim=3)

Converts an nrrd file at `nrrd_path::String` into a PNG file, saved to path `png_path::String`, using maximum intensity projection.
The optional argument `proj_dim` (default 3) can be changed to project in a different dimension.
"""
function nrrd_to_png(nrrd_path, png_path; proj_dim=3)
    stack = dropdims(maximum(read_img(NRRD(nrrd_path)), dims=proj_dim), dims=proj_dim)
    imsave(png_path, stack, vmin=median(stack), cmap="gray")
end
