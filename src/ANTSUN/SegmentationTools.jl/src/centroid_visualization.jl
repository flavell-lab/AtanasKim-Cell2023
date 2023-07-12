"""
    centroids_to_img(imsize, centroids)

Given an image of size `imsize`, converts `centroids` into an image mask of that size.
"""
function centroids_to_img(imsize, centroids)
    return map(x->Tuple(x) in centroids, CartesianIndices(imsize))
end

"""
    view_roi_3D(
        raw, predicted, img_roi; color_brightness=0.3, plot_size=(600,600), axis=3,
        raw_contrast=1, labeled_neurons=[], label_colors=[], neuron_color=nothing, overlay_intensity=0
    )

Plots instance segmentation image `img_roi`, where each object is given a different color.
Can also plot raw data and semantic segmentation data for comparison.

# Arguments

- `raw`: 3D raw image. If set to `nothing`, it will not be plotted.
- `predicted`: 3D semantic segmentation image. If set to `nothing`, it will not be plotted.
- `img_roi`: 3D instance segmentation image

# Optional keyword arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
- `plot_size`: size of the plot
- `axis`: axis to project, default 3
- `raw_contrast`: contrast of raw image, default 1
- `labeled_neurons`: neurons that should have a specific color, as an array of arrays.
- `label_colors`: an array of colors, one for each array in `labeled_neurons`
- `neuron_color`: the color of non-labeled neurons. If not supplied, all of them will be random different colors.
- `overlay_intensity`: intensity of ROI overlay on raw image
"""
function view_roi_3D(raw, predicted, img_roi; color_brightness=0.3, plot_size=(600,600), axis=3,
        raw_contrast=1, labeled_neurons=[], label_colors=[], neuron_color=nothing, overlay_intensity=0)
    @assert(length(labeled_neurons) == length(label_colors))
    plot_imgs = []
    overlay_img = zeros(RGB, size(raw))
    if !isnothing(img_roi)
        num = maximum(img_roi)+1
        if isnothing(neuron_color)
            colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
        else
            colors = [neuron_color for i=1:num]
        end
        colors[1] = RGB.(0,0,0)
        for i in 1:length(labeled_neurons)
            for neuron in labeled_neurons[i]
                colors[neuron+1] = label_colors[i]
            end
        end
        if overlay_intensity > 0
            overlay_img = map(x->colors[x+1]*overlay_intensity, img_roi)
        end
    end

    if !isnothing(raw)
        max_img = maximum(raw)
        push!(plot_imgs, map(x->RGB.(x/max_img*raw_contrast, x/max_img*raw_contrast, x/max_img*raw_contrast), raw) .+ overlay_img .* overlay_intensity)
    end
    if !isnothing(predicted)
        push!(plot_imgs, map(x->RGB.(x,x,x), predicted))
    end
    if !isnothing(img_roi)
        push!(plot_imgs, map(x->colors[x+1], img_roi))
    end

    @manipulate for z=1:size(plot_imgs[1])[axis]
        i = [(dim == axis) ? z : Colon() for dim=1:3]
        make_plot_grid([img[i[1],i[2],i[3]] for img in plot_imgs], length(plot_imgs), plot_size)
    end
end


"""
    view_roi_2D(raw, predicted, img_roi; color_brightness=0.3, plot_size=(600,600))

Plots instance segmentation image `img_roi`, where each object is given a different color.
Can also plot raw data and semantic segmentation data for comparison.

# Arguments

- `raw`: 2D raw image. If set to `nothing`, it will not be plotted.
- `predicted`: 2D semantic segmentation image. If set to `nothing`, it will not be plotted.
- `img_roi`: 2D instance segmentation image

# Optional keyword arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
- `plot_size`: size of the plot
"""
function view_roi_2D(raw, predicted, img_roi; color_brightness=0.3, plot_size=(600,600))
    plot_imgs = []
    if !isnothing(raw)
        max_img = maximum(raw)
        push!(plot_imgs, map(x->RGB.(x/max_img, x/max_img, x/max_img), raw))
    end
    if !isnothing(predicted)
        push!(plot_imgs, map(x->RGB.(x,x,x), predicted))
    end
    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    push!(plot_imgs, map(x->colors[x+1], img_roi))
    make_plot_grid(plot_imgs, length(plot_imgs), plot_size)
end

"""
    view_centroids_3D(img, centroids)

Displays the centroids of an image.

# Arguments:
 - `img`: Image
 - `centriods`: Centroids of the image, to be superimposed on the image.
"""
function view_centroids_3D(img, centroids)
    @manipulate for z=1:size(img)[3]
        view_centroids_2D(img[:,:,z], centroids)
    end
end

"""
    view_centroids_2D(img, centroids)

Displays the centroids of an image.

# Arguments:

- `img`: Image
- `centriods`: Centroids of the image, to be superimposed on the image. They can be 3D; if they are, the first dimension will be ignored.
"""
function view_centroids_2D(img, centroids)
    Plots.heatmap(img, fillcolor=:viridis, aspect_ratio=1, showaxis=false, flip=false)
    Plots.scatter!(map(x->x[2], centroids), map(x->x[1], centroids), flip=false, seriesalpha=0.3)
end
