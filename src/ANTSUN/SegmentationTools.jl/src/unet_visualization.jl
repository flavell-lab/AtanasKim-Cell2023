"""
    view_label_overlay(img, label, weight; contrast::Real=1, label_intensity::Real=0.5)

Makes image of the raw data overlaid with a translucent label.

# Arguments

- `img`: raw data image (2D)
- `label`: labels for the image
- `weight`: mask of which label values to include. Pixels with a weight of 0 will be plotted in the
    raw, but not labeled, data; pixels with a weight of 1 will be plotted with raw data overlaid with label.

## Optional keyword arguments
- `contrast::Real`: Contrast factor for raw image. Default 1 (no adjustment)
- `label_intensity::Real`: Intensity of label, from 0 to 1. Default 0.5.
"""
function view_label_overlay(img, label, weight; contrast::Real=1, label_intensity::Real=0.5)
    img_gray = map(x->min(x, 1-label_intensity), img * contrast * (1 - label_intensity) ./ maximum(img))
    green = img_gray + [weight[x] == 0 ? 0 : (1-label[x]) * label_intensity for x in CartesianIndices(size(img))]
    red = img_gray + [weight[x] == 0 ? 0 : label[x] * label_intensity for x in CartesianIndices(size(img))]
    blue = img_gray
    rgb_stack = RGB.(red, green, blue)
    return rgb_stack
end

"""
    visualize_prediction_accuracy_2D(predicted, actual, weight)

Generates an image which compares the predictions of the neural net with the label.
Green = match, red = mismatch.
Assumes the predictions and labels are binary 2D arrays.

# Arguments

- `predicted`: neural net predictions
- `actual`: actual labels
- `weight`: pixel weights; weight of 0 is ignored and not plotted.
"""
function visualize_prediction_accuracy_2D(predicted, actual, weight)
    inaccuracy = map(abs, predicted - actual)
    green = [weight[x] == 0 ? 0 : 1 - inaccuracy[x] for x in CartesianIndices(size(inaccuracy))]
    red = [weight[x] == 0 ? 0 : inaccuracy[x] for x in CartesianIndices(size(inaccuracy))]
    blue = zeros(size(inaccuracy))
    rgb_stack = RGB.(red, green, blue)
    return rgb_stack
end


"""
    visualize_prediction_accuracy_3D(predicted, actual, weight)

Generates an image which compares the predictions of the neural net with the label.
Green = match, red = mismatch.
Assumes the predictions and labels are binary 3D arrays.

# Arguments

- `predicted`: neural net predictions
- `actual`: actual labels
- `weight`: pixel weights; weight of 0 is ignored and not plotted.
"""
function visualize_prediction_accuracy_3D(predicted, actual, weight)
    @manipulate for z=1:size(raw)[3]
        visualize_prediction_accuracy_2D(predicted[:,:,z], actual[:,:,z], weight[:,:,z])
    end
end


"""
    make_plot_grid(plots, cols::Integer, plot_size)

Makes grid out of many smaller plots. 

# Arguments

- `plots`: List of things to be plotted. Each item must be something that could be input to the `plot` function.
- `cols::Integer`: Number of columns in array of plots to be created
- `plot_size`: Size of resulting plot per row.
"""
function make_plot_grid(plots, cols::Integer, plot_size)
    if length(plots) < cols
        cols = length(plots)
    end
    rows = (length(plots)-1)÷cols+1
    fig = plot(layout=(rows, cols), size=(plot_size[1], plot_size[2]*rows))
    for (i,plt) in enumerate(plots)
        row = (i-1)÷cols + 1
        col = (i-1)%cols + 1
        plot!(fig[row,col], plt, aspect_ratio=1, showaxis=false, legend=false, flip=false)
    end
    return fig
end

"""
    display_predictions_2D(
        raw, label, weight, predictions_array; cols::Integer=7, plot_size=(1800,750), 
        display_accuracy::Bool=true, contrast::Real=1
    )

Compares multiple different neural network predictions of the raw dataset,
in comparison with the label and weight samples. The order of the plots is a plot of the raw
data, followed by a plot of the weights, followed by plots of raw predictions and prediction vs label
differential (in the order the predictions were specified in the array).

# Arguments

- `raw`: 2D raw dataset
- `label`: labels on raw dataset. Set to `nothing` to avoid displaying labels (for instance, on a testing datset).
- `weight`: weights on the labels. Set to `nothing` if you are not displaying labels.
- `predictions_array`: various predictions of the raw dataset.

# Optional keyword arguments

- `cols::Integer`: maximum number of columns in the plot. Default 7.
- `plot_size`: size of plot per row. Default (1800, 750).
- `display_accuracy::Bool`: whether to display prediction accuracy (green for match, red for mismatch). Default true.
- `contrast::Real`: contrast of raw image. Default 1.
"""
function display_predictions_2D(raw, label, weight, predictions_array; cols::Integer=7, plot_size=(1800,750), display_accuracy::Bool=true, contrast::Real=1)
    plots = []
    if label != nothing
        push!(plots, view_label_overlay(raw, label, weight, contrast=contrast))
        push!(plots, view_label_overlay(weight, label, weight, contrast=contrast))
    else
        gray = map(x->min(x, 1), raw * contrast ./ maximum(raw))
        push!(plots, RGB.(gray, gray, gray))
    end

    for predictions in predictions_array
        if label != nothing
            push!(plots, view_label_overlay(predictions, label, weight, contrast=contrast))
            if display_accuracy
                push!(plots, visualize_prediction_accuracy_2D(predictions, label, weight))
            end
        else
            gray = map(x->min(x, 1), predictions * contrast ./ maximum(predictions))
            push!(plots, RGB.(gray, gray, gray))
        end
    end
    return make_plot_grid(plots, cols, plot_size)
end

"""
    display_predictions_3D(
        raw, label, weight, predictions_array; cols::Integer=7,
        plot_size=(1800,750), axis=3, display_accuracy::Bool=true, contrast=1
    )

Compares multiple different neural network predictions of the raw dataset,
using an interactive slider to toggle between z-planes of the 3D dataset.

# Arguments

- `raw`: 3D raw dataset
- `label`: labels on raw dataset. Set to `nothing` to avoid displaying labels (for instance, on a testing datset).
- `weight`: weights on the labels. Set to `nothing` if you are not displaying labels.
- `predictions_array`: various predictions of the raw dataset.

# Optional keyword arguments

- `cols::Integer`: maximum number of columns in the plot. Default 7.
- `plot_size`: size of plot per row. Default (1800, 750).
- `axis`: axis to project, default 3
- `display_accuracy::Bool`: whether to display prediction accuracy (green for match, red for mismatch). Default true.
- `contrast::Real`: contrast of raw image. Default 1.
"""
function display_predictions_3D(raw, label, weight, predictions_array; cols::Integer=7, plot_size=(1800,750), axis=3, display_accuracy::Bool=true, contrast=1)
    @manipulate for z=1:size(raw)[axis]
        i = [(dim == axis) ? z : Colon() for dim=1:3]
        display_predictions_2D(getindex(raw, i[1], i[2], i[3]), (isnothing(label) ? nothing : getindex(label, i[1], i[2], i[3])),
            (isnothing(weight) ? nothing : getindex(weight, i[1], i[2], i[3])), [getindex(predictions, i[1], i[2], i[3]) for predictions in predictions_array]; cols=cols, plot_size=plot_size, display_accuracy=display_accuracy, contrast=contrast)
    end
end

"""
    compute_mean_iou(raw_file, prediction_file; threshold=0.5)

Computes the mean IOU between `raw_file` and a `prediction_file` HDF5 files.

By default, assumes a threshold of 0.5, but this can be changed with the `threshold` parameter.
"""
function compute_mean_iou(raw_file, prediction_file; threshold=0.5)
    h5open(raw_file) do actual
        h5open(prediction_file) do pred
            label = read(actual, "label")[:,:,:,1,1]
            weights = read(actual, "weight")[:,:,:,1,1]
            predictions = read(pred, "predictions")[:,:,:,2]
            p = predictions .> threshold
            l = map(x->convert(Bool, x), label)
            neuron_i = 0
            neuron_u = 0
            bkg_i = 0
            bkg_u = 0
            for idx in CartesianIndices(p)
                neuron_i += (p[idx] & l[idx]) * weights[idx]
                neuron_u += (p[idx] | l[idx]) * weights[idx]
                bkg_i += (~p[idx] & ~l[idx]) * weights[idx]
                bkg_u += (~p[idx] | ~l[idx]) * weights[idx]
            end
            return (neuron_i/neuron_u + bkg_i/bkg_u) / 2
        end
    end
end

