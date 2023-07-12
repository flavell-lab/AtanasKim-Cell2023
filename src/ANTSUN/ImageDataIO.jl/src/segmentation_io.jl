"""
    load_training_set(h5_file::String)

Loads training (or validation dataset) in 3D dataset `h5_file::String`.
Returns `raw`, `label`, and `weight` fields.
"""
function load_training_set(h5_file::String)
    training_set = h5open(h5_file, "r")
    raw = read(training_set, "raw")[:,:,:]
    label = read(training_set, "label")[:,:,:]
    weight = read(training_set, "weight")[:,:,:]
    close(training_set)
    return (raw, label, weight)
end

"""
    load_predictions(h5_file::String; threshold=nothing)

Loads UNet predictions in 3D dataset `h5_file::String`.
If the file has a fourth dimension, assumes predictions are stored in its second entry.
Returns the portion of `predictions` field corresponding to foreground.
Can optionally set `threshold` to binarize predictions.
"""
function load_predictions(h5_file::String; threshold=nothing)
    prediction_set = h5open(h5_file, "r")
    predictions = read(prediction_set, "predictions")
    if length(size(predictions)) == 4
        predictions = predictions[:,:,:,2]
    end
    close(prediction_set)
    if threshold == nothing
        return predictions
    else
        return predictions .> threshold
    end
end

"""
    write_watershed_errors(watershed_errors, path)

Writes `watershed_errors` to `path`.
"""
function write_watershed_errors(watershed_errors, path)
    open(path, "w") do f
        for frame in keys(watershed_errors)
            s = ""
            for i=1:length(watershed_errors[frame])
                s = s*string(watershed_errors[frame][i])*","
            end
            write(f, "$(frame) $(s[1:end-1])\n")
        end
    end
end

"""
    read_watershed_errors(path)

Reads watershed errors from `path`.
"""
function read_watershed_errors(path)
    watershed_errors = Dict()
    open(path, "r") do f
        for line in eachline(f)
            s = split(line)
            frame = parse(Int32, s[1])
            if length(s) == 1
                watershed_errors[frame] = []
            else
                watershed_errors[frame] = map(x->parse(Int32, x), split(s[2], ","))
            end
        end
    end
    return watershed_errors
end


"""
```resample_img(img, scales; dtype="raw")```

Scales an image down by using linear interpolation for the raw image, and bkg-gap priority interpoalation for the labels.

# Arguments
- `img`: image to scale
- `scales`: array of factors to scale down (bin) by in each dimension. Must be positive. 

# Optional keyword arguments

- `dtype::String`: type of data - either "raw", "label", or "weight". Default raw.
"""
function resample_img(img, scales; dtype="raw")
    s = size(img)
    p = prod(scales)
    new_idx = Tuple(map(x->Int64(x), s .÷ scales))
    if dtype in ["raw", "label"]
        new_img = zeros(UInt16, new_idx)
    else
        new_img = zeros(new_idx)
    end
    for c1 in CartesianIndices(new_idx)
        prev_idx = (Tuple(c1) .- 1) .* scales .+ 1
        idx = prev_idx .+ scales
        min_idx = map(x->Int32(floor(x)), prev_idx)
        max_idx = map(x->(x == floor(x)) ? Int32(x - 1) : Int32(floor(x)), idx)
        if dtype == "label"
            tot = [0.0,0.0,0.0,0.0]
            for c2 in CartesianIndices(Tuple(collect((min_idx[i]:max_idx[i] for i=1:length(s)))))
                tot[img[c2]+1] += prod([min(c2[j] + 1, idx[j]) - max(prev_idx[j], c2[j]) for j=1:length(s)])
            end
            tot_scaled = tot ./ p
            if tot[4] >= 0.5 || tot_scaled[4] >= 0.5
                new_img[c1] = 3
            else
                a = argmax(tot[2:4])
                if tot[a+1] >= 1 || tot_scaled[a+1] >= 0.5
                    new_img[c1] = a
                else
                    new_img[c1] = 0
                end
            end
        else
            tot = 0
            for c2 in CartesianIndices(Tuple(collect((min_idx[i]:max_idx[i] for i=1:length(s)))))
                tot += img[c2] * prod([min(c2[j] + 1, idx[j]) - max(prev_idx[j], c2[j]) for j=1:length(s)])
            end
            if dtype == "raw"
                new_img[c1] = UInt16(round(tot / p))
            elseif dtype == "weight"
                new_img[c1] = tot / p
            end
        end
    end
    return new_img
end

