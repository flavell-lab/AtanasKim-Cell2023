mutable struct SegmentedInstance
    obj_id::Int
    centroid_idx_x::Int
    centroid_idx_y::Int
    area::Float64
    area_weighted::Float64 # intensity weighted area
    list_idx::Array{CartesianIndex{2},1}

    function SegmentedInstance(n::Int, list_idx::Array{CartesianIndex{2},1},
            img_raw, img_bin)
        x_bar, y_bar = weighted_centroid(img_raw, list_idx)
        area_weighted = weighted_moment(img_raw, list_idx, 0, 0)
        area = weighted_moment(img_bin, list_idx, 0, 0)

        new(n, Int(round(x_bar)), Int(round(y_bar)), area, area_weighted, list_idx)
    end
end

function get_segmented_instance(img_label, img_bin, img_raw)
    img_size = size(img_label)

    # total number of objects
    n_max = maximum(img_label)

    # hash table of pixel location for each object
    obj_hash = Dict{Int, Array{CartesianIndex{2}, 1}}()

    # populate the hash table
    for n = 1:n_max
        obj_hash[n] = findall(n .== img_label)
    end

    list_instance = []
    for n = 1:n_max
        list_idx = obj_hash[n]
        push!(list_instance, SegmentedInstance(n, list_idx, img_raw, img_bin))
    end

    list_instance
end
