function moment(list_idx::Vector{CartesianIndex{2}}, p, q)
    M_pq = 0.
    for idx = list_idx
        x, y = idx[1], idx[2]
        M_pq += x ^ p * y ^ q
    end
    
    M_pq
end

function moment(list_idx::Vector{CartesianIndex{3}}, p, q, r)
    M_pqr = 0.
    for idx = list_idx
        x, y, z = idx[1], idx[2], idx[3]
        M_pqr += x ^ p * y ^ q * z ^ r
    end
    
    M_pqr
end

function weighted_moment(img_raw::AbstractArray{T,2}, list_idx::Vector{CartesianIndex{2}}, p, q) where T
    M_pq = 0.
    for idx = list_idx
        x, y = idx[1], idx[2]
        M_pq += x ^ p * y ^ q * img_raw[x, y]
    end
    
    M_pq
end

function weighted_moment(img_raw::AbstractArray{T,3}, list_idx::Vector{CartesianIndex{3}}, p, q) where T
    M_pqr = 0.
    for idx = list_idx
        x, y, z = idx[1], idx[2], idx[3]
        M_pqr += x ^ p * y ^ q * z ^ r * img_raw[x, y, z]
    end

    M_pqr
end


function centroid(list_idx::Vector{CartesianIndex{2}})
    M_0 = length(list_idx)
    x_bar = moment(list_idx, 1, 0) / M_0
    y_bar = moment(list_idx, 0, 1) / M_0

    (x_bar, y_bar)
end

function centroid(list_idx::Vector{CartesianIndex{3}})
    M_0 = length(list_idx)
    x_bar = moment(list_idx, 1, 0, 0) / M_0
    y_bar = moment(list_idx, 0, 1, 0) / M_0
    z_bar = moment(list_idx, 0, 0, 1) / M_0

    (x_bar, y_bar, z_bar)
end


function weighted_centroid(img_raw::AbstractArray{T,2}, list_idx::Vector{CartesianIndex{2}}) where T
    M_0 = weighted_moment(img_raw, list_idx, 0, 0)
    x_bar = weighted_moment(img_raw, list_idx, 1, 0) / M_0
    y_bar = weighted_moment(img_raw, list_idx, 0, 1) / M_0

    (x_bar, y_bar)
end

function weighted_centroid(img_raw::AbstractArray{T,3}, list_idx::Vector{CartesianIndex{3}}) where T
    M_0 = weighted_moment(img_raw, list_idx, 0, 0, 0)
    x_bar = weighted_moment(img_raw, list_idx, 1, 0, 0) / M_0
    y_bar = weighted_moment(img_raw, list_idx, 0, 1, 0) / M_0
    z_bar = weighted_moment(img_raw, list_idx, 0, 0, 1) / M_0

    (x_bar, y_bar, z_bar)
end

function get_centroids(img_roi)
    list_id = sort(unique(img_roi))[2:end]
    list_centroid = []
    for roi = list_id
        push!(list_centroid, centroid(findall(img_roi .== roi)))
    end
    
    list_centroid
end
    
function get_centroids_round(img_roi)
    list_id = sort(unique(img_roi))[2:end]
    list_centroid = []
    for roi = list_id
        push!(list_centroid, round.(centroid(findall(img_roi .== roi))))
    end
    
    list_centroid
end
