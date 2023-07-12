"""
    register_neurons_overlap(roi_inferred, roi, activity_inferred, activity)

Matches ROIs from registered time points together based on the overlap heuristic.

# Arguments
- `inferred_roi`: ROI image array from the moving volume that has had the registration applied to it (eg via `transformix`)
- `roi`: ROI image array from the fixed time point
- `inferred_activity`: array of activities of each ROI in the marker channel from the moving time point
- `activity`: array of activities of each ROI in the marker channel from the fixed time point

# Returns
- `overlapping_rois`: a dictionary of ROI matches between the moving and fixed volumes.
    The values of the dictionary are the amount of overlap as a fraction of total ROI size.
    Eg: if ROI 15 in the moving volume has size 100, ROI 10 in the fixed volume has size 80, and the overlap has size 50,
    `(15, 10) => (0.5, 0.625)` would be an entry in the dictionary.

- `activity_mismatch`: a dictionary of the differences between the normalized activity of the ROIs
    between the moving and fixed time points.
"""
function register_neurons_overlap(roi_inferred, roi, activity_inferred, activity)
    size_roi = size(roi)
    size_roi_inf = size(roi_inferred)
    
    size_crop = [min(size_roi[i], size_roi_inf[i]) for i=1:length(size_roi)]
    
    roi = roi[1:size_crop[1], 1:size_crop[2], 1:size_crop[3]]
    roi_inferred = roi_inferred[1:size_crop[1], 1:size_crop[2], 1:size_crop[3]]

    overlapping_rois = Dict()
    activity_mismatch = Dict()

    # normalize activities
    activity_inferred = activity_inferred ./ mean(activity_inferred)
    activity = activity ./ mean(activity)

    max_roi_inf = maximum(roi_inferred)
    max_roi = maximum(roi)

    idx_nz_roi = findall(roi .!= 0)
    idx_nz_roi_inferred = findall(roi_inferred .!= 0)

    count_overlap = zeros(max_roi_inf, max_roi)

    n_roi = zeros(max_roi)
    n_roi_inferred = zeros(max_roi_inf)
    for i = idx_nz_roi
        roi_id = roi[i]
        n_roi[roi_id] += 1
    end

    for i = idx_nz_roi_inferred
        roi_inferred_id = roi_inferred[i]
        if roi_inferred_id > 0
            n_roi_inferred[roi_inferred_id] += 1
            roi_id = roi[i]
            if roi_id > 0
                count_overlap[roi_inferred_id,roi_id] += 1
            end
        end
    end

    for i = 1:max_roi_inf
        nz = count_overlap[i,:]
        for j = findall(nz .> 0)
            overlapping_rois[(i,j)] = (nz[j] / n_roi_inferred[i], nz[j] / n_roi[j])
            activity_mismatch[(i,j)] = abs(activity[j] - activity_inferred[i])
        end
    end
    
    return (overlapping_rois, activity_mismatch)
end

"""
    centroids_to_roi(img_roi)

Given an ROI image array `img_roi`, returns a dictionary whose keys are centroids of ROIs, with values the corresponding ROI.
"""
function centroids_to_roi(img_roi)
    centroids = Dict()
    indices = CartesianIndices(img_roi)
    for i=1:maximum(img_roi)
        total = sum(img_roi .== i)
        if total == 0
            continue
        end
        centroids[map(x->round(x), Tuple(sum((img_roi .== i) .* indices))./total)] = i
    end
    return centroids
end


function make_regmap_matrix(roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict;
        activity_diff_threshold::Real=0.3, watershed_error_penalty::Real=0.5, metric = "NCC", self_weight=0.5,
        size_mismatch_penalty=2, watershed_errors::Union{Nothing,Dict}=nothing, max_fixed_t::Int=0)
    label_map = Dict()
    regmap_matrix = [Array{Int64,1}(), Array{Int64,1}(), Array{Float64,1}()]
    label_weight = Dict()
    count = 1
    for (moving, fixed) in keys(roi_overlaps)
        moving_shifted = moving + max_fixed_t
        if !(moving_shifted in keys(label_map))
            label_map[moving_shifted] = Dict()
        end
        if !(fixed in keys(label_map))
            label_map[fixed] = Dict()
        end
        for (roi_moving, roi_fixed) in keys(roi_overlaps[(moving, fixed)])
            if !(roi_moving in keys(label_map[moving_shifted]))
                label_map[moving_shifted][roi_moving] = count
                count += 1
            end
            if !(roi_fixed in keys(label_map[fixed]))
                label_map[fixed][roi_fixed] = count
                count += 1
            end
            # weight of an edge is how well the ROIs overlap
            match_weight = minimum(roi_overlaps[(moving, fixed)][(roi_moving, roi_fixed)]) ^ size_mismatch_penalty
            # penalize for mNeptune activity being different
            match_weight *= activity_diff_threshold / (activity_diff_threshold + roi_activity_diff[(moving, fixed)][(roi_moving, roi_fixed)])
            # penalize for bad registration quality between time points
            match_weight /= (q_dict[(moving, fixed)][best_reg[(moving, fixed)]][metric])
            if !isnothing(watershed_errors) && roi_moving in watershed_errors[moving_shifted]
                match_weight *= watershed_error_penalty
            end
            if !isnothing(watershed_errors) && roi_fixed in watershed_errors[fixed]
                match_weight *= watershed_error_penalty
            end
            push!(regmap_matrix[1], label_map[moving_shifted][roi_moving])
            push!(regmap_matrix[2], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
            push!(regmap_matrix[2], label_map[moving_shifted][roi_moving])
            push!(regmap_matrix[1], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
        end
    end
    for i in 1:length(regmap_matrix[1])
        roi1 = regmap_matrix[1][i]
        label_weight[roi1] = get(label_weight, roi1, 0) + regmap_matrix[3][i]
    end
    m = mean(values(label_weight))
    if m == 0
        error("No successful registrations!")
    end
    for i in 1:length(regmap_matrix[1])
        roi1 = regmap_matrix[1][i]
        regmap_matrix[3][i] *= (1 - self_weight) / m
    end
    for i in 1:maximum(regmap_matrix[1])
        push!(regmap_matrix[1], i)
        push!(regmap_matrix[2], i)
        push!(regmap_matrix[3], self_weight)
    end

    regmap_matrix = sparse(regmap_matrix[1], regmap_matrix[2], regmap_matrix[3])
    return (regmap_matrix, label_map)
end


"""
    make_regmap_matrix(
        roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict, param::Dict;
        watershed_errors::Union{Nothing,Dict}=nothing, max_fixed_t::Int=0
    )

Assimilates ROIs from all time points and generates a quality-of-match matrix.

# Arguments

- `roi_overlaps::Dict`: a dictionary of dictionaries of ROI matches for each pair of moving and fixed time points
- `roi_activity_diff::Dict`: a dictionaon of dictionaries of the difference between normalized red ROI acivity, for each pair of moving and fixed time points
- `q_dict::Dict`: a dictionary of dictionaries of dictionaries of the registration quality for each metric for each resolution for each pair of moving and fixed time points
- `best_reg::Dict`: a dictionary of the registration that's being used, expressed as a tuple of resolutions, for each pair of moving and fixed time points
- `param::Dict`: a dictionary of parameter settings to use, including:
    * `activity_diff_threshold::Real`: parameter that controls how much red-channel activity differences are penalized in the matrix.
Smaller values = greater penalty. Default 0.3.
    * `watershed_error_penalty::Real`: parameter that control how much watershed/UNet errors are penalized in the matrix.
Smaller values = greater penalty. Default 0.5.
    * `quality_metric`: metric to use in the `q_dict` parameter. Default `NCC`.
    * `matrix_self_weight::Real`: amount of weight in the matrix to assign each ROI to itself. Default 0.5.
    * `size_mismatch_penalty::Real`: penalty to apply to overlapping ROIs that don't fully overlap.
Larger values = greater penalty. Default 2.

- `watershed_errors::Union{Nothing, Dict}`: a dictionary of ROIs that might have had watershed or UNet errors for every time point.
This dictionary must be pre-shifted if moving and fixed datasets are not the same. Ignored if nothing.

- `max_fixed_t::Int`: If the moving and fixed datasets are not the same, this number is added to each moving dataset time point
    to distinguish the datasets in the matrix and label map.

# Returns

- `regmap_matrix`, a matrix whose `(i,j)`th entry encodes the quality of the match between ROIs `i` and `j`.
- `label_map`: a dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.
"""
function make_regmap_matrix(roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict, param::Dict; watershed_errors::Union{Nothing,Dict}=nothing, max_fixed_t::Int=0)
    return make_regmap_matrix(roi_overlaps, roi_activity_diff, q_dict, best_reg,
        activity_diff_threshold=param["activity_diff_threshold"],
        watershed_error_penalty=param["watershed_error_penalty"],
        metric=param["quality_metric"],
        self_weight=param["matrix_self_weight"],
        size_mismatch_penalty=param["size_mismatch_penalty"],
        watershed_errors=watershed_errors,
        max_fixed_t=max_fixed_t)
end


"""
    pairwise_dist(regmap_matrix; threshold::Real=1e-16, dtype::Type=Float64)

Finds the distances between each pair of ROIs based on similarity between rows of the `regmap_matrix`.

# Arguments

- `regmap_matrix`: matrix of registration matches

# Optional keyword arguments

- `threshold::Real`: a small number added to the denominator to ensure there are no divide-by-zero errors. Default 1e-16.
- `dtype::Type`: Data type of matrix. Default Float64.
"""
function pairwise_dist(regmap_matrix; threshold::Real=1e-16, dtype::Type=Float64)
    d = regmap_matrix * transpose(regmap_matrix)
    l = length(regmap_matrix[:,1])
    r, c, v = findnz(d)
    for x=1:length(r)
        i = r[x]
        j = c[x]
        if i > j
            d[i,j] = -d[i,j] / sqrt(d[i,i] * d[j,j] + threshold)
            d[j,i] = d[i,j]
        end
    end
    for i=1:l
        d[i,i] = -1
    end
    return map(x->Float32(x), d)
end

"""
    delete_smeared_neurons(roi, threshold)

Deletes neurons that might come from huge deformations in registration from an `roi` (neurons that have more pixels than `threshold`).
"""
function delete_smeared_neurons(roi, threshold)
    roi_return = deepcopy(roi)
    list_roi_id = sort(unique(roi))
    list_roi_id = list_roi_id[2:end] # remove roi == 0
    
    for roi_id = list_roi_id
        if sum(roi_return .== roi_id) > threshold
            roi_return[roi_return .== roi_id] .= 0
        end
    end
        
    roi_return
end

"""
    update_label_map(label_map, matches)

Updates ROI label map `label_map` to include ROI matches `matches`, and returns updated version.
"""
function update_label_map(label_map, matches)
    new_label_map = Dict()
    for t in keys(label_map)
        new_label_map[t] = Dict()
        for roi in keys(label_map[t])
            new_label_map[t][roi] = matches[label_map[t][roi]]
        end
    end
    return new_label_map
end

"""
    invert_label_map(label_map)

Inverts ROI label map `label_map`, so that new ROIs map back to dictionaries mapping original time points to original ROIs.
"""
function invert_label_map(label_map)
    inverted_map = Dict()
    for t in keys(label_map)
        for roi in keys(label_map[t])
            if !(label_map[t][roi] in keys(inverted_map))
                inverted_map[label_map[t][roi]] = Dict()
            end
            if t in keys(inverted_map[label_map[t][roi]])
                push!(inverted_map[label_map[t][roi]][t], roi)
            else
                inverted_map[label_map[t][roi]][t] = [roi]
            end
        end
    end
    return inverted_map
end



function find_neurons(regmap_matrix, label_map; overlap_threshold::Real=0.005, height_threshold::Real=-0.01, dtype::Type=Float64, pair_match::Bool=false)
    inv_map = invert_label_map(label_map)
    dist = pairwise_dist(regmap_matrix, dtype=dtype)
    hmer = hclust_minimum_threshold(dist, inv_map, overlap_threshold, pair_match=pair_match)
    n = length(keys(inv_map))
    c_to_roi = Dict()
    n_to_c = [-i for i=1:n]
    for i in 1:n
        c_to_roi[-i] = [i]
    end

    @showprogress for i = 1:length(hmer.heights)
        if hmer.heights[i] >= height_threshold
            break
        end
        merged = copy(c_to_roi[hmer.mleft[i]])
        append!(merged, c_to_roi[hmer.mright[i]])
        c_to_roi[i] = merged
        merged_rois = sort(map(x->collect(keys(inv_map[x]))[1], merged))
        for j in merged
            n_to_c[j] = i
        end
    end
    new_label_map = update_label_map(label_map, n_to_c)
    new_inv_map = invert_label_map(new_label_map);
    return (new_label_map, new_inv_map, hmer)
end

"""
    find_neurons(regmap_matrix, label_map::Dict, param::Dict)

Groups ROIs into neurons based on a matrix of overlaps.

# Arguments:
- `regmap_matrix`: Matrix of distances between the ROIs
- `label_map::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.
- `param::Dict`: Dictionary containing `cluster_overlap_thresh` and `cluster_height_thresh` parameter settings to use for clustering. 

# Returns

- `new_label_map`: Dictionary of dictionaries mapping original ROIs to neuron labels, for each time point.
- `inv_map`: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label
- `hmer`: Raw clusters of the dataset.
"""
function find_neurons(regmap_matrix, label_map::Dict, param::Dict)
    return find_neurons(regmap_matrix, label_map,
        overlap_threshold=param["cluster_overlap_thresh"],
        height_threshold=param["cluster_height_thresh"])
end

"""
    match_neurons_across_datasets(label_map_1::Dict, label_map_2::Dict, inv_map_reg::Dict, max_fixed_t::Int)

Matches neurons across multiple datasets.

# Arguments
- `label_map_1::Dict`: Label map of ROI/time points to neuron ID for dataset 1.
- `label_map_2::Dict`: Label map of ROI/time points to neuron ID for dataset 2.
- `inv_map-reg::Dict`: Map of neuron ID to ROI/time points for the registration between datasets. Time points in dataset 2 are shifted by `max_fixed_t`.
- `max_fixed_t::Int`: Maximum time point in the first dataset.
"""
function match_neurons_across_datasets(label_map_1::Dict, label_map_2::Dict, inv_map_reg::Dict, max_fixed_t::Int)
    matches_12 = Dict()
    matches_21 = Dict()
    @showprogress for neuron in keys(inv_map_reg)
        if length(keys(inv_map_reg[neuron])) < 2
            continue
        end
        # we have a match between a neuron in each frame
        neuron1 = []
        neuron2 = []
        for t in keys(inv_map_reg[neuron])
            if t <= max_fixed_t
                if inv_map_reg[neuron][t][1] in keys(label_map_1[t])
                    append!(neuron1, label_map_1[t][inv_map_reg[neuron][t][1]])
                end
            elseif inv_map_reg[neuron][t][1] in keys(label_map_2[t - max_fixed_t])
                append!(neuron2, label_map_2[t - max_fixed_t][inv_map_reg[neuron][t][1]])
            end
        end
        if length(neuron1) == 0 || length(neuron2) == 0
            continue
        end
        for n1 in neuron1, n2 in neuron2
            if !(n1 in keys(matches_12))
                matches_12[n1] = Dict()
            end
            if !(n2 in keys(matches_12[n1]))
                matches_12[n1][n2] = 0
            end
            matches_12[n1][n2] += 1
            if !(n2 in keys(matches_21))
                matches_21[n2] = Dict()
            end
            if !(n1 in keys(matches_21[n2]))
                matches_21[n2][n1] = 0
            end
            matches_21[n2][n1] += 1
        end
    end
    return (matches_12, matches_21)
end

"""
    register_immobilized_rois(regmap_matrix, label_map_regmap, inv_map_regmap, label_map_freelymoving, valid_rois_freelymoving, param, reg_timept)

Registers immobilized ROIs to the freely moving dataset.

# Arguments:
- `regmap_matrix`: Matrix of distances between the ROIs
- `label_map_regmap::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels for the freely-moving-to-immobilized registration.
- `inv_map_regmap::Dict`: Dictionary of dictionaries mapping new ROI labels to original ROIs, for the freely-moving-to-immobilized registration.
- `label_map_freelymoving::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels in the freely-moving dataset (one of the outputs of `find_neurons`).
- `valid_rois_freelymoving`: List of which ROIs correspond to neurons in the freely-moving dataset.
- `param::Dict`: Dictionary containing `max_t` parameter setting to use for registration.
- `reg_timept`: Time point of the immobilized dataset to register.

# Returns:
- `roi_matches`: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs.
- `inv_matches`: Dictionary of dictionaries mapping matched freely moving ROIs to immobilized ROIs.
- `roi_matches_best`: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs, with only the best match for each immobilized ROI.
- `roi_match_confidence`: Dictionary of dictionaries mapping immobilized ROIs to the confidence of the match.
"""
function register_immobilized_rois(regmap_matrix, label_map_regmap, inv_map_regmap, label_map_freelymoving,
        valid_rois_freelymoving, param, reg_timept)

    s = size(regmap_matrix)
    roi_matches = Dict()
    inv_matches = Dict()
    t_imm = param["max_t"] + reg_timept

    for roi_imm in keys(label_map_regmap[t_imm])
        roi_matches[roi_imm] = Dict()
        roi_imm_label = label_map_regmap[t_imm][roi_imm]
        for roi_moving_label = 1:s[2]
            if (regmap_matrix[roi_imm_label, roi_moving_label] <= 0) || roi_moving_label == roi_imm_label
                continue
            end
            dict_moving = inv_map_regmap[roi_moving_label]
            @assert(length(keys(dict_moving)) == 1, "Inverse regmap matrix map must be 1:1 in time.")
            t_moving = collect(keys(dict_moving))[1]
            @assert(length(collect(values(dict_moving))[1]) == 1, "Inverse regmap matrix must be 1:1 in ROIs.")
            roi_moving = collect(values(dict_moving))[1][1]
            
            if !(t_moving in keys(label_map_freelymoving)) ||
                    !(roi_moving in keys(label_map_freelymoving[t_moving]))
                continue
            end
            
            neuron_label_moving = label_map_freelymoving[t_moving][roi_moving]
            neuron_moving = findall(n->n==neuron_label_moving, valid_rois_freelymoving)

            # the ROI we registered to was not succesfully mapped in freely-moving dataset
            if length(neuron_moving) == 0
                continue
            end
            
            @assert(length(neuron_moving)==1, "Number of neurons per label must be 1.")
            neuron_moving = neuron_moving[1]

            if neuron_moving in keys(roi_matches[roi_imm])
                roi_matches[roi_imm][neuron_moving] += regmap_matrix[roi_imm_label, roi_moving_label]
            else
                roi_matches[roi_imm][neuron_moving] = regmap_matrix[roi_imm_label, roi_moving_label]
            end

            if !(neuron_moving in keys(inv_matches))
                inv_matches[neuron_moving] = Dict()
            end
            if roi_imm in keys(inv_matches[neuron_moving])
                inv_matches[neuron_moving][roi_imm] += regmap_matrix[roi_imm_label, roi_moving_label]
            else
                inv_matches[neuron_moving][roi_imm] = regmap_matrix[roi_imm_label, roi_moving_label]
            end
        end
    end

    roi_match_best = zeros(Int, maximum(keys(roi_matches)))
    roi_match_confidence = zeros(maximum(keys(roi_matches)))
    for roi in keys(roi_matches)
        if length(keys(roi_matches[roi])) == 0
            continue
        end
        s1 = sum(values(roi_matches[roi]))
        for neuron in keys(roi_matches[roi])
            s2 = sum(values(inv_matches[neuron])) 
            confidence1 = 2 * roi_matches[roi][neuron] - s1
            confidence2 = 2 * inv_matches[neuron][roi] - s2
            confidence = min(confidence1, confidence2)
            if confidence > 0
                @assert(roi_match_best[roi] == 0, "Cannot have multiple confident matches")
                roi_match_best[roi] = Int(neuron)
                roi_match_confidence[roi] = confidence
            end
        end
    end
    roi_matches, inv_matches, roi_match_best, roi_match_confidence
end
