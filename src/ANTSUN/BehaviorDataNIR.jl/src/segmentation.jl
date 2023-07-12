function downsample_unet_input(img)
    @assert(size(img) == (968, 732))
    img = Float32.(img[4:963, 4:723])
    (img[1:2:end-1,1:2:end-1] .+ img[2:2:end,2:2:end] .+
        img[1:2:end-1,2:2:end] .+ img[2:2:end,1:2:end-1]) ./ 4
end

function largest_segment!(img_bin, img_label)
    img_label .= 0
    label_2d!(img_bin, img_label)
    list_seg = get_segmented_instance(img_label, img_bin, img_bin)
    largest_obj_id = sort(list_seg, by=x->x.area, rev=true)[1].obj_id
    
    img_label .== largest_obj_id
end

"""
    segment_worm!(model, img_raw, img_label; θ=0.75)

Segment the worm given img (size: (968, 732)).

Arguments
---------
* `model`: unet model
* `img_raw`: raw img
* `img_label`: pre-allocated label array <:Int32. Size: (480,360)
Optional arguments
--------
* `θ=0.75`: unet output threshold
"""
function segment_worm!(model, img_raw, img_label; θ=0.75)
    @assert(eltype(img_label) == Int32)
    @assert(size(img_label) == (480,360))
    img_raw_ds = downsample_unet_input(img_raw)
    img_unet = eval_model(UNet2D.standardize(img_raw_ds), model)
    img_bin = closing(img_unet .> θ)
    
    img_raw_ds, largest_segment!(img_bin, img_label)
end


function get_nn_graph(xs, ys)
    pts_array = hcat(xs, ys)
    nn = py_skl_neighbors.NearestNeighbors(n_neighbors=2).fit(pts_array)
    g = nn.kneighbors_graph()
    g_mat = py_nx.from_scipy_sparse_array(g)
    
    g, g_mat
end

function order_pts(xs, ys)
    pts_array = hcat(xs, ys)
    g, g_mat = get_nn_graph(xs, ys)
    list_ord = [collect(py_nx.dfs_preorder_nodes(g_mat, i)) for i = 0:length(xs)-1]
    list_cost = zeros(length(xs))

    for i = 1:length(xs)
        list_cost[i] = sum((pts_array[list_ord[i][1:end-1] .+ 1, :] .- 
                pts_array[list_ord[i][2:end] .+ 1, :]) .^ 2)
    end
    
    list_ord[findmin(list_cost)[2]] .+ 1
end

function longest_shortest(param, xs, ys; prev_med_axis=nothing)
    g, g_mat = get_nn_graph(xs, ys)
    g_arr = g.toarray()
    
    if !isnothing(prev_med_axis)
        for edge in g_mat.edges
            pt = ((xs[edge[1]+1]+xs[edge[2]+1])/2, (ys[edge[1]+1]+ys[edge[2]+1])/2)
            m = minimum([euclidean_dist(pt, (prev_med_axis[1][p], prev_med_axis[2][p])) for p in 1:length(prev_med_axis[1])])
            if m > param["max_med_axis_delta"]
               g_arr[edge[1]+1,edge[2]+1] = 0
               g_arr[edge[2]+1,edge[1]+1] = 0
            end
            # don't include edges that have to hop over barriers
            if euclidean_dist((xs[edge[1]+1], ys[edge[1]+1]), (xs[edge[2]+1], ys[edge[2]+1])) > 1.5
                g_arr[edge[1]+1,edge[2]+1] = 0
                g_arr[edge[2]+1,edge[1]+1] = 0
            end
        end
        g_mat = py_nx.to_networkx_graph(g_arr)
    end
    
    # find terminal nodes
    idx_deg1 = findall(dropdims(sum(g_arr, dims=1), dims=1) .!= 2)
    
    # find longest shortest path for end nodes combinations
    list_paths = []
    for nodes = combinations(idx_deg1, 2)
        if py_nx.has_path(g_mat, nodes[1]-1, nodes[2]-1)
            shortest_path = py_nx.shortest_path(g_mat, nodes[1]-1, nodes[2]-1)
            push!(list_paths, (length(shortest_path), shortest_path))
        else
            push!(list_paths, (0, [0]))
        end
    end
    idx_long_short = findmax(map(x->x[1], list_paths))[2]
    
    list_paths[idx_long_short][2] .+ 1 # path_longest_shortest
end


function generate_img_reconstruction(prev_med_axis, worm_thickness, img_size; trim=0, thickness=1, pad=0, recent_spline_len=1)
    img_reconstruct = zeros(Bool, img_size)
    points_added = Dict()
    len = min(length(prev_med_axis[1]),length(worm_thickness))-trim
    for z in trim+1:len
        d = Int32(round(worm_thickness[z])) + thickness + pad
        for x=max(prev_med_axis[1][z]-d,1):min(prev_med_axis[1][z]+d,img_size[1])
            for y=max(prev_med_axis[2][z]-d,1):min(prev_med_axis[2][z]+d,img_size[2])
                ed = euclidean_dist((x,y),(prev_med_axis[1][z],prev_med_axis[2][z]))
                if ed < d && ed >= d - thickness
                    img_reconstruct[x,y] = true
                    if !((x,y) in keys(points_added))
                        points_added[(x,y)] = [z]
                    else
                        append!(points_added[(x,y)], z)
                    end
                end
            end
        end
    end
    
    for z in trim+1:len
        d = Int32(round(worm_thickness[z])) + thickness + pad
        for x=max(prev_med_axis[1][z]-d,1):min(prev_med_axis[1][z]+d,img_size[1])
            for y=max(prev_med_axis[2][z]-d,1):min(prev_med_axis[2][z]+d,img_size[2])
                ed = euclidean_dist((x,y),(prev_med_axis[1][z],prev_med_axis[2][z]))
                if ed < d - thickness && img_reconstruct[x,y] && all([abs(w - z) <= recent_spline_len for w in points_added[(x,y)]])
                    img_reconstruct[x,y] = false
                end
            end
        end
    end
    
    return img_reconstruct
end

function generate_med_axis_mask(param, prev_med_axis, worm_thickness, img_size)
    img_reconstruct = generate_img_reconstruction(prev_med_axis, worm_thickness, img_size, 
        trim=0, pad=param["worm_thickness_pad"], thickness=param["boundary_thickness"], recent_spline_len=param["close_pts_threshold"])
    img_reconstruct_trim = generate_img_reconstruction(prev_med_axis, worm_thickness, img_size, 
        trim=param["trim_head_tail"], pad=param["worm_thickness_pad"], thickness=param["boundary_thickness"], recent_spline_len=param["close_pts_threshold"])
    return Bool.(true .- (img_reconstruct .& img_reconstruct_trim))
end

function self_proximity_detector(param, xs, ys)
    for i=1:length(xs)
        for j=1:length(xs)
            if abs(j-i) < param["loop_dist_threshold"]
                continue
            end
            if euclidean_dist((xs[i],ys[i]), (xs[j],ys[j])) <= param["med_axis_self_dist_threshold"]
                return true
            end
        end
    end
    return false
end

function medial_axis_from_py(param, img_med_axis, pts_n, img_size; prev_med_axis=nothing)
    array_pts = cat(map(x->[x[2], x[1]], findall(img_med_axis))..., dims=2)
    xs = array_pts[2,:]
    ys = array_pts[1,:]
    pts_order = longest_shortest(param, xs, ys, prev_med_axis=prev_med_axis)

    # reorder points
    xs = xs[pts_order]
    ys = ys[pts_order]
    
    # find head/tail and flip
    dist_nose_1 = euclidean_dist((xs[1], ys[1]), pts_n[1:2])
    dist_nose_end = euclidean_dist((xs[end], ys[end]), pts_n[1:2])

    if dist_nose_1 > dist_nose_end
        reverse!(xs)
        reverse!(ys)
    end

    s = img_size

    # crop out points that hit the edge of frame
    crop_max = length(xs)
    for i=1:length(xs)
        if xs[i] == s[1] || ys[i] == s[2]
            crop_max = i
            break
        end
    end

    crop_min = 1
    # crop out points in front of the nose - don't do this if low confidence
    if pts_n[3] > param["nose_confidence_threshold"]
        min_dist = Inf
        # don't crop points too far away from nose
        for i=1:min(length(xs), param["nose_crop_threshold"])
            d = euclidean_dist((xs[i], ys[i]), pts_n[1:2])
            if d < min_dist
                min_dist = d
                crop_min = i
            end
        end
    end
    xs = xs[crop_min:crop_max]
    ys = ys[crop_min:crop_max]

    return xs, ys, pts_order
end

function medial_axis(param, img_bin, pts_n; prev_med_axis=nothing, prev_pts_order=nothing, worm_thickness=nothing)
    is_omega = false
    if !isnothing(prev_pts_order)
        # this code can crash if the worm is circular - in which case, use omega routine
        try
            xs, ys, pts_order, is_omega = medial_axis(param, img_bin, pts_n)
            if (length(pts_order) > length(prev_pts_order) - param["med_axis_shorten_threshold"]) && 
                    !self_proximity_detector(param, prev_med_axis[1], prev_med_axis[2])
                return xs, ys, pts_order, is_omega
            end
        catch e
            is_omega = true
            xs = prev_med_axis[1]
            ys = prev_med_axis[2]
            pts_order = prev_pts_order
        end
        
        # can't solve omega turns until we have worm thickness
        if isnothing(worm_thickness)
            return xs, ys, pts_order, is_omega
        end
        
        mask = generate_med_axis_mask(param, prev_med_axis, worm_thickness, size(img_bin))
        img_med_axis = py_ski_morphology.medial_axis(img_bin, mask=mask)
        img_med_axis[Bool.(true .- mask)] .= false
    else
        # medial axis extraction
        img_med_axis = py_ski_morphology.medial_axis(img_bin)
    end
    
    xs, ys, pts_order = medial_axis_from_py(param, img_med_axis, pts_n, size(img_bin), prev_med_axis=prev_med_axis)
    return xs, ys, pts_order, is_omega
end

function get_worm_thickness(img_bin, xs, ys)
    dt = distance_transform(feature_transform(img_bin))
    dists = [dt[xs[x], ys[x]] for x=1:length(xs)]
    return dists
end

function get_img_boundary(img_bin; thickness=1)
    boundary = zeros(Bool, size(img_bin))
    for x=1:size(boundary,1)
        for y=1:size(boundary,2)
            if !img_bin[x,y] && 
                any(img_bin[max(x-thickness,1):min(x+thickness,size(img_bin,1)),
                        max(y-thickness,1):min(y+thickness,size(img_bin,2))])
                boundary[x,y] = true
            end
        end
    end
    return boundary
end

"""
    compute_worm_spline!(
        param, path_h5, worm_seg_model, worm_thickness, med_axis_dict, pts_order_dict, is_omega_dict,
        x_array, y_array, nir_worm_angle, eccentricity; timepts="all"
    )

Compute the worm spline for a given set of parameters. Writes to most of its input parameters.

Arguments:
- `param`: A dictionary containing the parameters for the worm spline computation.
- `path_h5`: The path to the HDF5 file containing the worm data.
- `worm_seg_model`: The worm segmentation model.
- `worm_thickness`: The thickness of the worm.
- `med_axis_dict`: A dictionary containing where to write medial axis data.
- `pts_order_dict`: A dictionary containing where to write the order of spline points.
- `is_omega_dict`: A dictionary containing where to write values of whether the worm is self-intersecting.
- `x_array`: An array containing the x-coordinates of the worm splines. Will be modified.
- `y_array`: An array containing the y-coordinates of the worm splines. Will be modified.
- `nir_worm_angle`: The angle of the worm.
- `eccentricity`: The eccentricity of the worm.
- `timepts`: The timepoints to compute the worm spline for. Defaults to "all".

Returns:
- `img_label`: An array containing the labeled image.
- `errors`: A dictionary containing any errors that occurred during computation.
"""
function compute_worm_spline!(param, path_h5, worm_seg_model, worm_thickness, med_axis_dict, pts_order_dict, is_omega_dict,
        x_array, y_array, nir_worm_angle, eccentricity; timepts="all")

    spline_interval = 1/param["num_center_pts"]
    img_label = zeros(Int32, param["img_label_size"][1], param["img_label_size"][2])
    errors = Dict()

    f = h5open(path_h5)
    pos_feature, pos_feature_unet = read_pos_feature(f)
    close(f)


    rng = timepts
    if typeof(timepts) == String
        rng = 1:size(pos_feature_unet,3)
    end

    omega_flag = false

    @showprogress for idx in rng
        try
            if idx in keys(is_omega_dict) && is_omega_dict[idx]
                omega_flag = true
            end

            # initialize dictionaries in case of crash
            pts_order_dict[idx] = pts_order_dict[idx-1]
            med_axis_dict[idx] = med_axis_dict[idx-1]
            is_omega_dict[idx] = false
            pts = pos_feature_unet[:,:,idx]
            pts_n = pts[1, :]
            
            # don't trust nose if other points are inaccurate
            pts_n[3] = minimum(pts[:,3])

            f = h5open(path_h5)
            img_raw = f["img_nir"][:,:,idx]
            close(f)
            img_raw_ds, img_bin = segment_worm!(worm_seg_model, img_raw, img_label)

            img_bin = Int32.(img_bin)

            med_xs, med_ys, pts_order, is_omega = medial_axis(param, img_bin, pts_n,
                prev_med_axis=med_axis_dict[idx-1], prev_pts_order=pts_order_dict[idx-1], worm_thickness=worm_thickness)
            pts_order_dict[idx] = pts_order
            med_axis_dict[idx] = (med_xs, med_ys)
            is_omega_dict[idx] = is_omega

            if !is_omega
                omega_flag = false
            end
            
            spl_data, spl = fit_spline(param, med_xs, med_ys, pts_n, n_subsample=15)
            spl_pts = spl(0:spline_interval:1, 1:2)

            x_array[idx, :] = spl_pts[:, 1] # timept x spline points
            y_array[idx, :] = spl_pts[:, 2] # timept x spline points

            # get worm axis with PCA on binary image
            worm_pts = map(x->collect(Tuple(x)), filter(x->img_bin[x]!=0, CartesianIndices(img_bin)))
            worm_centroid = reduce((x,y)->x.+y, worm_pts) ./ length(worm_pts)
            deltas = map(x->collect(x.-worm_centroid), worm_pts)
            cov_mat = cov(deltas)
            mat_eigvals = eigvals(cov_mat)
            mat_eigvecs = eigvecs(cov_mat)
            eigvals_order = sortperm(mat_eigvals, rev=true)
            long_axis = mat_eigvecs[eigvals_order[1],:]
            angle = vec_to_angle(long_axis)[1]
            angle_cn = vec_to_angle(worm_centroid .- pts_n[1:2])[1]
            if abs(recenter_angle(angle - angle_cn)) > pi/2
                angle = recenter_angle(angle+pi)
            end
            nir_worm_angle[idx] = recenter_angle(angle)
            eccentricity[idx] = mat_eigvals[eigvals_order[1]] / mat_eigvals[eigvals_order[2]]
        catch e
            errors[idx] = e
        end
    end
    return errors
end

function compute_worm_thickness(param, path_h5, worm_seg_model, med_axis_dict, is_omega_dict)
    img_label = zeros(Int32, param["img_label_size"][1], param["img_label_size"][2])

    lengths = Dict()
    for x in keys(med_axis_dict)
        if !isnothing(med_axis_dict[x]) && (!(x in keys(is_omega_dict)) || !is_omega_dict[x])
            lengths[x] = length(med_axis_dict[x][1])
        end
    end
    low = Int32(floor(percentile(values(lengths), param["min_len_percent"])))
    high = percentile(values(lengths), param["max_len_percent"])

    dists = zeros(length([idx for idx in keys(lengths) if lengths[idx] >= low && lengths[idx] <= high]), low)

    count = 1
    @showprogress for idx in keys(lengths)
        if lengths[idx] >= low && lengths[idx] <= high
            f = h5open(path_h5)
            img_raw = f["img_nir"][:,:,idx]
            close(f)
            
            img_raw_ds, img_bin = segment_worm!(worm_seg_model, img_raw, img_label)
            img_bin = Bool.(true .- img_bin)

            dt = distance_transform(feature_transform(img_bin))
            dists[count,:] .= [dt[med_axis_dict[idx][1][x], med_axis_dict[idx][2][x]] for x=1:length(med_axis_dict[idx][1])][1:low]
            count += 1
        end
    end
    return mean(dists, dims=1)[1,:], count
end

function nn_dist(t, spl)
    spl_pts = spl(t, 1:2)

    sqrt.(dropdims(sum((spl_pts[2:end, :] .- spl_pts[1:end-1, :]) .^ 2, dims=2), dims=2))
end

function cost_dist!(t, spl, dists)
    t = clamp.(t, 0, 1)
    t[1] = 0.
    t[end] = 1.
    dists .= nn_dist(t, spl)
    
    sum((dists .- mean(dists)) .^ 2)
end

function fit_spline(A::Matrix; t=nothing)
    t = isnothing(t) ? LinRange(0,1,size(A,1)) : t
    Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())),
        t, 1:2)
end

function fit_spline(param, xs, ys, pts_n; n_subsample=15)
    # subsample data points
    spl_data = cat(xs[1:n_subsample:end], ys[1:n_subsample:end], dims=2)
    if (length(xs) - 1) % n_subsample != 0
        spl_data = vcat(spl_data, [xs[end], ys[end]]')
    end
    
    if pts_n[3] > param["nose_confidence_threshold"]
        spl_data[1,:] .= pts_n[1:2]
    end

    # fit initial spline
    spl_init = fit_spline(spl_data)

    # optimize to uniformly distribute the pts
    t0 = collect(0:0.01:1)
    dists = zeros(Float64, length(t0)-1)
    res_dist = optimize(t -> cost_dist!(t, spl_init, dists), t0, ConjugateGradient(),
        Optim.Options(g_abstol=1e-4))
    x_opt = res_dist.minimizer

    return spl_data, fit_spline(spl_init(x_opt, 1:2))
end

"""
Computes equally-spaced points along the worm spline.

# Arguments:
- `param`: Dictionary of parameters that includes the following values:
    - `num_center_pts`: Number of points along the spline
    - `segment_len`: Length of each segment
- `x_array`: x-locations of spline at each time point
- `y_array`: y-locations of spline at each time point
"""
function get_segment_end_matrix(param, x_array, y_array)
    segment_end_matrix = []
    num_center_pts = param["num_center_pts"]
    segment_len = param["segment_len"]
    @showprogress for t = 1 : size(x_array, 1)
        idx = 1
        segment_end_lst = []
        sum_segment = 0

        while idx < num_center_pts
            while sum_segment < segment_len
                sum_segment += euclidean_dist((x_array[t, idx + 1], y_array[t, idx + 1]),
                    (x_array[t, idx], y_array[t, idx]))
                idx += 1

                if idx == num_center_pts
                    break
                end

            end
            sum_segment -= segment_len

            push!(segment_end_lst, idx)
        end
        
        push!(segment_end_matrix, segment_end_lst)
    end
    return segment_end_matrix
end
