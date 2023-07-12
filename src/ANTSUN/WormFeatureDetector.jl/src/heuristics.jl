"""
    elastix_difficulty_hsn_nr(
        rootpath::String, frame1::Integer, frame2::Integer, 
        hsn_location_file::String, nr_location_file::String; nr_weight::Real=1
    )

Computes registration difficulty between two frames based on the HSN and nerve ring locations

# Arguments:
- `rootpath::String`: working directory path; all other directory inputs are relative to this
- `frame1::Integer`: first frame
- `frame2::Integer`: second frame
- `hsn_location_file::String`: path to file containing HSN locations
- `nr_location_file::String`: path to file containing nerve ring locations
# Heuristic parameters (optional):
- `nr_weight::Real`: weight of nerve ring location relative to HSN location. Default 1.
"""
function elastix_difficulty_hsn_nr(rootpath::String, frame1::Integer, frame2::Integer, 
        hsn_location_file::String, nr_location_file::String; nr_weight::Real=1)
    # load HSN locations
    hsn_locs = Dict()
    open(joinpath(rootpath, hsn_location_file)) do hsn
        for line in eachline(hsn)
            s = [parse(Int32, x) for x in split(line)]
            hsn_locs[s[1]] = s[2:4]
        end
    end
    # load nerve ring locations
    nr_locs = Dict()
    open(joinpath(rootpath, nr_location_file)) do nr
        for line in eachline(nr)
            s = [parse(Int32, x) for x in split(line)]
            nr_locs[s[1]] = s[2:4]
        end
    end

    return sqrt(sum((hsn_locs[frame1] .- hsn_locs[frame2]).^2))
                + nr_weight * sqrt(sum((nr_locs[frame1] .- nr_locs[frame2]).^2))

end

"""
    elastix_difficulty_wormcurve!(
        curves::Array{<:Any,1}, img1::Union{Nothing,Array{<:AbstractFloat,3}}, img2::Union{Nothing,Array{<:AbstractFloat,3}},
        t1::Int, t2::Int, head_pos_t1::Union{Nothing,Dict}, head_pos_t2::Union{Nothing,Dict}; downscale::Int=3, 
        num_points::Int=9, headpt::Int=4, tailpt::Int=7, path_dir_fig::Union{Nothing,String}=nothing
    )

Computes registration difficulty between two time points based on the worm curvature heuristic.
Requires that the data be filtered in some way (eg: total-variation filtering),
and that the head position of the worm is known in each time point.

# Arguments:
- `curves::Array{<:Any,1}`: Array of worm curves found so far. The method will attempt to find the worm's curvature in this array,
    and will compute and add it to the array if not found.
- `img1::Array{<:AbstractFloat,3}`: image 1 array (volume), or `nothing` if the curve was already computed
- `img2::Array{<:AbstractFloat,3}`: image 2 array (volume), or `nothing` if the curve was already computed
- `t1::Int`: time point 1
- `t2::Int`: time point 2
- `head_pos_t1::Dict`: head position dictionary at time point 1
- `head_pos_t2::Dict`: head position dictionary at time point 2

## Other parameters (optional):
- `path_dir_fig::Union{Nothing,String}`: Path to save figures of worm curvature. If `nothing`, figures will not be generated.
- `downscale::Integer`: log2(factor) by which to downscale the image before processing. Default 3 (ie: downscale by a factor of 8)
- `num_points::Integer`: number of points (not including head) in generated curve. Default 9.
- `headpt::Integer`: First position from head (in index of curves) to be aligned. Default 4.
- `tailpt::Integer`: Second position from head (in index of curves) to be aligned. Default 7.
"""
function elastix_difficulty_wormcurve!(curves::Array{<:Any,1}, img1::Union{Nothing,Array{<:AbstractFloat,3}}, img2::Union{Nothing,Array{<:AbstractFloat,3}},
        t1::Int, t2::Int, head_pos_t1::Union{Nothing,Dict}, head_pos_t2::Union{Nothing,Dict}; downscale::Int=3, num_points::Int=9, headpt::Int=4, tailpt::Int=7,
        path_dir_fig::Union{Nothing,String}=nothing)

    if !isnothing(img1)
        img1 = maxprj(img1, dims=3)
    end

    if !isnothing(img2)
        img2 = maxprj(img2, dims=3)
    end

    if isnothing(img1)
        x1_c, y1_c = curves[t1]
    else
        x1_c, y1_c = find_curve(img1, downscale, head_pos_t1[t1]./2^downscale, num_points)
    end
    
    if isnothing(img2)
        x2_c, y2_c = curves[t2]
    else
        x2_c, y2_c = find_curve(img2, downscale, head_pos_t2[t2]./2^downscale, num_points)
    end

    if !isnothing(path_dir_fig)
        if !isnothing(img1)
            create_dir(path_dir_fig)
            fig = heatmap(transpose(img1), fillcolor=:grays, aspect_ratio=1, flip=false, showaxis=false, legend=false)
            scatter!(fig, x1_c.-1, y1_c.-1, color="red");
            scatter!(fig, [x1_c[1].-1], [y1_c[1].-1], color="cyan", markersize=5);
            savefig(fig, joinpath(path_dir_fig, "$(t1).png"));
        end
        if !isnothing(img2)
            create_dir(path_dir_fig)
            fig = heatmap(transpose(img2), fillcolor=:grays, aspect_ratio=1, flip=false, showaxis=false, legend=false)
            scatter!(fig, x2_c.-1, y2_c.-1, color="red");
            scatter!(fig, [x2_c[1].-1], [y2_c[1].-1], color="cyan", markersize=5);
            savefig(fig, joinpath(path_dir_fig, "$(t2).png"));
        end
    end

    if !isassigned(curves, t1)
        curves[t1] = (x1_c, y1_c)
    end
    if !isassigned(curves, t2)
        curves[t2] = (x2_c, y2_c)
    end

    return curve_distance(x1_c, y1_c, x2_c, y2_c, headpt=headpt, tailpt=tailpt)
end

"""
    elastix_difficulty_wormcurve!(
        curves::Array{<:Any,1}, param::Dict, param_path_fixed::Dict, param_path_moving::Dict, t1::Int, t2::Int, ch::Int;
        save_curve_fig::Bool=false, max_fixed_t::Union{Integer,Nothing}=nothing
    )

Computes registration difficulty between two time points based on the worm curvature heuristic.
Requires that the data be filtered in some way (eg: total-variation filtering),
and that the head position of the worm is known in each time point.

# Arguments
- `curves::Dict`: Array of worm curves found so far. The method will attempt to find the worm's curvature in this array,
and will compute and add it to the array if not found.
- `param::Dict`: Dictionary of parameter settings, including:
    - `worm_curve_n_pts`: number of points (not including head) in generated curve.
    - `worm_curve_head_idx`: First position from head (in index of curves) to be aligned.
    - `worm_curve_tail_idx`: Second position from head (in index of curves) to be aligned.
    - `worm_curve_downscale`: log2(factor) by which to downscale the image before processing.
- `param_path_fixed::Dict`: Dictionary of paths for the fixed (`t1`) time point, including:
    - `path_dir_nrrd_filt`: Path to filtered cropped NRRD files
    - `path_head_pos`: Path to head position
    - `path_dir_worm_curve`: Path to save worm curve images
    - `get_basename`: Function that takes as input a time point and a channel and gives the base name of the corresponding NRRD file. 
- `param_path_moving::Dict`: Dictionary of paths for the moving (`t2`) time point, including the same keys as for the fixed dictionary.
- `t1::Int`: Fixed time point
- `t2::Int`: Moving time point
- `ch::Int`: Channel
- `save_curve_fig::Bool` (optional, default `false`): whether to save worm curve images
- `max_fixed_t::Union{Integer,Nothing}` (optional, default `nothing`): If using two different data sets, the maximum time point in the fixed dataset.
    Moving dataset time points will be incremented by this amount.
"""
function elastix_difficulty_wormcurve!(curves::Array{<:Any,1}, param::Dict, param_path_fixed::Dict, param_path_moving::Dict, t1::Int, t2::Int, ch::Int;
        save_curve_fig::Bool=false, max_fixed_t::Union{Integer,Nothing}=nothing)

    worm_curve_n_pts = param["worm_curve_n_pts"] 
    worm_curve_tail_idx = param["worm_curve_tail_idx"]
    worm_curve_head_idx = param["worm_curve_head_idx"]
    worm_curve_downscale = param["worm_curve_downscale"]
    
    if isnothing(max_fixed_t)
        max_fixed_t = 0
    else
        if t1 > max_fixed_t || t2 <= max_fixed_t
            return Inf
        end
    end

    if isassigned(curves, t1)
        img1 = nothing
        head_pos_t1 = nothing
    else
        path_nrrd_t1 = joinpath(param_path_fixed["path_dir_nrrd_filt"],
            param_path_fixed["get_basename"](t1, ch) * ".nrrd")
        img1 = Float64.(read_img(NRRD(path_nrrd_t1)))
        head_pos_t1 = read_head_pos(param_path_fixed["path_head_pos"])
    end

    if isassigned(curves, t2)
        img2 = nothing
        head_pos_t2 = nothing
    else
        path_nrrd_t2 = joinpath(param_path_moving["path_dir_nrrd_filt"],
            param_path_moving["get_basename"](t2 - max_fixed_t, ch) * ".nrrd")
        img2 = Float64.(read_img(NRRD(path_nrrd_t2)))
        head_pos_t2 = read_head_pos(param_path_moving["path_head_pos"])
    end

    if !isnothing(max_fixed_t) && !isnothing(head_pos_t2)
        head_pos_t2_shifted = Dict()
        for t in keys(head_pos_t2)
            head_pos_t2_shifted[t + max_fixed_t] = head_pos_t2[t]
        end
        head_pos_t2 = head_pos_t2_shifted
    end

    path_dir_worm_curve = save_curve_fig ? param_path_fixed["path_dir_worm_curve"] : nothing
    
    elastix_difficulty_wormcurve!(curves, img1, img2, t1, t2, head_pos_t1, head_pos_t2,
        downscale=worm_curve_downscale, num_points=worm_curve_n_pts, headpt=param["worm_curve_head_idx"],
        tailpt=param["worm_curve_tail_idx"], path_dir_fig=path_dir_worm_curve)
end
