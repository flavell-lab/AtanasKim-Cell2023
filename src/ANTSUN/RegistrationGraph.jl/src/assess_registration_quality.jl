"""
    make_quality_dict(problems, evaluation_functions::Dict, selection_metric::String, resolutions; mask_dir::Union{String,Nothing}=nothing)

Computes the quality of registration using NCC, nearest-neighbors distance between centroids, and manual annotation.
Returns a dictionary of registration quality values for each resolution, another dictionary of the best resolution for each problem,
and a dictionary of registration resolutions that failed.
Outputs a text file containing registration quality values at the best resolution.
It is assumed that smaller values are better for the metrics.
# Arguments
- `problems`: list of registration problems to compute the quality of
- `evaluation_functions::Dict`: dictionary of metric names to functions that evaluate elastix quality on a pair of images.
    The evaluation functions will be given `rootpath`, `fixed`, `moving`, `resolution`, and possibly `mask_dir` as input, so be sure their
    other parameters have been initialized correctly. It is assumed that the functions output floating-point metric values.
- `selection_metric::String`: which metric should be used to select the best registration out of the set of possible registrations
- `resolutions`: an array of resolution values to be using. Each value is represented as a tuple `(i,j)`, where `i` is the number of parameter file
    to use and `j` is the resolution for registrations using that parameter file. Both are 0-indexed.
## Optional Keyword Arguments
- `mask_dir::Union{String,Nothing}`: directory to a mask file. Statistics will not be computed on regions outside the mask.
    If left blank, no mask will be used or passed to the evaluation functions.
"""
function make_quality_dict(problems, evaluation_functions::Dict, selection_metric::String, resolutions; mask_dir::Union{String,Nothing}=nothing)
    dict = Dict()
    best_reg = Dict()
    errors = Dict()
    func_names = keys(evaluation_functions)
    @showprogress for (moving, fixed) in problems
        best_resolution = resolutions[1]
        best_result = Inf
        dict[(moving, fixed)] = Dict()
        errors[(moving, fixed)] = Dict()
        for resolution in resolutions
            dict[(moving, fixed)][resolution] = Dict()
            errors[(moving, fixed)][resolution] = Dict()
            for metric in func_names
                func = evaluation_functions[metric]
                try
                    if isnothing(mask_dir)
                        result = func(moving, fixed, resolution)
                    else
                        result = func(moving, fixed, resolution, mask_dir)
                    end
                    dict[(moving,fixed)][resolution][metric] = result
                    if metric == selection_metric && result < best_result
                        best_result = result
                        best_resolution = resolution
                    end
                catch e
                    dict[(moving,fixed)][resolution][metric] = Inf
                    errors[(moving, fixed)][resolution][metric] = e
                end
            end
        end
        best_reg[(moving, fixed)] = best_resolution
    end
    return dict, best_reg, errors
end

"""
    make_quality_dict(param_path::Dict, param::Dict, problems, evaluation_functions)

Computes the quality of registration using NCC, nearest-neighbors distance between centroids, and manual annotation.
Returns a dictionary of registration quality values for each resolution, another dictionary of the best resolution for each problem,
and a dictionary of registration resolutions that failed.
Outputs a text file containing registration quality values at the best resolution.
It is assumed that smaller values are better for the metrics.

# Arguments

- `param_path::Dict`: Dictionary containing `path_dir_mask` entry to the path of masks (or `nothing` if no masks are used)
- `param::Dict`: Dictionary containing the following keys:
    - `quality_metric::String`: which metric should be used to select the best registration out of the set of possible registrations
    - `good_registration_resolutions`: an array of resolution values to be using. Each value is represented as a tuple `(i,j)`, where `i` is the number of parameter file
        to use and `j` is the resolution for registrations using that parameter file. Both are 0-indexed.
- `problems`: list of registration problems to compute the quality of
- `evaluation_functions::Dict`: dictionary of metric names to functions that evaluate elastix quality on a pair of images.
    The evaluation functions will be given `rootpath`, `fixed`, `moving`, `resolution`, and possibly `mask_dir` as input, so be sure their
    other parameters have been initialized correctly. It is assumed that the functions output floating-point metric values.
"""
function make_quality_dict(param_path::Dict, param::Dict, problems, evaluation_functions)
    return make_quality_dict(problems, evaluation_functions, param["quality_metric"], param["good_registration_resolutions"], mask_dir=param_path["path_dir_mask"])
end