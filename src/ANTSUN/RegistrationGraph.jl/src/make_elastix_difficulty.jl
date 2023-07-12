"""
    generate_elastix_difficulty(path_elastix_difficulty::String, t_range, heuristic::Function)

Generates an elastix difficulty file based on the given heuristic.

# Arguments
- `path_elastix_difficulty::String`: output file
- `t_range`: list or range of time points to compute the difficulty
- `heuristic`: a heuristic function that evaluates "distance" betwen two frames.
    The function will be given `t1`, and `t2` as input, so be sure its
    other parameters have been initialized correctly. It is assumed that the function outputs floating-point values.
"""
function generate_elastix_difficulty(path_elastix_difficulty::String, t_range, heuristic::Function)
    n_t = length(t_range)
    difficulty = zeros(n_t, n_t)

    @showprogress for (i,t1) = enumerate(t_range)
        # need to initialize array before multithreading (otherwise it crashes)
        if i == 1
            for j=i+1:length(t_range)
                t2 = t_range[j]
                difficulty[i, j] = heuristic(t1, t2)
            end
        else
            for j=i+1:length(t_range)
                t2 = t_range[j]
                difficulty[i, j] = heuristic(t1, t2)
            end
        end
    end
    
    open(path_elastix_difficulty, "w") do f
        write(f, string(collect(t_range))*"\n")
        write(f, replace(string(difficulty), ";"=>"\n"))
    end
    
    return difficulty
end

"""
    generate_elastix_difficulty(param_path::Dict, t_range, heuristic::Function)

Generates an elastix difficulty file based on the given heuristic.

# Arguments
- `param_path::Dict`: Dictionary of paths containing `path_elastix_difficulty` key to the path of the elastix difficulty output file
- `t_range`: list or range of time points to compute the difficulty
- `heuristic`: a heuristic function that evaluates "distance" betwen two frames.
    The function will be given `t1`, and `t2` as input, so be sure its
    other parameters have been initialized correctly. It is assumed that the function outputs floating-point values.
"""
function generate_elastix_difficulty(param_path::Dict, t_range, heuristic::Function)
    path_elastix_difficulty = param_path["path_elastix_difficulty"]
    generate_elastix_difficulty(path_elastix_difficulty, t_range, heuristic)
end
