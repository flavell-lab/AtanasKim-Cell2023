function read_txt(path::String)
    open(path, "r") do f
        mapfoldl(x -> x * "\n", *, readlines(f))
    end
end

"""
    read_script_paths(path_txt)

Read and verify list of .sh sbatch file paths from a txt file
"""
function read_script_paths(path_txt)
    script_path_str = read_txt(path_txt)

    script_path_list = string.(split(script_path_str, "\n")[1:end-1])

    # check if they exist
    for f_ = script_path_list
        if !(isfile(f_))
            error("sbatch path does not exist: $f_")
        end
    end

    script_path_list
end

"""
    read_sbatch_array_len(path_sh)

Returns array length of `path_sh`
"""
function read_sbatch_array_len(path_sh::String)
    str_sbatch = read_txt(path_sh)

    # find array spec and computes the array length
    for line_ = split(chomp(str_sbatch), "\n")
        if startswith(line_, "#SBATCH --array=")
            str_array = split(line_, "=")[2]
            if occursin("-", str_array) # 1-999
                array_start, array_end = parse.(Int, split(str_array, "-"))
                return length(array_start:array_end)
            elseif occursin(",", str_array) # 1,2,3,5,9
                return length(split(str_array, ","))
            else
                error("cannot parse the array spec from: $path_sh")
            end
        end
    end

    # non-array job, n subjob is 1
    return 1
end
