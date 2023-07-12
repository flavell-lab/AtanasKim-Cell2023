"""
    back_one_dir(dir)

Given a directory as input, outputs the parent directory.
"""
function back_one_dir(dir)
    if dir[end] == "/"
        dir = dir[1:end-1]
    end
    prefix = ""
    if dir[1] == '/'
        prefix = "/"
    end
    return prefix*reduce(joinpath, split(dir, "/")[1:end-1])
end

"""
    get_filename(path_to_file)

Given path to a file, outputs the filename.
"""
function get_filename(path_to_file)
    if path_to_file[end] == "/"
        return split(path_to_file, "/")[end-1]
    else
        return split(path_to_file, "/")[end]
    end
end
