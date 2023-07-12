function create_dir(path_dir::String)
    if !isdir(path_dir)
        mkdir(path_dir)
    end
end

function read_txt(path::String)
    open(path, "r") do f
        mapfoldl(x -> x * "\n", *, readlines(f))
    end
end

function write_txt(path::String, text::String)
    open(path, "w") do f
        write(f, text)
    end
end
