"""
    write_dict(dict, out::AbstractString; kv_delimiter::AbstractString=" ", kk_delimiter::AbstractString="\n")

Writes a dictionary to a file.

# Arguments
- `dict`: Dictionary to write
- `out::AbstractString`: filename to write to
- `kv_delimiter::AbstractString` (optional, default " ") delimiter between keys and values
- `kk_delimiter::AbstractString` (optional, default "\n") delimiter between subsequent keys
"""
function write_dict(dict, out::AbstractString; kv_delimiter::AbstractString=" ", kk_delimiter::AbstractString="\n")
    open(out, "w") do f
        for k in keys(dict)
            write(f, "$(k)$(kv_delimiter)$(dict[k])$(kk_delimiter)")
        end
    end
end

"""
    extract_key(dict, key, n::Integer, extract::Bool)

Adds or retrieves a key from a given layer of a nested dictionary.

# Arguments
- `dict`: Dictionary to operate on
- `key`: Key to interact with
- `n::Integer`: The layer of the dictionary to interact with
- `extract::Bool`: If true, extracts the key from the dictionary; if false, adds the key to the dictionary instead.
"""
function extract_key(dict, key, n::Integer, extract::Bool)
    new_dict = Dict()
    if n == 1
        if extract
            return dict[key]
        else
            new_dict[key] = dict
            return new_dict
        end
    end
    for k in keys(dict)
        new_dict[k] = extract_key(dict[k], key, n-1, extract)
    end
    return new_dict
end


"""
    read_2d_dict(input::AbstractString, outer_key_dtype::Type, inner_key_dtype::Type, val_dtype::Type)

Reads 2D dictionary (dictionary of dictionaries) from a file.

# Arguments
- `input::AbstractString`: input file
- `outer_key_dtype::Type`: data type of outer keys
- `inner_key_dtype::Type`: data type of inner keys
- `val_dtype::Type`: data type of values of inner dict
"""
function read_2d_dict(input::AbstractString, outer_key_dtype::Type, inner_key_dtype::Type, val_dtype::Type)
    dict = Dict()
    open(input) do f
        for line in eachline(f)
            k = Tuple(parse_1d_tuple(split(line, "Dict")[1], outer_key_dtype))
            if length(k) == 1
                k = k[1]
            end
            v = parse_1d_dict(split(line, "}(")[2], inner_key_dtype, val_dtype)
            dict[k] = v
        end
    end
    return dict
end


"""
    parse_1d_tuple(tuple_str::AbstractString, dtype::Type)

Parses one-dimensional string tuple `tuple_str::AbstractString` into a tuple of the specified `dtype::Type`
"""
function parse_1d_tuple(tuple_str::AbstractString, dtype::Type)
    return Tuple(map(x->parse(dtype, x), split(replace(tuple_str, r"\(|\)|\[|\]"=>""), ",")))
end


"""
    parse_1d_dict(dict_str::AbstractString, key_dtype::Type, val_dtype::Type)

Parses a 1D dictionary; the keys and values must be at most 1D arrays. Arguments:

- `dict_str::AbstractString`: string representation of dictionary to parse
- `key_dtype::Type`: element data type of dictionary keys
- `val_dtype::Type`: element data type of dictionary values
"""
function parse_1d_dict(dict_str::AbstractString, key_dtype::Type, val_dtype::Type)
    dict = Dict()
    kv_pairs = split(dict_str, "=>")
    for i=1:length(kv_pairs) - 1
        key = split_arrays(kv_pairs[i])[2 - (i == 1)]
        if key isa String
            key = parse(key_dtype, key)
        else
            key = Tuple(map(x->parse(key_dtype, x), key))
        end
        val = split_arrays(kv_pairs[i+1])[1]
        if val isa String
            val = parse(val_dtype, val)
        else
            val = Tuple(map(x->parse(val_dtype, x), val))
        end
        dict[key] = val
    end
    return dict
end



"""
    split_arrays(arrays::AbstractString; fwd_delimiters=['[', '('], back_delimiters=[')', ']'], val_delimiters=[','], ignore_chars=[' ', '\n'])

Parses a string multi-dimensional array or list of arrays `arrays::AbstractString` into its component arrays.

# Arguments
- `arrays::AbstractString`: string of arrays to parse

# Optional keyword arguments
- `fwd_delimiters`: list of characters to delineate when an array starts
- `back_delimiters`: list of characters to delineate when an array ends
- `val_delimiters`: list of characters to delineate separating values of the array
- `ignore_chars`: list of characters to ignore and not put in the final array
"""
function split_arrays(arrays::AbstractString; fwd_delimiters=['[', '('], back_delimiters=[')', ']'], val_delimiters=[','], ignore_chars=[' ', '\n'])
    arrs = []
    index = [1]
    curr_str = ""
    for c in arrays
        if (c in fwd_delimiters) || (c in back_delimiters) || (c in val_delimiters)
            if curr_str != ""
                push!(multi_index_array(arrs, index[1:end-1]), string(curr_str))
                curr_str = ""
            end
        end
        if c in fwd_delimiters
            push!(multi_index_array(arrs, index[1:end-1]), [])
            push!(index, 1)
        elseif c in back_delimiters
            index = index[1:end-1]
        elseif c in val_delimiters
            index[end] += 1
        elseif !(c in ignore_chars)
            curr_str *= c
        end
    end
    push!(multi_index_array(arrs, index[1:end-1]), string(curr_str))
    return arrs
end

"""
    add_get_basename!(param_path::Dict)

Adds `get_basename` key to the dictionary `param_path`.
"""
function add_get_basename!(param_path::Dict)
    param_path["get_basename"] = (t::Int, ch::Int) -> param_path["img_prefix"]*"_t$(lpad(t, 4, "0"))_ch$(ch)"
end


"""
    add_get_basename!(param_path::Dict, param::Dict)

Adds `get_basename` key to the dictionary `param_path`, given information about multiple dataset timing in `param`.
"""
function add_get_basename!(param_path::Dict, param::Dict)
    param_path["get_basename"] = (t::Int, ch::Int) -> let
        t_eff = t
        idx = 1
        while t_eff > param["max_ts"][idx]
            t_eff -= param["max_ts"][idx]
            idx +=1
        end
        param_path["imgs_prefix"][idx]*"_t$(lpad(t_eff, 4, "0"))_ch$(ch)"
    end
end

"""
    change_rootpath!(param_path::Dict, new_rootpath::String)

Updates a dictionary of paths `param_path` by changing the old rootpath to the `new_rootpath`
"""
function change_rootpath!(param_path::Dict, new_rootpath::String)
    old_rootpath = param_path["path_root_process"]
    for k in keys(param_path)
        if typeof(param_path[k]) == String
            param_path[k] = replace(param_path[k], old_rootpath=>new_rootpath)
        end
    end
end


"""
    multi_index_array(array, index)

Indexes a nested `array` at `index` as though the array was a mulit-dimensional array.
"""
function multi_index_array(array, index)
    sub_array = array
    for v in index
        sub_array = sub_array[v]
    end
    return sub_array
end
