"""
    add_list_dict!(dict, key, item)

If `key` is not in `dict`, add it with the value `[item]`.
If `key` is in `dict`, append `item` to the value.

# Arguments
- `dict::Dict`: The dictionary to add to.
- `key`: The key to add.
- `item`: The item to add.
"""
function add_list_dict!(dict, key, item)
    if haskey(dict, key)
        push!(dict[key], item)
    else
        dict[key] = [item]
    end
    
    nothing
end
