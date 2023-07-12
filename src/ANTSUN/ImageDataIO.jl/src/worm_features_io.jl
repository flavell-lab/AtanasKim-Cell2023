"""
    function read_head_pos(head_path::String)

Reads the worm head position from the file `head_path::String`.
Returns a dictionary mapping frame => head position of the worm at that frame.
"""
function read_head_pos(head_path::String)
    head_pos = Dict()
    open(head_path) do f
        for line in eachline(f)
            l = split(line)
            head_pos[parse(Int64, l[1])] = Tuple(map(x->parse(Int64, x), l[2:end]))
        end
    end
    return head_pos
end

"""
    read_nrrd(rootpath, img_prefix, nrrd_path, frame, channel)

Reads NRRD file from `\$(rootpath)/\$(nrrd_path)/\$(img_prefix)_t\$(channel).nrrd` and outputs resulting image.
"""        
function read_nrrd(rootpath, img_prefix, nrrd_path, frame, channel)
    return read_img(NRRD(joinpath(rootpath, nrrd_path, img_prefix*"_t"*string(frame, pad=4)*"_ch$(channel).nrrd")))
end
