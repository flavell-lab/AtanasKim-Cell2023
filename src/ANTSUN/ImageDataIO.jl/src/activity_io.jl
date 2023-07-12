"""
    write_activity(activity, out)

Writes list of ROI activities `activity` to output file `out`.
"""
function write_activity(activity, out)
    open(out, "w") do f
        for a in activity
            write(f, string(a) * "\n")
        end
    end
end

"""
    read_activity(input)
 
Reads ROI activities from `input`.
"""
function read_activity(input)
    activity = []
    open(input) do f
        for line in eachline(f)
            push!(activity, parse(Float64, line))
        end
    end
    return activity
end

