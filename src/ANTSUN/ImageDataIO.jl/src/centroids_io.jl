"""
    write_centroids(centroids, out)

Writes `centroids` to transformix-compatible file `out`
"""
function write_centroids(centroids, out)
    open(out, "w") do f
        write(f, "index\n")
        write(f, string(length(centroids)) * "\n")
        for centroid in centroids
            write(f, replace(string(centroid), r"\(|\,|\)" => "") * "\n")
        end
    end
end

"""
    read_centroids_transformix(input)

Reads centroids from transformix output file `input`
"""
function read_centroids_transformix(input)
    centroids = []
    open(input) do f
        for line in eachline(f)
            push!(centroids, Tuple(map(x->parse(Int64,x),split(split(split(line, "OutputIndexFixed = [ ")[2], " ]")[1]," "))))
        end
    end
    return centroids
end

"""
    read_centroids_roi(input)

Reads centroids from `write_centroids` output file `input`
"""
function read_centroids_roi(input)
    centroids = []
    open(input) do f
        count = 1
        for line in eachline(f)
            if count < 3
                count += 1
                continue
            end
            push!(centroids, Tuple(map(x->parse(Float64,x), split(line))))
        end
    end
    return centroids
end

