"""
    to_dict(graph::SimpleWeightedGraph)

Converts a `graph::SimpleWeightedGraph` into an adjacency dictionary
node => Dict({neighbor1 => weight1, neighbor2=> weight2, ...})
"""
function to_dict(graph::SimpleWeightedGraph)
    dict = Dict()
    for edge in edges(graph)
        if !(src(edge) in keys(dict))
            dict[src(edge)] = Dict()
        end
        if !(dst(edge) in keys(dict))
            dict[dst(edge)] = Dict()
        end
        dict[src(edge)][dst(edge)] = weight(edge)
        dict[dst(edge)][src(edge)] = weight(edge)
    end
    return dict
end


"""
    optimize_subgraph(graph::SimpleWeightedGraph)

Finds the minimum node of a graph, which has the smallest average shortest path length
to each other node in that graph. Unconnected nodes are counted as having a path length equal to the
highest edge weight.
# Parameters
- `graph::SimpleWeightedGraph`: a graph
# Returns:
- `min_node::Integer`: the minimum node
- `subgraph::SimpleWeightedDiGraph`: an unweighted graph whose edges consist of shortest paths from the minimum node of the original graph to other nodes.
- `maximum_problem_chain::Integer`: the number of vertices in the longest chain of registration problems in the subgraph.
This graph is directed, and only has edges going away from the minimum node.
"""
function optimize_subgraph(graph::SimpleWeightedGraph)
    dist = []
    paths = []
    max_val = maximum(map(e->(weight(e) == Inf ? 0 : weight(e)), edges(graph)))
    # get all shortest paths
    for i=1:nv(graph)
        paths_i = dijkstra_shortest_paths(graph, i)
        push!(dist, paths_i.dists)
        push!(paths, enumerate_paths(paths_i))
    end
    # find node with minimum shortest paths
    min_avg, min_ind = findmin([sum(map(x->(x > max_val) ? max_val : x, d)) for d in dist])
    min_paths = paths[min_ind]
    subgraph = SimpleWeightedDiGraph(nv(graph))
    weight_dict = to_dict(graph)
    for path in min_paths
        for i=2:length(path)
            add_edge!(subgraph, path[i-1], path[i], weight_dict[path[i]][path[i-1]])
        end
    end
    return (min_ind, subgraph, maximum([length(path) for path in min_paths]))
end

"""
    make_voting_subgraph(graph::SimpleWeightedGraph, degree::Integer)

Makes a subgraph of a `graph`, where each node is connected to their `degree` closest neighbors.
Returns the subgraph, and an array of nodes that were disconnected from the rest of the nodes.
"""
function make_voting_subgraph(graph::SimpleWeightedGraph, degree::Integer)
    subgraph = SimpleWeightedDiGraph(nv(graph))
    added_edges = []
    dict = to_dict(graph)
    for vertex1 in keys(dict)
        sorted_distances = sort([dist for dist in dict[vertex1]], by=dist->dist[2])
        for (vertex2, dist) in sorted_distances[1:degree]
            v1 = min(vertex1, vertex2)
            v2 = max(vertex1, vertex2)
            if !((v1, v2) in added_edges)
                push!(added_edges, (v1, v2))
                add_edge!(subgraph, v1, v2, dist)
            end
        end
    end
    unconnected = find_unconnected(added_edges)
    if length(unconnected) > 0
        @warn "Disconnected graph"
    end
    return (subgraph, unconnected)
end

"""
    find_unconnected(edges)

Finds a set of nodes that aren't connected to the rest of the nodes through the `edges`.
"""
function find_unconnected(edges)
    vertices = []
    for edge in edges
        if !(edge[1] in vertices)
            push!(vertices, edge[1])
        end
        if !(edge[2] in vertices)
            push!(vertices, edge[2])
        end
    end

    connected_vertices = [vertices[1]]
    updated = true
    while updated
        updated = false
        for edge in edges
            if (edge[1] in connected_vertices) && !(edge[2] in connected_vertices)
                push!(connected_vertices, edge[2])
                updated = true
            end
            if (edge[2] in connected_vertices) && !(edge[1] in connected_vertices)
                push!(connected_vertices, edge[1])
                updated = true
            end
        end
    end
    
    return [vertex for vertex in vertices if !(vertex in connected_vertices)]
end



"""
    load_graph(elx_difficulty::String; func=nothing, difficulty_importance::Real=0.05)

Loads an adjacency matrix from a file and stores it as a graph.
You can specify a function to apply to each weight.
This is often helpful in cases where the heuristic obeys the triangle inequality,
to avoid the function from generating a star graph and mostly ignoring the heuristic.
By default, the function is x->x^1.05, which allows the algorithm to
split especially difficult registration problems into many steps.
If this results in too many failed registrations, try increasnig the difficulty_importance parameter;
conversely, if there is too much error accumulation over long chains of registrations, try decreasing it.
# Arguments
- `elx_difficulty::String`: a path to a text file containing a list of frames and an adjacency matrix.
- `func` (optional): a function to apply to each element of the adjacency matrix
- `difficulty_importance` (optional, default 0.05): if `func` is not provided, it will be set to `x->x^(1+difficulty_importance)`
Returns a graph `graph::SimpleWeightedGraph` storing the adjacency matrix.
"""
function load_graph(elx_difficulty::String; func=nothing, difficulty_importance::Real=0.05)
    graph = nothing
    imgs = nothing
    if func == nothing
        func = x->x^(1+difficulty_importance)
    end
    open(elx_difficulty) do f
        count = 0
        for line in eachline(f)
            if count == 0
                # the array of images
                imgs = map(x->parse(Int64, x), split(replace(line, r"\[|\]|Any" => ""), ", "))
                # number of nodes is the highest frame number
                graph = SimpleWeightedGraph(maximum(imgs))
            else
                for e in enumerate(map(x->func(parse(Float64, x)), split(replace(line, r"\[|\]" => ""))))
                    if e[2] != Inf && e[2] != 0
                        # difficulty array may have skipped already-deleted frames
                        # make sure to index into imgs to get accurate frame numbers
                        add_edge!(graph, imgs[count], imgs[e[1]], e[2])
                    end
                end
            end
            count = count + 1
        end
    end
    return graph
end
 
"""
    output_graph(subgraph::SimpleWeightedDiGraph, outfile::String; max_fixed_t::Int=0)

Outputs `subgraph::SimpleWeightedDiGraph` to an output file `outfile` containing a list of edges in `subgraph`.
Can set `max_fixed_t::Int` parameter if a dataset-alignment registration is being done.
"""
function output_graph(subgraph::SimpleWeightedDiGraph, outfile::String; max_fixed_t::Int=0)
    open(outfile, "w") do f
        for edge in edges(subgraph)
            e1 = Int16(src(edge))
            e2 = Int16(dst(edge))
            if e1 > max_fixed_t && e2 <= max_fixed_t
                e1 -= max_fixed_t
            end
            # other dataset is always moving
            if e2 > max_fixed_t && e1 <= max_fixed_t
                e3 = e2 - max_fixed_t
                e2 = e1
                e1 = e3
            end
            write(f, string(e1)*" "*string(e2)*"\n")
        end
    end
end

"""
    remove_frame(graph::SimpleWeightedGraph, frame::Integer)
s
Given a graph `graph::SimpleWeightedGraph` and a problematic `frame::Integer`, deletes the frame from the graph without changing frame indices.
"""
function remove_frame(graph::SimpleWeightedGraph, frame::Integer)
    new_graph = SimpleWeightedGraph(nv(graph))
    for edge in edges(graph)
        if frame != src(edge) && frame != dst(edge)
            add_edge!(new_graph, src(edge), dst(edge), weight(edge))
        end
    end
    return new_graph
end


"""
    update_graph(reg_quality_arr::Array{String,1}, graph::SimpleWeightedGraph, metric::String; metric_tfm=identity)

Recomputes the difficulty graph based on registration quality data.
Returns a new graph where difficulties of registration problems are scaled by the quality metric.

# Arguments

- `reg_quality_arr::Array{String,1}` is an array of paths to files containing registration quality data
- `graph::SimpleWeightedGraph` is the difficulty graph to be updated
- `metric::String` is which quality metric to use to update the graph

# Optional keyword arguments

- `metric_tfm`: Function to apply to each metric value. Default identity.
"""
function update_graph(reg_quality_arr::Array{String,1}, graph::SimpleWeightedGraph, metric::String; metric_tfm=identity)
    new_graph = copy(graph)
    d = to_dict(graph)
    for reg_quality in reg_quality_arr
        open(reg_quality, "r") do f
            first = true
            idx = 0
            for line in eachline(f)
                data = split(line)
                if first
                    idx = findfirst(data.==metric)
                    first = false
                    continue
                end
                moving,fixed = map(x->parse(Int32, x), split(data[1], "to"))
                metric_val = parse(Float64, data[idx])
                new_difficulty = d[moving][fixed] * metric_tfm(metric_val)
                add_edge!(new_graph, moving, fixed, new_difficulty)
            end
        end
    end
    return new_graph
end

"""
    remove_previous_registrations(previous_problems, subgraph::SimpleWeightedDiGraph)

Removes previous registrations from the subgraph.
# Arguments:
- `previous_problems`: list of registration problems
- `subgraph::SimpleWeightedDiGraph`: current subgraph
"""
function remove_previous_registrations(previous_problems, subgraph::SimpleWeightedDiGraph)
    subgraph_purged = SimpleWeightedDiGraph(nv(subgraph))
    for edge in edges(subgraph)
        if !((src(edge), dst(edge)) in previous_problems)
            add_edge!(subgraph_purged, src(edge), dst(edge), weight(edge))
        end
    end
    return subgraph_purged
end
