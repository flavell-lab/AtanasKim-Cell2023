function cluster_dist(clusters)
    n = length(clusters.order)
    distances = zeros(n,n)
    for k=2:n
        tree = cutree(clusters, k=k)
        for i=1:n
            for j=1:n
                if tree[i] != tree[j]
                    distances[i,j] = max(distances[i,j], clusters.heights[n-k+1])
                end
            end
        end
    end
    return distances
end

"""
    dendrocolor!(dg, clusters, colors; progenitors=nothing, k=nothing)

Color the dendrogram `dg` according to the cluster assignments in `clusters`. 
The `colors` argument should be a vector of colors, one for each cluster. 
The `progenitors` argument can be a vector of indices of the progenitor nodes to use for coloring.
Alternatively, the `k` argument can be an integer specifying the number of clusters to use for coloring.

# Arguments
- `dg`: A `StatsPlots` dendrogram of a hierarchical clustering result.
- `clusters`: The clusters being plotted in the dendrogram.
- `colors::Vector{Color}`: The colors to use for coloring.
- `progenitors::Vector{Int} = nothing`: The indices of the progenitor nodes to use for coloring.
- `k::Int = nothing`: The number of clusters to use for coloring.

# Returns
- `node_assignments::Dict{Int,Int}`: A dictionary mapping node indices to cluster assignments.

Also in-place modifies the plot object `dg` to color it correctly.
"""
function dendrocolor!(dg, clusters, colors; progenitors=nothing, k=nothing)
    @assert(!(isnothing(progenitors) && isnothing(k)), "Must specify either progenitor or k coloring.")
    n = length(clusters.order)
    
    dist = cluster_dist(clusters)
    node_assignments = Dict()
    
    if !isnothing(progenitors)
        for i in 1:n
            m = minimum(dist[i,progenitors])
            all_min = findall(j->dist[i,progenitors[j]]==m, 1:length(progenitors))
            if length(all_min) == 1
                node_assignments[-i] = all_min[1]
            end
        end
    elseif !isnothing(k)
        cut_cluster = cutree(clusters, k=k)
        for i in 1:n
            node_assignments[-i] = cut_cluster[i]
        end
    end
        
      
    assignment_nums = nothing
    if !isnothing(progenitors)
        assignment_nums = [sum(values(node_assignments) .== i) for i=1:length(progenitors)]
    elseif !isnothing(k)
        assignment_nums = [sum(values(node_assignments) .== i) for i=1:k]
    end
        
    
    for i=1:length(clusters.heights)
        x,y = clusters.merges[i,:]
        if haskey(node_assignments, x) && assignment_nums[node_assignments[x]] == 1
            dg[1][i].plotattributes[:linecolor] = colors[node_assignments[x]]
        end
        
        if haskey(node_assignments, y) && assignment_nums[node_assignments[y]] == 1
            dg[1][i].plotattributes[:linecolor] = colors[node_assignments[y]]
        end
            
        if haskey(node_assignments, x) && haskey(node_assignments, y) && node_assignments[x] == node_assignments[y]
            node_assignments[i] = node_assignments[x]
            dg[1][i].plotattributes[:linecolor] = colors[node_assignments[x]]
        end
    end
    return node_assignments
end
