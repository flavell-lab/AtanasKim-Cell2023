
"""
    test_signifiance_connectome(dict_axis::Dict, dict_feature::Dict, f_select::Function, bins)

Test the significance of the connectome plot by comparing the distribution of the selected features with the distribution of all features.

# Arguments
- `dict_axis::Dict`: dictionary of the axis values, key is the neuron name, value is the axis value
- `dict_feature::Dict`: dictionary of the feature values, key is the neuron name, value is the feature value
- `f_select::Function`: function to select the feature, the function should take a single argument and return a boolean value
- `bins`: number of bins for the histogram
"""
function test_signifiance_connectome(dict_axis::Dict, dict_feature::Dict, f_select::Function, bins; verbose=true)
    list_all = Float64[]
    list_select = Float64[]

    for (neuron,x) = dict_axis
        if !occursin(r"[A-Z]{2}\d", neuron) # check if not vc motor
            if haskey(dict_feature, neuron) 
                # neuron provided with or the class does not have dv & lr
                
                push!(list_all, x)
                v = dict_feature[neuron]
                if f_select(v)
                    push!(list_select, x)
                end
            else
                class, dv, lr = get_neuron_class(neuron)
                
                class_dv = class
                if !(dv == "missing" || dv == "undefined")
                    class_dv = class * dv
                end
                
                if haskey(dict_feature, class_dv)
                    push!(list_all, x)
                    v = dict_feature[class_dv]
                    if f_select(v)
                        push!(list_select, x)
                    end
                else
                    # println("$class missing in class dict")
                end
            end
        end # if not vc motor
    end
    
    result = Dict[]
    for range_bin_indices = 1:(length(bins) - 1)
    
        # observed frequencies
        observed = fit(Histogram, list_all, bins).weights
        observed_special = fit(Histogram, list_select, bins).weights

        # the proportion of selected in the range of interest
        observed_proportion = sum(observed_special[range_bin_indices]) / sum(observed_special)

        # expected proportion of selected points in the range of interest
        expected_proportion = sum(observed[range_bin_indices]) / sum(observed)

        p_hat = observed_proportion
        p_pop = expected_proportion
        n = sum(observed_special)

        z_score = (p_hat - p_pop) / sqrt((p_pop * (1 - p_pop))/(n))
        p_value = 2 * (1 - cdf(Normal(), abs(z_score)))
        
        if verbose
            println("bin start: $(bins[range_bin_indices]) p_val: $p_value")
        end

        dict_bin = Dict("bin_start" => bins[range_bin_indices], "bin_end"=>bins[range_bin_indices+1], "p_value"=>p_value)
        push!(result, dict_bin)
    end
    
    result
end

"""
Test the inter-connectivity intra-fraction of a graph by comparing the fraction of intra-connections of the selected nodes with the fraction of intra-connections of a random sample of nodes.

# Arguments
- `g`: the graph to test
- `list_node_select::Vector`: list of neurons to test
- `list_node_sample::Vector`: list of neurons to randomly sample from
- `n_trial::Int`: number of trials to perform

# Returns
- `intra_frac`: the interconnectivity fraction of the selected nodes
- `intra_rand_frac_intra`: the interconnectivity fraction of the randomly-sampled nodes
"""
function test_inter_connectivity_intra_frac(g, list_node_select::Vector; list_node_sample::Vector, n_trial::Int=1000)
    n_node_select = length(list_node_select)
    
    list_frac_intra = Float64[]
    for node = list_node_select
        total_in = 0
        total_out = 0
        total_intra_in = 0
        total_intra_out = 0
        
        for (node_from,node_to,edge_data) = g.in_edges(node, data=true)
            syn = edge_data["weight"]
            if node_from in list_node_select && !isnan(syn)
                total_intra_in += syn
            end
            total_in += syn
        end
        
        for (node_from,node_to,edge_data) = g.out_edges(node, data=true)
            syn = edge_data["weight"]
            if node_to in list_node_select && !isnan(syn)
                total_intra_out += syn
            end
            total_out += syn
        end
        
        total = total_in + total_out
        total_intra = total_intra_in + total_intra_out 
        push!(list_frac_intra, total_intra / total)
    end
    intra_frac = mean(list_frac_intra)
   
    
    ## random control
    list_node_intersect = intersect(list_node_sample, g.nodes())
    list_rand_frac_intra = zeros(n_trial)
    @showprogress for i_trial = 1:n_trial
        list_node_rand = sample(list_node_intersect, n_node_select, replace=false)
        list_neuron = list_node_rand
        
        list_frac_intra = Float64[]
        for node = list_neuron
            total_in = 0
            total_out = 0
            total_intra_in = 0
            total_intra_out = 0

            for (node_from,node_to,edge_data) = g.in_edges(node, data=true)
                syn = edge_data["weight"]
                if node_from in list_neuron
                    total_intra_in += syn
                end
                total_in += syn
            end

            for (node_from,node_to,edge_data) = g.out_edges(node, data=true)
                syn = edge_data["weight"]
                if node_to in list_neuron
                    total_intra_out += syn
                end
                total_out += syn
            end

            total = total_in + total_out
            total_intra = total_intra_in + total_intra_out 
            push!(list_frac_intra, total_intra / total)
        end
        
        list_rand_frac_intra[i_trial] = mean(list_frac_intra)
    end
    
    intra_frac, list_rand_frac_intra
end
