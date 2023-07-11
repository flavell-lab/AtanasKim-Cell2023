function get_neuron_type_wh_dvlr(class)
    list_motor_prefix = ["AS", "VD", "VC", "VB", "VA", "DA", "DB", "DD"]
    
    REF = NeuroPALData.NEURON_REF_DICT
    
    if haskey(REF, class)
        return REF[class]["type"]
    elseif any(startswith.(class, list_motor_prefix))
        return "motorneuron"
    elseif class in ["LegacyBodyWallMuscles", "DefecationMuscles"] || startswith(class, "pm")
        return "muscle"
    else
        return "unknown"
    end
end

function get_neuron_type_wv(class)
    manual_list_unknown = ["CEPshD", "CEPshV", "GLRD", "GLRV", "NR_fragment", "vncfrag", "G1", "G2", "CAN","VAn","Fragment"]
    manual_list_muscle = ["pm5", "pm1", "pm4", "pm3", "pm6", "pm7", "mc3", "mc2", "pm2"]
    manual_list_modulatory = ["HSNL", "HSNR", "NSM"]
    manual_list_interneuron = ["I1", "I2", "I3", "I4", "I5", "I6"]
    manual_list_motorneuron = ["M1", "M2", "M3", "M4", "M5", "MC", "MI"]

    # remove d/v and l/r
    try
        class, dv, lr = get_neuron_class(class)
    catch
    end
    
    if haskey(witvliet_type, class)
        return witvliet_type[class]
    else
        if startswith(class, "BWM")
            return "muscle"
        elseif class in manual_list_unknown
            return "unknown"
        elseif class in manual_list_muscle
            return "muscle"
        elseif class in manual_list_modulatory
            return "modulatory"
        elseif class in manual_list_interneuron
            return "inter"
        elseif class in manual_list_motorneuron
            return "motor"
        elseif endswith(class, "D") || endswith(class, "V")
            witvliet_type[class[1:end-1]]
        else
            error("$class is unknown")
        end
    end
end

function get_node_name(neuron; dv=true, lr=true)
    try
        class, dv_, lr_ = get_neuron_class(neuron)
        return_str = class
        if dv && (dv_ != "undefined") && (dv_ != "missing")
            return_str *= dv_
        end
        if lr && (lr_ != "undefined") && (lr_ != "missing")
            return_str *= lr_
        end
        
        return return_str
    catch
        return neuron
    end
end

# generate graph
function get_graph_white(min_n_edge=1; merge_dv=false, merge_lr=false, synapse_electrical=true, synapse_chemical=true)
    list_connectome = [data_connectome_white]
    
    list_neuron = []
    for connectome = list_connectome
        append!(list_neuron, map(x->get_node_name(x["pre"], dv=!merge_dv, lr=!merge_lr), connectome))
        append!(list_neuron, map(x->get_node_name(x["post"], dv=!merge_dv, lr=!merge_lr), connectome))
    end
    list_neuron = sort(unique(list_neuron))
        
    g = py_nx.DiGraph()
    for neuron = list_neuron
        g.add_node(neuron)
    end
        
    dict_synapese = Dict()
    for connectome = list_connectome
        for synapse = connectome
            syn_type = synapse["typ"] == 0 ? "chemical" : "electrical"

            pre = get_node_name(synapse["pre"], dv=!merge_dv, lr=!merge_lr)
            post = get_node_name(synapse["post"], dv=!merge_dv, lr=!merge_lr)
            edge_count = sum(synapse["syn"])
                        
            k = (pre,post,syn_type)
            if haskey(dict_synapese, k)
                dict_synapese[k] += edge_count
            else
                dict_synapese[k] = edge_count
            end
        end
    end
     
    # add edge
    dict_synapse_combine = Dict()
    for ((pre,post,syn_type), edge_count) = dict_synapese
        if pre != post
            if edge_count > min_n_edge
                if syn_type == "chemical" && synapse_chemical
                    if synapse_chemical
                        k = (pre,post)
                        if haskey(dict_synapse_combine, k)
                            dict_synapse_combine[k] += edge_count
                        else
                            dict_synapse_combine[k] = edge_count
                        end                
                    end
                elseif syn_type == "electrical"
                    if synapse_electrical
                        # treat electrical synapse as bidirectional
                        for k = [(pre,post), (post,pre)]
                            if haskey(dict_synapse_combine, k)
                                dict_synapse_combine[k] += edge_count
                            else
                                dict_synapse_combine[k] = edge_count
                            end
                        end
                    end
                else
                    error("unknown synapse type $syn_type")
                end
            end # if edge_count > min_n_edge
        end # if pre != post
    end # for dict_synapese
    
    for ((pre,post), edge_count) = dict_synapse_combine
        g.add_edge(pre, post, weight=(edge_count))  
    end

    # remove orphan node
    for node = collect(g.nodes())
        if g.in_degree(node) + g.out_degree(node) == 0
            g.remove_node(node)
            # println("removing $node")
        end
    end
    
    g
end

function get_graph_white_p(min_n_edge=1; merge_dv=false, merge_lr=false,
    synapse_electrical=true, synapse_chemical=true)
    g_wh_lr_p = get_graph_white(min_n_edge, merge_dv=merge_dv, merge_lr=merge_lr,
        synapse_electrical=synapse_electrical, synapse_chemical=synapse_chemical)
    g = g_wh_lr_p

    # remove orphan node, pharyngeal
    for node = collect(g.nodes())
        if g.in_degree(node) + g.out_degree(node) == 0
            g.remove_node(node)
            # println("removing $node orphaned")
            continue
        end

        if haskey(NeuroPALData.NEURON_REF_DICT, node)
            if NeuroPALData.NEURON_REF_DICT[node]["note"] != "pharynx"
                g.remove_node(node)
                # println("removing $node non-pharynx")
            else
                # println("pharynx node $node")
            end
        else
            g.remove_node(node)
            # println("removing $node not in dict")
        end
    end

    g
end

function get_graph_witvliet(min_n_edge=2; merge_dv=false, merge_lr=false, synapse_electrical=true, synapse_chemical=true)
    list_connectome = [data_connectome_witvliet_7, data_connectome_witvliet_8]
    
    # get list of neurons in the connectome
    list_neuron = []
    for connectome = list_connectome
        append!(list_neuron, map(x->get_node_name(x["pre"], dv=!merge_dv, lr=!merge_lr), connectome))
        append!(list_neuron, map(x->get_node_name(x["post"], dv=!merge_dv, lr=!merge_lr), connectome))
    end
    list_neuron = sort(unique(list_neuron))
    
    g = py_nx.DiGraph()
    for neuron = list_neuron
        g.add_node(neuron)
    end

    dict_synapese = Dict()
    for connectome = list_connectome
        for synapse = connectome
            syn_type = synapse["typ"] == 0 ? "chemical" : "electrical"

            pre = get_node_name(synapse["pre"], dv=!merge_dv, lr=!merge_lr)
            post = get_node_name(synapse["post"], dv=!merge_dv, lr=!merge_lr)
            edge_count = sum(synapse["syn"])
                        
            k = (pre,post,syn_type)
            if haskey(dict_synapese, k)
                dict_synapese[k] += edge_count
            else
                dict_synapese[k] = edge_count
            end
        end
    end
        
    dict_synapse_combine = Dict()
    for ((pre,post,syn_type), edge_count) = dict_synapese
        if pre != post
            if edge_count > min_n_edge
                if syn_type == "chemical"
                    if synapse_chemical
                        k = (pre,post)
                        if haskey(dict_synapse_combine, k)
                            dict_synapse_combine[k] += edge_count
                        else
                            dict_synapse_combine[k] = edge_count
                        end                
                    end
                elseif syn_type == "electrical"
                    if synapse_electrical
                        # treat electrical synapse as bidirectional
                        for k = [(pre,post), (post,pre)]
                            if haskey(dict_synapse_combine, k)
                                dict_synapse_combine[k] += edge_count
                            else
                                dict_synapse_combine[k] = edge_count
                            end
                        end
                    end
                else
                    error("unknown synapse type $syn_type")
                end
            end # if edge_count > min_n_edge
        end # if pre != post
    end
    
    #for ((pre,post,syn_type), edge_count) = dict_synapese
    for ((pre,post), edge_count) = dict_synapse_combine
        g.add_edge(pre, post, weight=edge_count)  
    end

    g
end

function get_sensory_muscle(g)
    list_node = collect(g.nodes)
    list_sensory = String[]
    list_muscle = String[]

    for node = list_node
        type_ = get_neuron_type_wh_dvlr(node)
        if type_ == "sensory"
            push!(list_sensory, node)
        elseif startswith(type_, "muscle")
            push!(list_muscle, node)
        end
    end
    
    list_sensory, list_muscle
end

function get_connectome_plot_lists(dict_x, dict_y, dict_feature, f_feature::Function=identity)
    list_x = Float64[]
    list_y = Float64[]
    list_f = valtype(dict_feature)[]

    for (neuron,x) = dict_x
        if get_neuron_type_wh_dvlr(neuron) !== "motorneuron"
            class, dv, lr = get_neuron_class(neuron)
            
            class_dv = class
            if !(dv == "missing" || dv == "undefined")
                class_dv = class * dv
            end
            
            if haskey(dict_feature, class_dv)
                push!(list_x, x)
                push!(list_y, dict_y[neuron])
                push!(list_f, dict_feature[class_dv])
            else
                # missing
            end

        end
    end
    
    list_x, list_y, list_f
end