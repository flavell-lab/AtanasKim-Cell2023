const path_neuron_class_all = joinpath(dirname(@__FILE__), "reference", "neuron_class_all.csv")

const LIST_REF_CLASS_SEXTET = ["IL1", "IL2", "RMD", "GLR"]

const LIST_REF_NEURON, LIST_REF_CLASS, LIST_REF_CLASS_LR, LIST_REF_CLASS_DV, NEURON_REF_DICT = let
    csv_ = readdlm(path_neuron_class_all,',')
    list_neuron = String.(csv_[2:end,1])
    
    list_class_lr = String[]
    list_class_dv = String[]
    
    neuron_dict = Dict{String, Dict{String,String}}()
    for i = 2:size(csv_,1)
        DV = csv_[i,3]
        LR = csv_[i,4]
        DV = DV == "missing" ? "undefined" : DV
        LR = LR == "missing" ? "undefined" : LR
        dict_ = Dict("class"=>csv_[i,2], "DV"=>DV, "LR"=>LR,
            "type"=>csv_[i,5], "category"=>csv_[i,6], "note"=>csv_[i,7])
        neuron_dict[csv_[i,1]] = dict_
        
        if LR != "undefined"
            push!(list_class_lr, csv_[i,2])
        end
        if DV != "undefined"
            push!(list_class_dv, csv_[i,2])
        end
    end
    
    list_class = unique(csv_[2:end,2])
    list_class_lr = unique(list_class_lr)
    list_class_dv = unique(list_class_dv)
    list_neuron, list_class, list_class_lr, list_class_dv, neuron_dict
end


function string_char_n_match(s1::String, s2::String)
    n = 0
    for i = 1:min(length(s1), length(s2))
        if s1[i] == s2[i]
            n += 1
        else
            return n
        end
    end
    
    return n
end

"""
    invert_left_right(neuron::String)

Invert the left-right orientation of a neuron name.

e.g. `invert_left_right("RMEL")` returns `("RMER", true)
# Arguments
- `neuron::String`: Neuron label (e.g. RME)
"""
function invert_left_right(neuron)
    class, LR, DV = get_neuron_class(neuron)
    if class ∈ list_ref_class_lr
        if neuron[end] == 'L'
            return neuron[1:end-1] * 'R', true
        elseif neuron[end] == 'R'
            return neuron[1:end-1] * 'L', true
        else
            error("LR should be L or R but was given $(neuron[end])")
        end
    end
    
    return neuron, false
end

"""
    invert_dorsal_ventral(neuron::String)

Invert the dorsal-ventral orientation of a neuron name.

e.g. `invert_dorsal_ventral("RMEV")` returns `("RMED", true)
# Arguments
- `neuron::String`: Neuron label (e.g. RME)
"""
function invert_dorsal_ventral(neuron)
    class, LR, DV = get_neuron_class(neuron)
    
    neuron_ = collect(neuron)
    idx_adjust = length(neuron)
    q_flip = false
    if class ∈ LIST_REF_CLASS_DV
        if class ∈ LIST_REF_CLASS_LR && class * "DL" ∈ LIST_REF_CLASS
            idx_adjust -= 1
        end
        
        if neuron_[idx_adjust] == 'D'
            neuron_[idx_adjust] = 'V'
        elseif neuron_[idx_adjust] == 'V'
            neuron_[idx_adjust] = 'D'
        else
            error("DV should be D or V but was given $(neuron_[idx_adjust])")
        end
        
        q_flip = true
    end
    
    return join(neuron_), q_flip
end

"""
    get_neuron_class(neuron::String)

Returns (neuron class, DV, LR)

e.g. `get_neuron_class("RMEL")` returns `("RME", "L", "undefined")
"undefined": D/V or L/R is not defined for the neuron (e.g. "AVA" would return "undefined" for D/V")
"missing": D/V or L/R is missing for the neuron (i.e. was not able to label)
# Arguments
- `neuron::String`: Neuron label (e.g. RME)
"""
function get_neuron_class(neuron)
    list_class = sort(unique([v["class"] for (k,v) = NEURON_REF_DICT]))
    neuron_ = occursin("-", neuron) ? split(neuron, "-")[1] : neuron
    neuron_ = String(strip(neuron_))
    class = neuron_
    LR = "undefined"
    DV = "undefined"
  
    if haskey(NEURON_REF_DICT, neuron_)
        neuron_info = NEURON_REF_DICT[neuron_]
        class = neuron_info["class"]
        LR = neuron_info["LR"]
        DV = neuron_info["DV"]
    elseif occursin('?', neuron_)
        n_match, idx_max = findmax(string_char_n_match.(neuron_, list_class))
        if n_match > 1
            class = list_class[idx_max]
            @assert length(class) == n_match
            @assert 0 < length(neuron_) - n_match < 3
            q_lr = class ∈ LIST_REF_CLASS_LR
            q_dv = class ∈ LIST_REF_CLASS_DV
            
            if length(neuron_) - n_match == 2
                @assert q_lr && q_dv
                LR = neuron_[end] == '?' ? "missing" : join(neuron_[end])
                DV = neuron_[end-1] == '?' ? "missing" : join(neuron_[end-1])
            else # D or V
                @assert q_lr || q_dv
                if q_lr
                    LR = neuron_[end] == '?' ? "missing" : join(neuron_[end])
                else
                    DV = neuron_[end] == '?' ? "missing" : join(neuron_[end])
                end
            end
        else
            error("neuron $(neuron_) is unknown")
        end
    else
        error("neuron $(neuron_) is unknown")
    end
    
    class, DV, LR
end
