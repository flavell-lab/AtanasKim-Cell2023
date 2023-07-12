"""
    get_list_class(list_neuropal_label_to_roi, aggregate=true)

Return a list of all classes in the datasets

# Arguments
- `list_neuropal_label_to_roi`: list of dictionaries mapping neuropal labels to roi
- `aggregate::Bool`: if true, return a list of unique classes, otherwise return a list of list of classes
"""
function get_list_class(list_neuropal_label_to_roi, aggregate=true)
    list_class = map(x->sort(unique(collect(keys(x)))), list_neuropal_label_to_roi)
    if aggregate
        return sort(unique(reduce(vcat, list_class)))
    else
        return list_class
    end
end

"""
    get_list_class_dv(list_neuropal_label_to_roi, aggregate=true)   

Return a list of all classes with D/V in the datasets

# Arguments
- `list_neuropal_label_to_roi`: list of dictionaries mapping neuropal labels to roi
- `aggregate`: if true, return a list of unique classes, otherwise return a list of list of classes
"""
function get_list_class_dv(list_neuropal_label_to_roi, aggregate=true)
    list_class_ = []
    for label_to_roi = list_neuropal_label_to_roi
        list_ = []
        for (class, dict_) = label_to_roi
            for class_dv = unique(map(x->(x["neuron_class"], x["DV"]), dict_))
                if !(class_dv in list_) && class_dv[2] != "missing"
                    push!(list_, class_dv)
                end
            end
        end
        push!(list_class_, sort(map(x->x[2] == "undefined" ? x[1] : x[1] * x[2], list_)))
    end
    
    if aggregate
        return sort(unique(reduce(vcat, list_class_)))
    else
        return list_class_
    end
end

"""
    generate_list_class_custom_order(list_neuropal_order_info, list_class_dv=nothing)

Return a list of all classes with D/V in the datasets

# Arguments
- `list_neuropal_order_info`: custom ordering data
- `list_class_dv`: list of classes with D/V to check for missing classes
"""
function generate_list_class_custom_order(list_neuropal_order_info, list_class_dv=nothing)
    list_class_order = String.(list_neuropal_order_info[2:end,3])
    list_dv_order = String.(list_neuropal_order_info[2:end,4])
    list_lr_order = String.(list_neuropal_order_info[2:end,5])
    
    list_class_ = Tuple[]
    for i = 1:size(list_class_order, 1)
        class = list_class_order[i]
        dv = list_dv_order[i]
        lr = list_lr_order[i]
        
        info_ = Union{String,Nothing}[]
        if dv == "D" || dv == "V"
            @assert class ∈ NeuroPALData.LIST_REF_CLASS_DV
            append!(info_, [class, class * dv, dv])
        elseif dv == "undefined"
            append!(info_, [class, class, "undefined"])
        elseif dv == "nothing"
            append!(info_, [class, class, nothing])
        else
            error("unknown dv configuration for $class")
        end
        
        if lr == "L" || lr == "R"
            @assert class ∈ NeuroPALData.LIST_REF_CLASS_LR
            push!(info_, lr)
            info_[2] *= lr
        else
            if class ∈ NeuroPALData.LIST_REF_CLASS_LR
                push!(info_, nothing)
            else
                push!(info_, "undefined")
            end
        end
        
        push!(list_class_, tuple(info_...))
    end

    # check that no class is missing in the ordering file
    if !isnothing(list_class_dv)
        for class = list_class_dv
            if !(class ∈ map(x->x[2], list_class_))
                @warn "class $(class) is not in the ordering file"
            end
        end
    end

    list_class_
end

"""
    categorize_dorsal_ventral(label, roi_match, roi_match_confidence, θ_confidence, dict_fit)

Categorize a neuron as dorsal or ventral

# Arguments
- `label`: dictionary of label information
- `roi_match`: dictionary of roi matching
- `roi_match_confidence`: dictionary of roi matching confidence
- `θ_confidence`: confidence threshold for roi matching
- `dict_fit`: dictionary of fit information
"""
function categorize_dorsal_ventral(label, roi_match, roi_match_confidence, θ_confidence, dict_fit)
    (roi_gcamp, match_confidence) = match_roi(label["roi_id"],
        roi_match, roi_match_confidence, θ_confidence)
    
    n_d = 0 
    n_v = 0  
    if isa(roi_gcamp, Int)
        for (k,v) =  dict_fit["categorization"] # loop time segments
            b = v["θh"]
            if roi_gcamp in b["all"]
                if roi_gcamp in b["dorsal"]
                   n_d += 1 
                end
                if roi_gcamp in b["ventral"]
                    n_v += 1
                end
            end
        end
    end

    if n_d == 0
        if n_v == 0 # d: 0, v: 0
            return nothing
        else # d: 0, v: 1
            return "V"
        end
    else
        if n_v == 0 # d: 1, v: 0
            return "D"
        else # d: 1, v: 1
            return nothing
        end
    end
end

"""
    get_label_class(class::AbstractString, class_dv::Union{String,Nothing}, list_label,
        roi_match, roi_match_confidence, θ_confidence, dict_fit,
        list_class_classify_dv_enc=nothing)

Return a list of labels that match the class and D/V

Arguments
- `class`: class to match
- `class_dv`: D/V to match
- `list_label`: list of labels
- `roi_match`: dictionary of roi matching
- `roi_match_confidence`: dictionary of roi matching confidence
- `θ_confidence`: confidence threshold for roi matching
- `dict_fit`: dictionary of fit information
- `list_class_classify_dv_enc`: list of classes to classify D/V using encodding
"""
function get_label_class(class::AbstractString, class_dv::Union{String,Nothing}, list_label,
    roi_match, roi_match_confidence, θ_confidence, dict_fit,
    list_class_classify_dv_enc=nothing)
    list_match = []

    if class ∈ NeuroPALData.LIST_REF_CLASS_DV
        for label_ = list_label
            label_dv = if class ∈ list_class_classify_dv_enc
                categorize_dorsal_ventral(label_, roi_match,
                    roi_match_confidence, θ_confidence, dict_fit)
            else
                label_["DV"]
            end
            if label_dv == class_dv
                push!(list_match, label_)
            end
        end
    else # no dv        
        append!(list_match, list_label)
    end

    list_match
end
