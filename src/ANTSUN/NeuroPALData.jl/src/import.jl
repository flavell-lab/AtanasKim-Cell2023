function get_neuron_roi(roi)
    if isa(roi, AbstractString)
        if occursin("/", roi)
            return parse.(Int, split(roi, "/"))
        else
            return [parse(Int, roi)]
        end
    elseif isa(roi, Int)
        return [roi]
    elseif isa(roi, Float64)
        if isinteger(roi)
            return [convert(Int, roi)]
        else
            error("ROI($roi) is not integer")
        end
    else
        error("unknown data type for neuron ROI")
    end
end

"""
    import_neuropal_label(path_label::String)

Import NeuroPAL label data from a file.

Returns neuropal_roi_to_label, neuropal_label_to_roi
# Arguments
- `path_label::String`: path to the label file. Supported: csv, xlsx
"""
function import_neuropal_label(path_label::String; verbose=true)
    if endswith(path_label, ".xlsx")
        list_sheets = XLSX.openxlsx(path_label, mode="r") do xlsx
            XLSX.sheetnames(xlsx)
        end
        list_sheets_label = sort(filter(x->occursin("labels",x) && !occursin("progress",x), list_sheets))
        sheet_ = list_sheets_label[end]
        println("reading $(sheet_) for $path_label")
        sheet_label = XLSX.readtable(path_label, sheet_)
        col_label = sheet_label.column_labels
        data_ = vcat(reshape(string.(col_label), (1,length(col_label))),
            hcat(sheet_label.data...))
        
        import_neuropal_label(data_, verbose=verbose)
    elseif endswith(path_label, ".csv")
        data_ = readdlm(path_label, ',')
        import_neuropal_label(data_)
    else
        error("unsupported data type. supported: csv, xlsx")
    end
end

"""
    import_neuropal_label(data_::Matrix)

Import NeuroPAL label data from a matrix.

Returns neuropal_roi_to_label, neuropal_label_to_roi
# Arguments
- `data_::Matrix`: a matrix of data. The first row is the column label.
"""
function import_neuropal_label(data_::Matrix; verbose=true)
    neuropal_roi_to_label = Dict{Int, Vector{Dict}}()
    list_roi = get_neuron_roi.(data_[2:end,3])
    list_roi_flat = sort(vcat(list_roi...))

    # check if there are repeated ROI
    list_roi_repeat = unique(filter(x->count(x .== list_roi_flat) > 1, list_roi_flat))
    list_roi_norepeat = sort(setdiff(unique(list_roi_flat), list_roi_repeat))
    if length(list_roi_repeat) > 0 && verbose
        @warn("ROI $(list_roi_repeat) are repeated. Automatically excluded.")
    end

    # check if there are repeated label
    for (k,v) = countmap(data_[2:end,1])
        if v > 1 && verbose
            @warn("Label $k is repeated $v times. NOT automatically excluded.")
        end
    end

    for roi_id = list_roi_norepeat
        idx_row_match = findall(roi_id .∈ list_roi)        
        list_match = Dict{String,Any}[]

        for i_row = (idx_row_match .+ 1) # offset column label row
            label = data_[i_row,1]
            neuron_class, DV, LR = get_neuron_class(label)
            roi_id_ = data_[i_row,3]
            confidence = data_[i_row,4]
            comment = data_[i_row,5]
            region = data_[i_row,6]
            
            match_ = Dict{}()
            match_["label"] = label
            match_["roi_id"] = get_neuron_roi(roi_id_)
            match_["confidence"] = confidence
            match_["region"] = region
            match_["neuron_class"] = neuron_class
            match_["LR"] = LR
            match_["DV"] = DV
            
            push!(list_match, match_)
        end        
        
        neuropal_roi_to_label[roi_id] = list_match
    end
    
    neuropal_label_to_roi = Dict{String, Any}()
    list_class = map(x->get_neuron_class(x)[1], data_[2:end, 1])
    for class = unique(list_class)
        idx_row_match = findall(class .== list_class)
        list_match = Dict{String,Any}[]
        for i_row = (idx_row_match .+ 1) # offset column label
            label = data_[i_row,1]
            neuron_class, DV, LR = get_neuron_class(label)
            roi_id_ = data_[i_row,3]
            confidence = data_[i_row,4]
            comment = data_[i_row,5]
            region = data_[i_row,6]

            match_ = Dict{}()
            match_["label"] = label
            match_["roi_id"] = get_neuron_roi(roi_id_)
            match_["confidence"] = confidence
            match_["region"] = region
            match_["neuron_class"] = neuron_class
            match_["LR"] = LR
            match_["DV"] = DV

            # println("$label, $DV - $(typeof(DV))")

            # add only if not repeated
            if all([!(roi in list_roi_repeat) for roi = get_neuron_roi(roi_id_)])
                push!(list_match, match_)
            end
        end        

        neuropal_label_to_roi[class] = list_match
    end
    
    neuropal_roi_to_label, neuropal_label_to_roi
end
