zstd = FlavellBase.standardize

function get_idx_splits(data_dict, n_recording=1)
    if n_recording == 1
        return [1:data_dict["max_t"]]
    else
        idx_splits = Vector{UnitRange{Int64}}(undef, n_recording)
        rg_end = 1
        for i = 1:n_recording
            idx_splits[i] = rg_end:(rg_end-1)+data_dict["max_t_$i"]
            rg_end = data_dict["max_t_$i"] + 1
        end
        @assert(idx_splits[end][end] == data_dict["max_t"])

        return idx_splits
    end
end

function trace_array_rm_nan_neuron(trace::Array{<:AbstractFloat,2}, dim_time=2)
    trace_ = trace[(dropdims(sum(isnan.(trace), dims=dim_time), dims=dim_time) .== 0), :]

    match_org_to_skip = Dict()
    match_skip_to_org = Dict()
    for (x,y) = zip(1:size(trace,1),
            findall((dropdims(sum(isnan.(trace), dims=dim_time), dims=dim_time) .== 0)))
        match_org_to_skip[y] = x
        match_skip_to_org[x] = y
    end
    
    trace_, match_org_to_skip, match_skip_to_org
end

function filter_velocity(velocity::Vector{T}, derivative_threshold=0.2) where T
    v_filt = Vector{Union{Missing,Float64}}(undef, length(velocity))
    v_filt .= velocity
    
    n_t = length(velocity)
    
    list_idx_rm = []
    for i = findall(abs.(diff(velocity)) .> derivative_threshold)
        push!(list_idx_rm, i:min(i+4, n_t))
    end
    list_idx_rm = union(vcat(list_idx_rm...))
    
    v_filt[list_idx_rm] .= missing
    
    v_filt = Impute.impute(v_filt, Impute.Interpolate())

    n_missing_seg = 0
    t_last = -1
    for i = n_t:-1:1
        if ismissing(v_filt[i])
            n_missing_seg += 1
            v_filt[i] = 0.
            t_last = i
        else
            break
        end
    end
    
    if t_last > 0
        v_filt[t_last:end] .= mean(v_filt[t_last-6:t_last-1])
    end
    
    if n_missing_seg > 3
        @warn("the last segment (n=$n_missing_seg) had a velocity spike that could not be imputed after filtering. Setting to mean from $(t_last-6:t_last)")
    end
    
    convert(Vector{T}, v_filt)
end


function concrete_type(array::Array{Any})
    list_type = union(typeof.(array))
    if length(list_type) > 1
        error("more than 1 type exists in the array")
    else
        return convert.(list_type[1], array)
    end
end
