function extrapolate_neurons(consistent_neurons, parameters, rngs_use, fit_results, θh_pos_is_ventral)
    all_params = zeros(sum([length(v) for v in values(consistent_neurons)]), 11)
    ids = []
    all_behs = zeros(1600 * length(keys(consistent_neurons)), 5)
    idx = 1
    idx_beh = 1
    rngs_valid = [5,6]
    @showprogress for dataset in keys(consistent_neurons)
        rng = rngs_use[dataset]
        for n=consistent_neurons[dataset]
            all_params[idx,:] .= parameters[dataset][n]
            push!(ids, (dataset, rng, n))
            idx += 1
        end
        len = length(fit_results[dataset]["v"])
        all_behs[idx_beh:idx_beh+len-1, 1] .= fit_results[dataset]["v"]
        all_behs[idx_beh:idx_beh+len-1, 2] .= fit_results[dataset]["θh"] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+len-1, 3] .= fit_results[dataset]["P"]
        all_behs[idx_beh:idx_beh+len-1, 4] .= fit_results[dataset]["ang_vel"] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+len-1, 5] .= fit_results[dataset]["curve"]
        idx_beh += len
    end
    all_params = all_params[1:idx-1,:]
    all_behs = all_behs[1:idx_beh-1,:]

    models = zeros(size(all_params,1), size(all_behs,1))
    for i=1:size(models,1)
        ps = deepcopy(all_params[i,1:8])
        ps[3] = ps[3] * (2*θh_pos_is_ventral[ids[i][1]]-1)
        ps[6] = 0
        models[i,:] .= model_nl8(size(all_behs,1), ps..., all_behs[:,1], all_behs[:,2], all_behs[:,3])
    end
    return models, ids, all_behs, all_params, fit(PCA, models)
end

function make_distance_matrix(models)
    distance_matrix = zeros(size(models,1), size(models,1))
    norms = Dict()
    @showprogress for i=1:size(models,1)-1
        if !(i in keys(norms))
            norms[i] = sum(models[i,:] .^ 2)
        end
        for j=i+1:size(models,1)
            if !(j in keys(norms))
                norms[j] = sum(models[j,:] .^ 2)
            end
            distance_matrix[i,j] = 1 - sum(models[i,:] .* models[j,:]) / sqrt(norms[i] * norms[j])
            distance_matrix[j,i] = distance_matrix[i,j]
        end
    end
    return distance_matrix
end

function invert_id(id, ids)
    # ignore ranges
    return [i for i=1:length(ids) if ids[i][1] == id[1] && ids[i][3] == id[2]][1]
end

function invert_array(arr)
    inv = zeros(eltype(arr), length(arr))
    for i=1:length(inv)
        inv[arr[i]] = i
    end
    return inv
end
