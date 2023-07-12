function fit_decoder(fit_results, datasets, decode_vars, t_val, λ_dict; traces_use=nothing)
    dict_cost = Dict()
    dict_pred = Dict()
    dict_sol = Dict()
    @showprogress for (i,data_key) in enumerate(decode_vars)
        λ = λ_dict[data_key]
        dict_sol[data_key] = Dict()
        dict_cost[data_key] = Dict()
        dict_pred[data_key] = Dict()
        for dataset = datasets
            data_dict = fit_results[dataset]
            if !isnothing(traces_use)
                trace_array = traces_use[dataset]
            else
                trace_array = deepcopy(data_dict["trace_array"])
            end
            beh = data_dict[data_key]
            quality = zeros(length(λ), length(t_val))
            dict_cost[data_key][dataset] = zeros(length(λ), length(t_val))
            dict_pred[data_key][dataset] = zeros(length(λ), length(t_val), 1600)
            dict_sol[data_key][dataset] = []
            if var(beh) == 0
                continue
            end
            beh_z = (beh .- mean(beh)) ./ sqrt(mean((beh .- mean(beh)) .^ 2))
            for (k,t_eval) = enumerate(t_val)
                t_train = [t for t in 1:1600 if !(t in t_eval)]
                data_train = beh_z[t_train]
                data_eval = beh_z[t_eval]

                var_solution = glmnet(transpose(trace_array[:,t_train]), data_train, lambda=λ, intercept=false)
                push!(dict_sol[data_key][dataset], var_solution)
                
                dict_pred[data_key][dataset][:,k,:] .= transpose(GLMNet.predict(var_solution, transpose(trace_array)))
                
                
                for j=1:length(λ)
                    dict_cost[data_key][dataset][j,k] = cost_mse(dict_pred[data_key][dataset][j,k,t_eval], data_eval)
                end
            end
        end
    end
    return dict_sol, dict_pred, dict_cost
end

function compute_variance_explained(dict_cost, fit_results, t_eval, decode_vars, λ_dict; min_P_variance=0.5)
    dict_quality = Dict()
    for (i,beh) in enumerate(decode_vars)
        dict_quality[beh] = Dict()
        for (j,λ) in enumerate(λ_dict[beh])
            dict_quality[beh][j] = Dict()
            dict_quality[beh][j]["overall"] = Float64[]
            dict_quality[beh][j]["overall_dataset"] = Float64[]
            for dataset in keys(dict_cost[beh])
                dict_quality[beh][j][dataset] = Dict()
                if beh == "P" && std(fit_results[dataset]["P"]) < min_P_variance
                    continue
                end
                for (k,t_eval) = enumerate(t_val)
#                     if beh == "P" && (std(fit_results[dataset]["P"][t_eval]) < min_P_variance || std(fit_results[dataset]["P"][[t for t=1:1600 if !(t in t_eval)]]) < min_P_variance)
#                         continue
#                     end
                    varexp = 1 - dict_cost[beh][dataset][j,k]# ./ var(zscore(fit_results[dataset][beh])[t_eval])
                    dict_quality[beh][j][dataset][k] = varexp
                    push!(dict_quality[beh][j]["overall"], varexp)
                end
                push!(dict_quality[beh][j]["overall_dataset"], mean(values(dict_quality[beh][j][dataset])))
            end
        end
    end
    return dict_quality
end