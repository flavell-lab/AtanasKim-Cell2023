"""
    fit_decoder(
        fit_results::Dict, datasets::Vector, decode_vars::Vector, t_val::Vector, λ_dict::Dict;
        traces_use::Union{Nothing, Dict}=nothing, v_condition::String="all"
    )

Fit a decoder to predict the values of `decode_vars` from the neural traces in `fit_results` using L1-regularized linear regression.

# Arguments
- `fit_results::Dict`: A dictionary containing the results of fitting CePNEM to the neural data.
- `datasets::Vector`: A vector of dataset names to fit the decoder to.
- `decode_vars::Vector`: A vector of behavior variable names to decode.
- `t_val::Vector`: A vector of timepoints to use for cross-validation.
- `λ_dict::Dict`: A dictionary of regularization strengths for each variable.

# Optional Arguments
- `traces_use::Union{Nothing, Dict}=nothing`: A dictionary of trace arrays to use instead of the ones in `fit_results`.
- `v_condition::String="all"`: The velocity condition to use for selecting timepoints. Can be "all", "fwd", or "rev".

# Returns
- `dict_sol::Dict`: A dictionary of the fitted decoder models.
- `dict_pred::Dict`: A dictionary of the predicted values of `decode_vars` for each timepoint.
- `dict_cost::Dict`: A dictionary of the mean squared error between the predicted and actual values of `decode_vars` for each timepoint.
"""
function fit_decoder(fit_results::Dict, datasets::Vector, decode_vars::Vector, t_val::Vector, λ_dict::Dict;
                    traces_use::Union{Nothing, Dict}=nothing, v_condition::String="all")
    dict_cost = Dict{String, Dict}()
    dict_pred = Dict{String, Dict}()
    dict_sol = Dict{String, Dict}()
    for (i,data_key) in enumerate(decode_vars)
        λ = λ_dict[data_key]
        dict_sol[data_key] = Dict()
        dict_cost[data_key] = Dict()
        dict_pred[data_key] = Dict()
        for dataset in datasets
            data_dict = fit_results[dataset]
            if !isnothing(traces_use)
                trace_array = traces_use[dataset]
            else
                trace_array = deepcopy(data_dict["trace_array"])
            end
            if size(trace_array, 1) == 0
                @warn("No neurons for dataset $dataset")
                continue
            end
            beh = data_dict[data_key]
            timepoints_use=1:size(trace_array,2)
            if v_condition == "fwd"
                timepoints_use = [t for t in timepoints_use if fit_results[dataset]["v"][t] >= 0]
            elseif v_condition == "rev"
                timepoints_use = [t for t in timepoints_use if fit_results[dataset]["v"][t] <= 0]
            end
            quality = zeros(length(λ), length(t_val))
            dict_cost[data_key][dataset] = zeros(length(λ), length(t_val))
            dict_pred[data_key][dataset] = zeros(length(λ), length(t_val), 1600)
            dict_sol[data_key][dataset] = []
            if var(beh) == 0
                continue
            end
            beh_z = (beh .- mean(beh[timepoints_use])) ./ sqrt(mean((beh[timepoints_use] .- mean(beh[timepoints_use])) .^ 2))
            for (k,t_eval) = enumerate(t_val)
                t_train = [t for t in timepoints_use if !(t in t_eval)]
                t_eval_use = [t for t in t_eval if t in timepoints_use]
                data_train = beh_z[t_train]
                data_eval = beh_z[t_eval_use]

                var_solution = glmnet(transpose(trace_array[:,t_train]), data_train, lambda=λ, intercept=false)
                push!(dict_sol[data_key][dataset], var_solution)
                
                dict_pred[data_key][dataset][:,k,:] .= transpose(GLMNet.predict(var_solution, transpose(trace_array)))
                
                
                for j=1:length(λ)
                    dict_cost[data_key][dataset][j,k] = cost_mse(dict_pred[data_key][dataset][j,k,t_eval_use], data_eval)
                end
            end
        end
    end
    return dict_sol, dict_pred, dict_cost
end

"""
    compute_variance_explained(dict_cost::Dict, fit_results::Dict, t_val::Vector, decode_vars::Vector, λ_dict::Dict; min_P_variance::Float64=0.5)

Compute the variance explained by the decoder for each dataset, behavior variable, regularization strength, and timepoint.

# Arguments
- `dict_cost::Dict`: A dictionary of the mean squared error between the predicted and actual values of `decode_vars` for each timepoint.
- `fit_results::Dict`: A dictionary containing the results of fitting CePNEM to the neural data.
- `t_val::Vector`: A vector of timepoints to use for cross-validation.
- `decode_vars::Vector`: A vector of behavior variable names to decode.
- `λ_dict::Dict`: A dictionary of regularization strengths for each variable.

# Optional Arguments
- `min_P_variance::Float64=0.5`: The minimum variance of the `P` variable for a dataset to be included in the analysis.

# Returns
- `dict_quality::Dict`: A dictionary of the variance explained by the decoder for each dataset, behavior variable, regularization strength, and timepoint.
"""
function compute_variance_explained(dict_cost::Dict, fit_results::Dict, t_val::Vector, decode_vars::Vector, λ_dict::Dict; min_P_variance::Float64=0.5)
    dict_quality = Dict{String, Dict}()
    for (i,beh) in enumerate(decode_vars)
        dict_quality[beh] = Dict{Int, Dict}()
        for (j,λ) in enumerate(λ_dict[beh])
            dict_quality[beh][j] = Dict{String, Any}()
            dict_quality[beh][j]["overall"] = Float64[]
            dict_quality[beh][j]["overall_dataset"] = Float64[]
            for dataset in keys(dict_cost[beh])
                dict_quality[beh][j][dataset] = Dict{Int, Float64}()
                if beh == "P" && std(fit_results[dataset]["P"]) < min_P_variance
                    continue
                end
                for (k,t_eval) = enumerate(t_val)
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

"""
    average_dict_qualities(dict_qualities::Dict)

Compute the average variance explained by the decoder for each dataset, behavior variable, regularization strength, and timepoint.

# Arguments
- `dict_qualities::Dict`: A dictionary of the variance explained by the decoder for each dataset, behavior variable, regularization strength, and timepoint.

# Returns
- `dict_quality::Dict`: A dictionary of the average variance explained by the decoder for each dataset, behavior variable, regularization strength, and timepoint.
"""
function average_dict_qualities(dict_qualities::Dict)
    dict_quality = Dict()
    for beh in keys(dict_qualities[1])
        dict_quality[beh] = Dict()
        for j in keys(dict_qualities[1][beh])
            dict_quality[beh][j] = Dict()
            for dataset in keys(dict_qualities[1][beh][j])
                if (dataset in ["overall", "overall_dataset"])
                    dict_quality[beh][j][dataset] = mean([dict_qualities[i][beh][j][dataset] for i in 1:length(dict_qualities)])
                else
                    dict_quality[beh][j][dataset] = Dict()
                    for k in keys(dict_qualities[1][beh][j][dataset])
                        dict_quality[beh][j][dataset][k] = mean([dict_qualities[i][beh][j][dataset][k] for i in 1:length(dict_qualities)])
                    end
                end
            end
        end
    end
    return dict_quality
end
