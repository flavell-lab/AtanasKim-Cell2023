function idx_splitify_rng(rng, idx_splits, ewma_trim)
    rng_new = Int32[]
    for t in rng
        use = true
        for split in idx_splits
            if t in split[1]:split[1]+ewma_trim
                use = false
                break
            end
        end
        if use
            push!(rng_new, t)
        end
    end
    return rng_new
end

function generate_reg_L2_nl10d(list_idx_ps::Union{Vector{Int}, UnitRange{Int64}})
    idx_ = setdiff(list_idx_ps, 5:6)

    function (ps_::Vector{T}) where T
        ps = zeros(T,4)
        ps[idx_] .= ps_

        ((ps[2]) ^ 2 + (ps[3]) ^ 2 + (ps[4]) ^ 2 +
            (ps[1] * ps[2]) ^ 2 + (ps[1] * ps[3]) ^ 2 + (ps[1] * ps[4]) ^ 2 + 
            (2 * ps[1] * ps[2]) ^ 2 + (2 * ps[1] * ps[3]) ^ 2 + (2 * ps[1] * ps[4]) ^ 2) / 
            (1 + ps[1] ^ 2)
    end
end


function fit_model(trace, xs, xs_s, f_init_ps, f_generate_model, f_generate_reg,
        idx_splits, idx_train, idx_tests, ewma_trim, λ_reg; opt_g=:G_MLSL_LDS, opt_l=:LD_LBFGS, max_eval=5000)
    # get thresholds
    list_θ = zeros(eltype(xs), 5)

    # train
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = f_init_ps(xs)
    f = f_generate_model(xs_s, list_θ, ewma_trim, idx_splits)

    f_reg = f_generate_reg(list_idx_ps_reg)

    res = fit_model_nlopt_bound_reg(trace, f, cost_mse, ps_0, ps_min, ps_max, 
        idx_train, idx_train, Symbol(opt_g), Symbol(opt_l), max_time=240, max_eval=max_eval,
        λ=λ_reg, f_reg=f_reg, idx_ps=list_idx_ps_reg, xtol=1e-7, ftol=1e-15)



    u_opt = res[1][2]
    n_eval = res[2]

    # evaluate cost
    y = trace
    y_pred = f(u_opt)

    # test
    cost_tests = Float64[]
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = f_init_ps(xs)
    y_test_pred = f_generate_model(xs_s, list_θ, ewma_trim, [1:size(xs_s,2)])(u_opt)
    for idx_test in idx_tests
        push!(cost_tests, cost_mse(trace[idx_test], y_test_pred[idx_test]))
    end

    cost_train = cost_mse(y, y_pred, idx_train, idx_train)

    return (cost_train, cost_tests, u_opt, n_eval)::Tuple{Float64, Vector{Float64}, Vector{Float64}, Int32}
end


"""
Uses MSE minimization to fit models of neural activity over the full time range.
"""
function fit_mse_models(fit_results, analysis_dict, datasets, get_h5_data; ewma_trim=50, λ_reg=0)
    mse_fits_combinedtrain = Dict()
    @showprogress for data_uid = datasets
        mse_fits_combinedtrain[data_uid] = Dict()
        
        data_dict = get_h5_data(data_uid)
        P_thresh = 0
    
        n_neuron = data_dict["n_neuron"]
        idx_splits = data_dict["idx_splits"]
        idx_splits_trim = trim_idx_splits(idx_splits, (50,0))
        
        # set up behavior
        n_t = maximum(idx_splits[end])
        xs = zeros(5, n_t)
        xs[1,:] = data_dict["velocity"]
        xs[2,:] = data_dict["θh"]
        xs[3,:] = data_dict["pumping"]
        xs[4,:] = data_dict["ang_vel"]
        xs[5,:] = data_dict["curve"]
        
        xs_s = deepcopy(xs)
        xs_s[1,:] .= xs_s[1,:] ./ v_STD
        xs_s[2,:] .= xs_s[2,:] ./ θh_STD
        xs_s[3,:] .= xs_s[3,:] ./ P_STD
        
        
        for rng1=1:length(fit_results[data_uid]["ranges"])
            for rng2=rng1+1:length(fit_results[data_uid]["ranges"])
                rngs = collect(deepcopy(fit_results[data_uid]["ranges"][rng1]))
                append!(rngs, fit_results[data_uid]["ranges"][rng2])
                idx_train = idx_splitify_rng(rngs, idx_splits, ewma_trim)
    
                mse_fits_combinedtrain[data_uid][(rng1,rng2)] = Dict()
                for idx_neuron = analysis_dict["encoding_changes_corrected"][data_uid][(rng1,rng2)]["all"]
                    trace_array = data_dict["trace_array"]
                    trace = trace_array[idx_neuron,:]
    
                    f_generate_model = generate_model_nl10d
                    f_init_ps = init_ps_model_nl10d
                    f_generate_reg = generate_reg_L2_nl10d
    
                    mse_fits_combinedtrain[data_uid][(rng1,rng2)][idx_neuron] = Dict()
    
                    idx_tests = [idx_splitify_rng(rng, idx_splits, ewma_trim) for rng in fit_results[data_uid]["ranges"]]
    
                    res = fit_model(trace, xs, xs_s, f_init_ps,
                            f_generate_model,
                            f_generate_reg,
                            idx_splits, idx_train, idx_tests,
                            ewma_trim, λ_reg, max_eval=5000)
                    (cost_train, cost_tests, u_opt, n_eval) = res
    
                    mse_fits_combinedtrain[data_uid][(rng1,rng2)][idx_neuron]["cost_train"] = cost_train
                    mse_fits_combinedtrain[data_uid][(rng1,rng2)][idx_neuron]["cost_test"] = cost_tests
                    mse_fits_combinedtrain[data_uid][(rng1,rng2)][idx_neuron]["u_opt"] = u_opt
                    mse_fits_combinedtrain[data_uid][(rng1,rng2)][idx_neuron]["n_eval"] = n_eval
                end
            end
        end
    end
    return mse_fits_combinedtrain
end


"""
Computes MSE of the Gen model fits to the data.
"""
function compute_CePNEM_MSE(fit_results, analysis_dict, datasets, get_h5_data; ewma_trim=50, reset_param_6::Bool=true)
    mse_ewma_skip = Dict()
    @showprogress for dataset = datasets
        if !(dataset in keys(mse_ewma_skip))
            mse_ewma_skip[dataset] = Dict()
        end
        data_dict = get_h5_data(dataset) 
        idx_splits = data_dict["idx_splits"]

        neurons_ec = Int32[]

        for rng1 in 1:length(fit_results[dataset]["ranges"])-1
            for rng2 in rng1+1:length(fit_results[dataset]["ranges"])
                neurons_ec = union(neurons_ec, analysis_dict["encoding_changes_corrected"][dataset][(rng1,rng2)]["all"])
            end
        end
        

        for neuron in neurons_ec
            mse_ewma_skip[dataset][neuron] = zeros(length(fit_results[dataset]["ranges"]),length(fit_results[dataset]["ranges"]),size(fit_results[dataset]["sampled_trace_params"],3))
            for rng1=1:length(fit_results[dataset]["ranges"])
                for idx=1:size(fit_results[dataset]["sampled_trace_params"],3)
                    ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng1,neuron,idx,1:8])
                    if reset_param_6
                        ps[6] = 0
                    end
    
                    model = model_nl8(size(fit_results[dataset]["trace_array"],2), ps..., fit_results[dataset]["v"], fit_results[dataset]["θh"], fit_results[dataset]["P"])
                    for rng2=1:length(fit_results[dataset]["ranges"])
                        rngs_timepts = idx_splitify_rng(collect(deepcopy(fit_results[dataset]["ranges"][rng2])), idx_splits, ewma_trim) 
                        mse_ewma_skip[dataset][neuron][rng1,rng2,idx] = cost_mse(fit_results[dataset]["trace_array"][neuron,rngs_timepts], model[rngs_timepts])
                    end
                end
            end
        end
    end
    return mse_ewma_skip
end