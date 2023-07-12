VALID_V_COMPARISONS = [(1,2), (2,1), (3,4), (4,3)]

"""
Evaluate model NL8, deconvolved, given `params` and `v`, `θh`, and `P`. Does not use the sigmoid.
"""
function deconvolved_model_nl8(params::Vector{Float64}, v::Float64, θh::Float64, P::Float64)
    return ((params[1]+1)/sqrt(params[1]^2+1) - 2*params[1]/sqrt(params[1]^2+1)*(v/v_STD < 0)) * (
        params[2] * (v/v_STD) + params[3] * (θh/θh_STD) + params[4] * (P/P_STD) + params[5]) + params[8]
end

"""
Computes the valid range of a behavior `beh` (eg: velocity cropped to a given time range).
Computes percentile based on `beh_percent`, and uses 4 points instead of 2 for velocity (`beh_idx = 1`)
"""
function compute_range(beh::Vector{Float64}, beh_percent::Real, beh_idx::Int)
    @assert(beh_percent < 50)
    if beh_idx == 1
        min_beh = percentile(beh[beh .< 0], 2*beh_percent)
        max_beh = percentile(beh[beh .> 0], 100-2*beh_percent)
    else
        min_beh = percentile(beh, beh_percent)
        max_beh = percentile(beh, 100-beh_percent)
    end
    
    if beh_idx == 1
        return [min_beh, min_beh/100, max_beh/100, max_beh]
    else
        return [min_beh, max_beh]
    end
end

"""
Computes deconvolved activity of model NL8 for each sampled parameter in `sampled_trace_params`,
at a lattice defined by `v_rng`, `θh_rng`, and `P_rng`.
"""
function get_deconvolved_activity(sampled_trace_params, v_rng, θh_rng, P_rng)
    n_traces = size(sampled_trace_params,1)
    deconvolved_activity = zeros(n_traces, length(v_rng), length(θh_rng), length(P_rng))
    for x in 1:n_traces
        for (i,v_) in enumerate(v_rng)
            for (j,θh_) in enumerate(θh_rng)
                for (k,P_) in enumerate(P_rng)
                    deconvolved_activity[x,i,j,k] = deconvolved_model_nl8(sampled_trace_params[x,:],v_,θh_,P_)
                end
            end
        end
    end

    return deconvolved_activity
end


"""
Makes deconvolved lattices for each dataset, time range, and neuron in `fit_results`.
Returns velocity, head angle, and pumping ranges, and the deconvolved activity of each neuron at each lattice point defined by them,
for both statistically useful ranges (first return value), and full ranges (second return value) designed for plotting consistency.
Set `dataset_mapping` to change which behavioral dataset to use for computing ranges.
"""
function make_deconvolved_lattice(fit_results, beh_percent, plot_thresh; dataset_mapping=nothing)
    deconvolved_activity = Dict()
    v_ranges = Dict()
    θh_ranges = Dict()
    P_ranges = Dict()
    deconvolved_activity_plot = Dict()
    v_ranges_plot = Dict()
    θh_ranges_plot = Dict()
    P_ranges_plot = Dict()
    @showprogress for dataset in keys(fit_results)
        deconvolved_activity[dataset] = Dict()
        v_ranges[dataset] = Dict()
        θh_ranges[dataset] = Dict()
        P_ranges[dataset] = Dict()

        deconvolved_activity_plot[dataset] = Dict()

        dset = (isnothing(dataset_mapping) ? dataset : dataset_mapping[dataset])

        v_all = fit_results[dset]["v"]
        θh_all = fit_results[dset]["θh"]
        P_all = fit_results[dset]["P"]
        
        v_ranges_plot[dataset] = Dict()
        θh_ranges_plot[dataset] = Dict()
        P_ranges_plot[dataset] = Dict()

        for rng=1:length(fit_results[dataset]["ranges"])
            deconvolved_activity[dataset][rng] = Dict()

            deconvolved_activity_plot[dataset][rng] = Dict()

            v_ranges_plot[dataset][rng] = compute_range(v_all, plot_thresh, 1)
            θh_ranges_plot[dataset][rng] = compute_range(θh_all, plot_thresh, 2)
            P_ranges_plot[dataset][rng] = compute_range(P_all, plot_thresh, 3)

            v = v_all[fit_results[dataset]["ranges"][rng]]
            θh = θh_all[fit_results[dataset]["ranges"][rng]]
            P = P_all[fit_results[dataset]["ranges"][rng]]

            results = fit_results[dataset]["sampled_trace_params"]

            v_ranges[dataset][rng] = compute_range(v, beh_percent, 1)
            θh_ranges[dataset][rng] = compute_range(θh, beh_percent, 2)
            P_ranges[dataset][rng] = compute_range(P, beh_percent, 3)
            
            

            for neuron=1:size(results,2)
                deconvolved_activity[dataset][rng][neuron] =
                        get_deconvolved_activity(results[rng,neuron,:,:], v_ranges[dataset][rng],
                                θh_ranges[dataset][rng], P_ranges[dataset][rng])
                
                deconvolved_activity_plot[dataset][rng][neuron] =
                        get_deconvolved_activity(results[rng,neuron,:,:], v_ranges_plot[dataset][rng],
                                θh_ranges_plot[dataset][rng], P_ranges_plot[dataset][rng])
            end
        end
    end
    return (v_ranges, θh_ranges, P_ranges, deconvolved_activity), (v_ranges_plot, θh_ranges_plot, P_ranges_plot, deconvolved_activity_plot)
end


"""
Computes neuron p-values by comparing differences between two different deconvolved activities to a threshold.
(Ie: the p-value of rejecting the null hypothesis that the difference is negative or less than the threshold - 
if p=0, then we can conclude the neuron has the given activity.)
To find encoding of a neuron, set the second activity to 0.
To find encoding change, set it to a different time window.
To find distance between neurons, set `compute_p = false` and specify the `metric` (default `abs`)
to use to compare medians of the two posteriors.
To use a statistic other than median, change `stat`.
"""
function neuron_p_vals(deconvolved_activity_1, deconvolved_activity_2, signal, threshold_artifact::Real, threshold_weak::Real, relative_encoding_strength; compute_p::Bool=true, metric::Function=abs, stat::Function=median)
    categories = Dict()
    
    s = size(deconvolved_activity_1)
    categories["v_encoding"] = compute_p ? ones(s[2], s[2], s[3], s[4]) : zeros(s[2], s[2], s[3], s[4])

    v_ratio = relative_encoding_strength["std_deconv_v"] ./ relative_encoding_strength["std_v"] 
    θh_ratio = relative_encoding_strength["std_deconv_θh"] ./ relative_encoding_strength["std_θh"]
    P_ratio = relative_encoding_strength["std_deconv_P"] ./ relative_encoding_strength["std_P"]

    v_thresh = max.(threshold_artifact / signal, v_ratio .* threshold_weak)
    θh_thresh = max.(threshold_artifact / signal, θh_ratio .* threshold_weak)
    P_thresh = max.(threshold_artifact / signal, P_ratio .* threshold_weak)
    
    for (i,j) in VALID_V_COMPARISONS
        if i > j
            continue
        end
        for k in 1:s[3]
            for m in 1:s[4]
                # count equal points as 0.5
                diff_1 = deconvolved_activity_1[:,i,k,m] .- deconvolved_activity_1[:,j,k,m]
                diff_2 = deconvolved_activity_2[:,i,k,m] .- deconvolved_activity_2[:,j,k,m]
                categories["v_encoding"][i,j,k,m] = compute_p ? prob_P_greater_Q(diff_1 .+ v_thresh, diff_2) : metric(signal*stat(diff_2 ./ v_ratio) - signal*stat(diff_1 ./ v_ratio))
                categories["v_encoding"][j,i,k,m] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- v_thresh, diff_2) : metric(signal*stat(diff_1 ./ v_ratio) - signal*stat(diff_2 ./ v_ratio))
            end
        end
    end

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,2,1,1]) .- (deconvolved_activity_1[:,3,1,1] .- deconvolved_activity_1[:,4,1,1])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,2,1,1]) .- (deconvolved_activity_2[:,3,1,1] .- deconvolved_activity_2[:,4,1,1])
    categories["v_rect_neg"] = compute_p ? prob_P_greater_Q(diff_1 .+ v_thresh, diff_2) : metric(signal*stat(diff_2 ./ v_ratio) - signal*stat(diff_1 ./ v_ratio))
    categories["v_rect_pos"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- v_thresh, diff_2) : metric(signal*stat(diff_1 ./ v_ratio) - signal*stat(diff_2 ./ v_ratio))

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,2,1,1]) .+ (deconvolved_activity_1[:,3,1,1] .- deconvolved_activity_1[:,4,1,1])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,2,1,1]) .+ (deconvolved_activity_2[:,3,1,1] .- deconvolved_activity_2[:,4,1,1])
    categories["v_fwd"] = compute_p ? prob_P_greater_Q(diff_1 .+ v_thresh, diff_2) : metric(signal*stat(diff_2 ./ v_ratio) - signal*stat(diff_1 ./ v_ratio))
    categories["v_rev"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- v_thresh, diff_2) : metric(signal*stat(diff_1 ./ v_ratio) - signal*stat(diff_2 ./ v_ratio))

    for i = [1,4]
        k = (i == 1) ? "rev_θh_encoding" : "fwd_θh_encoding"
        diff_1 = deconvolved_activity_1[:,i,1,1] .- deconvolved_activity_1[:,i,2,1]
        diff_2 = deconvolved_activity_2[:,i,1,1] .- deconvolved_activity_2[:,i,2,1]
        categories[k*"_act"] = compute_p ? prob_P_greater_Q(diff_1 .+ θh_thresh, diff_2) : metric(signal*stat(diff_2 ./ θh_ratio) - signal*stat(diff_1 ./ θh_ratio))
        categories[k*"_inh"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- θh_thresh, diff_2) : metric(signal*stat(diff_2 ./ θh_ratio) - signal*stat(diff_1 ./ θh_ratio))

        k = (i == 1) ? "rev_P_encoding" : "fwd_P_encoding"
        diff_1 = deconvolved_activity_1[:,i,1,1] .- deconvolved_activity_1[:,i,1,2]
        diff_2 = deconvolved_activity_2[:,i,1,1] .- deconvolved_activity_2[:,i,1,2]
        categories[k*"_act"] = compute_p ? prob_P_greater_Q(diff_1 .+ P_thresh, diff_2) : metric(signal*stat(diff_2 ./ P_ratio) - signal*stat(diff_1 ./ P_ratio))
        categories[k*"_inh"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- P_thresh, diff_2) : metric(signal*stat(diff_2 ./ P_ratio) - signal*stat(diff_1 ./ P_ratio))
    end

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,1,2,1]) .- (deconvolved_activity_1[:,4,1,1] .- deconvolved_activity_1[:,4,2,1])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,1,2,1]) .- (deconvolved_activity_2[:,4,1,1] .- deconvolved_activity_2[:,4,2,1])
    categories["θh_rect_neg"] = compute_p ? prob_P_greater_Q(diff_1 .+ θh_thresh, diff_2) : metric(signal*stat(diff_2 ./ θh_ratio) - signal*stat(diff_1 ./ θh_ratio))
    categories["θh_rect_pos"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- θh_thresh, diff_2) : metric(signal*stat(diff_1 ./ θh_ratio) - signal*stat(diff_2 ./ θh_ratio))

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,1,2,1]) .+ (deconvolved_activity_1[:,4,1,1] .- deconvolved_activity_1[:,4,2,1])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,1,2,1]) .+ (deconvolved_activity_2[:,4,1,1] .- deconvolved_activity_2[:,4,2,1])
    categories["θh_pos"] = compute_p ? prob_P_greater_Q(diff_1 .+ θh_thresh, diff_2) : metric(signal*stat(diff_2 ./ θh_ratio) - signal*stat(diff_1 ./ θh_ratio))
    categories["θh_neg"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- θh_thresh, diff_2) : metric(signal*stat(diff_1 ./ θh_ratio) - signal*stat(diff_2 ./ θh_ratio))
    

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,1,1,2]) .- (deconvolved_activity_1[:,4,1,1] .- deconvolved_activity_1[:,4,1,2])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,1,1,2]) .- (deconvolved_activity_2[:,4,1,1] .- deconvolved_activity_2[:,4,1,2])
    categories["P_rect_neg"] = compute_p ? prob_P_greater_Q(diff_1 .+ P_thresh, diff_2) : metric(signal*stat(diff_2 ./ P_ratio) - signal*stat(diff_1))
    categories["P_rect_pos"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- P_thresh, diff_2) : metric(signal*stat(diff_1 ./ P_ratio) - signal*stat(diff_2))

    diff_1 = (deconvolved_activity_1[:,1,1,1] .- deconvolved_activity_1[:,1,1,2]) .+ (deconvolved_activity_1[:,4,1,1] .- deconvolved_activity_1[:,4,1,2])
    diff_2 = (deconvolved_activity_2[:,1,1,1] .- deconvolved_activity_2[:,1,1,2]) .+ (deconvolved_activity_2[:,4,1,1] .- deconvolved_activity_2[:,4,1,2])
    categories["P_pos"] = compute_p ? prob_P_greater_Q(diff_1 .+ P_thresh, diff_2) : metric(signal*stat(diff_2 ./ P_ratio) - signal*stat(diff_1 ./ P_ratio))
    categories["P_neg"] = compute_p ? 1 - prob_P_greater_Q(diff_1 .- P_thresh, diff_2) : metric(signal*stat(diff_1 ./ P_ratio) - signal*stat(diff_2 ./ P_ratio))

    return categories
end

"""
Categorizes all neurons from their deconvolved activities.

# Arguments:
- `deconvolved_activities_1`: Deconvolved activities of neurons.
- `deconvolved_activities_2`: Either 0 (to check neuron encoding), or deconvolved activities of neurons at an earlier time point (to check encoding change).
- `p`: Significant p-value.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
- `threshold_artifact`: Deconvolved activity must differ by at least this much, to filter out motion artifacts
- `threshold_weak`: Non-deconvolved activity must differ by at least this much, to filter out weak overfitting encodings
- `relative_encoding_strength`: Relative encoding strength of the neurons
- `rel_enc_str_correct`: Threshold for deleting neurons with too low relative encoding strength.
- `compute_feeding` (optional, default `true`): Whether to consider feeding behavior in the analysis.
- `ewma1` (optional): If set, compute EWMA difference between activities and include it in the `all` category. Put the EWMA values for the later timepoint here.
- `ewma2` (optional): If set, compute EWMA difference between activities and include it in the `all` category. Put the EWMA values for the earlier timepoint here.
"""
function categorize_neurons(deconvolved_activities_1, deconvolved_activities_2, p::Real, θh_pos_is_ventral::Bool, trace_original, threshold_artifact::Real, threshold_weak::Real, relative_encoding_strength; rel_enc_str_correct::Float64=0.0, compute_feeding::Bool=true, ewma1=nothing, ewma2=nothing)
    categories = Dict()
    categories["v"] = Dict()
    categories["v"]["rev"] = []
    categories["v"]["fwd"] = []
    categories["v"]["rev_slope_pos"] = []
    categories["v"]["rev_slope_neg"] = []
    categories["v"]["rect_pos"] = []
    categories["v"]["rect_neg"] = []
    categories["v"]["fwd_slope_pos"] = []
    categories["v"]["fwd_slope_neg"] = []
    categories["v"]["all"] = []
    categories["θh"] = Dict()
    categories["θh"]["fwd_ventral"] = []
    categories["θh"]["fwd_dorsal"] = []
    categories["θh"]["rev_ventral"] = []
    categories["θh"]["rev_dorsal"] = []
    categories["θh"]["rect_dorsal"] = []
    categories["θh"]["rect_ventral"] = []
    categories["θh"]["all"] = []
    categories["P"] = Dict()
    categories["P"]["fwd_act"] = []
    categories["P"]["fwd_inh"] = []
    categories["P"]["rev_act"] = []
    categories["P"]["rev_inh"] = []
    categories["P"]["rect_act"] = []
    categories["P"]["rect_inh"] = []
    categories["P"]["all"] = []

    compute_ewma = !isnothing(ewma1) && !isnothing(ewma2)

    if compute_ewma
        categories["ewma_pos"] = []
        categories["ewma_neg"] = []
    end
    categories["all"] = []

    neuron_cats = Dict()
    for neuron = keys(deconvolved_activities_1)
        signal = std(trace_original[neuron,:]) / mean(trace_original[neuron, :])

        neuron_cats[neuron] = neuron_p_vals(deconvolved_activities_1[neuron], deconvolved_activities_2[neuron], signal, threshold_artifact, threshold_weak, relative_encoding_strength[neuron])
    end

    max_n = maximum(keys(neuron_cats))

    corrected_p_vals = Dict()
    corrected_p_vals["v"] = Dict()
    corrected_p_vals["v"]["rev"] = ones(max_n)
    corrected_p_vals["v"]["fwd"] = ones(max_n)
    corrected_p_vals["v"]["rev_slope_pos"] = ones(max_n)
    corrected_p_vals["v"]["rev_slope_neg"] = ones(max_n)
    corrected_p_vals["v"]["rect_pos"] = ones(max_n)
    corrected_p_vals["v"]["rect_neg"] = ones(max_n)
    corrected_p_vals["v"]["fwd_slope_pos"] = ones(max_n)
    corrected_p_vals["v"]["fwd_slope_neg"] = ones(max_n)
    corrected_p_vals["v"]["all"] = ones(max_n)
    corrected_p_vals["θh"] = Dict()
    corrected_p_vals["θh"]["fwd_ventral"] = ones(max_n)
    corrected_p_vals["θh"]["fwd_dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["rev_ventral"] = ones(max_n)
    corrected_p_vals["θh"]["rev_dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["rect_dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["rect_ventral"] = ones(max_n)
    corrected_p_vals["θh"]["dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["ventral"] = ones(max_n)
    corrected_p_vals["θh"]["all"] = ones(max_n)
    corrected_p_vals["P"] = Dict()
    corrected_p_vals["P"]["fwd_act"] = ones(max_n)
    corrected_p_vals["P"]["fwd_inh"] = ones(max_n)
    corrected_p_vals["P"]["rev_act"] = ones(max_n)
    corrected_p_vals["P"]["rev_inh"] = ones(max_n)
    corrected_p_vals["P"]["act"] = ones(max_n)
    corrected_p_vals["P"]["inh"] = ones(max_n)
    corrected_p_vals["P"]["rect_act"] = ones(max_n)
    corrected_p_vals["P"]["rect_inh"] = ones(max_n)
    corrected_p_vals["P"]["all"] = ones(max_n)

    if compute_ewma
        corrected_p_vals["ewma_pos"] = ones(max_n)
        corrected_p_vals["ewma_neg"] = ones(max_n)
    end
    corrected_p_vals["all"] = ones(max_n)

    v_p_vals_uncorr = ones(max_n,4,4)
    v_p_vals = ones(max_n,4,4)
    v_p_vals_rect_neg = ones(max_n)
    v_p_vals_rect_pos = ones(max_n)
    θh_p_vals_rect_neg = ones(max_n)
    θh_p_vals_rect_pos = ones(max_n)
    P_p_vals_rect_neg = ones(max_n)
    P_p_vals_rect_pos = ones(max_n)


    v_p_vals_all_uncorr = ones(max_n)
    θh_p_vals_all_uncorr = ones(max_n)
    P_p_vals_all_uncorr = ones(max_n)
    p_vals_all_uncorr = ones(max_n)
    v_p_vals_all = ones(max_n)
    θh_p_vals_all = ones(max_n)
    P_p_vals_all = ones(max_n)
    p_vals_all = ones(max_n)


    v_p_rel_enc_str = Dict()
    θh_p_rel_enc_str = Dict()
    P_p_rel_enc_str = Dict()

    for neuron in keys(neuron_cats)
        v_p_rel_enc_str[neuron] = 1-prob_P_greater_Q(relative_encoding_strength[neuron]["v"], rel_enc_str_correct)
        θh_p_rel_enc_str[neuron] = 1-prob_P_greater_Q(relative_encoding_strength[neuron]["θh"], rel_enc_str_correct)
        P_p_rel_enc_str[neuron] = 1-prob_P_greater_Q(relative_encoding_strength[neuron]["P"], rel_enc_str_correct)
    end

    # for velocity, take best θh and P values but MH correct
    for (i,j) = VALID_V_COMPARISONS
        for neuron in keys(neuron_cats)
            v_p_vals_uncorr[neuron,i,j] = compute_AND(v_p_rel_enc_str[neuron], neuron_cats[neuron]["v_encoding"][i,j,2,2])
        end
        # use BH correction since these are independent values
        v_p_vals[:,i,j] .= adjust(v_p_vals_uncorr[:,i,j], BenjaminiHochberg())
    end


    for neuron in keys(neuron_cats)


        v_p_vals_rect_neg[neuron] = compute_AND(v_p_rel_enc_str[neuron], neuron_cats[neuron]["v_rect_neg"])
        v_p_vals_rect_pos[neuron] = compute_AND(v_p_rel_enc_str[neuron], neuron_cats[neuron]["v_rect_pos"])

        θh_p_vals_rect_neg[neuron] = compute_AND(θh_p_rel_enc_str[neuron], neuron_cats[neuron]["θh_rect_neg"])
        θh_p_vals_rect_pos[neuron] = compute_AND(θh_p_rel_enc_str[neuron], neuron_cats[neuron]["θh_rect_pos"])

        P_p_vals_rect_neg[neuron] = compute_AND(P_p_rel_enc_str[neuron], neuron_cats[neuron]["P_rect_neg"])
        P_p_vals_rect_pos[neuron] = compute_AND(P_p_rel_enc_str[neuron], neuron_cats[neuron]["P_rect_pos"])

        adjust_v_p_vals = Vector{Float64}()
        adjust_θh_p_vals = Vector{Float64}()
        adjust_P_p_vals = Vector{Float64}()
        all_p_vals = Vector{Float64}()
        for (i,j) = VALID_V_COMPARISONS
            if i > j
                continue
            end
            # correct for anticorrelation between forward and reverse neurons
            push!(adjust_v_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(v_p_vals_uncorr[neuron,i,j], v_p_vals_uncorr[neuron,j,i]))))
            push!(all_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(v_p_vals_uncorr[neuron,i,j], v_p_vals_uncorr[neuron,j,i]))))
        end
        n = neuron

        push!(adjust_θh_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["fwd_θh_encoding_act"], neuron_cats[n]["fwd_θh_encoding_inh"]))))
        push!(adjust_θh_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["rev_θh_encoding_act"], neuron_cats[n]["rev_θh_encoding_inh"]))))
        push!(adjust_θh_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["θh_rect_neg"], neuron_cats[n]["θh_rect_pos"]))))
        push!(adjust_θh_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["θh_pos"], neuron_cats[n]["θh_neg"]))))

        push!(all_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["fwd_θh_encoding_act"], neuron_cats[n]["fwd_θh_encoding_inh"]))))
        push!(all_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["rev_θh_encoding_act"], neuron_cats[n]["rev_θh_encoding_inh"]))))
        push!(all_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["θh_rect_neg"], neuron_cats[n]["θh_rect_pos"]))))
        push!(all_p_vals, compute_AND(θh_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["θh_pos"], neuron_cats[n]["θh_neg"]))))

        if compute_feeding
            push!(adjust_P_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["fwd_P_encoding_act"], neuron_cats[n]["fwd_P_encoding_inh"]))))
            push!(adjust_P_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["rev_P_encoding_act"], neuron_cats[n]["rev_P_encoding_inh"]))))
            push!(adjust_P_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["P_rect_neg"], neuron_cats[n]["P_rect_pos"]))))
            push!(adjust_P_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["P_pos"], neuron_cats[n]["P_neg"]))))

            push!(all_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["fwd_P_encoding_act"], neuron_cats[n]["fwd_P_encoding_inh"]))))
            push!(all_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["rev_P_encoding_act"], neuron_cats[n]["rev_P_encoding_inh"]))))
            push!(all_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["P_rect_neg"], neuron_cats[n]["P_rect_pos"]))))
            push!(all_p_vals, compute_AND(P_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["P_pos"], neuron_cats[n]["P_neg"]))))
        end

        push!(adjust_v_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["v_rect_neg"], neuron_cats[n]["v_rect_pos"]))))
        push!(adjust_v_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["θh_rect_neg"], neuron_cats[n]["θh_rect_pos"]))))
        push!(adjust_v_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["P_rect_neg"], neuron_cats[n]["P_rect_pos"]))))
        push!(adjust_v_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["v_fwd"], neuron_cats[n]["v_rev"]))))

        push!(all_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["v_rect_neg"], neuron_cats[n]["v_rect_pos"]))))
        push!(all_p_vals, compute_AND(v_p_rel_enc_str[neuron], min(1,2*min(neuron_cats[n]["v_fwd"], neuron_cats[n]["v_rev"]))))

        if compute_ewma
            ewma_neg = prob_P_greater_Q(ewma1[n,:], ewma2[n,:])
            ewma_pos = prob_P_greater_Q(ewma2[n,:], ewma1[n,:])
            corrected_p_vals["ewma_pos"][n] = ewma_pos
            corrected_p_vals["ewma_neg"][n] = ewma_neg
            push!(all_p_vals, min(1,2*min(ewma_pos, ewma_neg)))
        end

        # use BH correction since correlations are expected to be positive after above correction
        p_vals_all_uncorr[neuron] = minimum(adjust(all_p_vals, BenjaminiHochberg()))
        v_p_vals_all_uncorr[neuron] = minimum(adjust(adjust_v_p_vals, BenjaminiHochberg()))
        θh_p_vals_all_uncorr[neuron] = minimum(adjust(adjust_θh_p_vals, BenjaminiHochberg()))
        if compute_feeding
            P_p_vals_all_uncorr[neuron] = minimum(adjust(adjust_P_p_vals, BenjaminiHochberg()))
        end
    end

    if compute_ewma
        corrected_p_vals["ewma_pos"] = adjust(corrected_p_vals["ewma_pos"], BenjaminiHochberg())
        corrected_p_vals["ewma_neg"] = adjust(corrected_p_vals["ewma_neg"], BenjaminiHochberg())
        categories["ewma_pos"] = [n for n in 1:max_n if corrected_p_vals["ewma_pos"][n] < p]
        categories["ewma_neg"] = [n for n in 1:max_n if corrected_p_vals["ewma_neg"][n] < p]
    end


    v_p_vals_rect_pos = adjust(v_p_vals_rect_pos, BenjaminiHochberg())
    v_p_vals_rect_neg = adjust(v_p_vals_rect_neg, BenjaminiHochberg())

    P_p_vals_rect_pos = adjust(P_p_vals_rect_pos, BenjaminiHochberg())
    P_p_vals_rect_neg = adjust(P_p_vals_rect_neg, BenjaminiHochberg())

    v_p_vals_all = adjust(v_p_vals_all_uncorr, BenjaminiHochberg())
    θh_p_vals_all = adjust(θh_p_vals_all_uncorr, BenjaminiHochberg())
    P_p_vals_all = adjust(P_p_vals_all_uncorr, BenjaminiHochberg())
    p_vals_all = adjust(p_vals_all_uncorr, BenjaminiHochberg())


    fwd_p_vals = adjust([compute_AND(v_p_rel_enc_str[n], neuron_cats[n]["v_fwd"]) for n in sort(collect(keys(neuron_cats)))], BenjaminiHochberg())
    rev_p_vals = adjust([compute_AND(v_p_rel_enc_str[n], neuron_cats[n]["v_rev"]) for n in sort(collect(keys(neuron_cats)))], BenjaminiHochberg())
    categories["v"]["rev"] = [n for n in 1:max_n if rev_p_vals[n] < p]
    categories["v"]["fwd"] = [n for n in 1:max_n if fwd_p_vals[n] < p]
    categories["v"]["rev_slope_pos"] = [n for n in 1:max_n if v_p_vals[n,1,2] < p]
    categories["v"]["rev_slope_neg"] = [n for n in 1:max_n if v_p_vals[n,2,1] < p]
    categories["v"]["rect_pos"] = [n for n in 1:max_n if v_p_vals_rect_pos[n] < p]
    categories["v"]["rect_neg"] = [n for n in 1:max_n if v_p_vals_rect_neg[n] < p]
    categories["v"]["fwd_slope_pos"] = [n for n in 1:max_n if v_p_vals[n,3,4] < p]
    categories["v"]["fwd_slope_neg"] = [n for n in 1:max_n if v_p_vals[n,4,3] < p]
    categories["v"]["all"] = [n for n in 1:max_n if v_p_vals_all[n] < p]

    corrected_p_vals["v"]["rev"] .= rev_p_vals[:]
    corrected_p_vals["v"]["fwd"] .= fwd_p_vals[:]
    corrected_p_vals["v"]["rev_slope_pos"] .= v_p_vals[:,1,2]
    corrected_p_vals["v"]["rev_slope_neg"] .= v_p_vals[:,2,1]
    corrected_p_vals["v"]["rect_pos"] .= v_p_vals_rect_pos[:]
    corrected_p_vals["v"]["rect_neg"] .= v_p_vals_rect_neg[:]
    corrected_p_vals["v"]["fwd_slope_pos"] .= v_p_vals[:,3,4]
    corrected_p_vals["v"]["fwd_slope_neg"] .= v_p_vals[:,4,3]
    corrected_p_vals["v"]["all"] .= v_p_vals_all[:]

    if !θh_pos_is_ventral
        fwd_θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["fwd_θh_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
        fwd_θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["fwd_θh_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
        rev_θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["rev_θh_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
        rev_θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["rev_θh_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
        θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["θh_pos"]) for n in 1:max_n], BenjaminiHochberg())
        θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["θh_neg"]) for n in 1:max_n], BenjaminiHochberg())
        θh_p_vals_rect_dorsal = adjust(θh_p_vals_rect_pos, BenjaminiHochberg())
        θh_p_vals_rect_ventral = adjust(θh_p_vals_rect_neg, BenjaminiHochberg())
    else
        fwd_θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["fwd_θh_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
        fwd_θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["fwd_θh_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
        rev_θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["rev_θh_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
        rev_θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["rev_θh_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
        θh_dorsal = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["θh_neg"]) for n in 1:max_n], BenjaminiHochberg())
        θh_ventral = adjust([compute_AND(θh_p_rel_enc_str[n], neuron_cats[n]["θh_pos"]) for n in 1:max_n], BenjaminiHochberg())
        θh_p_vals_rect_dorsal = adjust(θh_p_vals_rect_neg, BenjaminiHochberg())
        θh_p_vals_rect_ventral = adjust(θh_p_vals_rect_pos, BenjaminiHochberg())
    end

    categories["θh"]["fwd_ventral"] = [n for n in 1:max_n if fwd_θh_ventral[n] < p]
    categories["θh"]["fwd_dorsal"] = [n for n in 1:max_n if fwd_θh_dorsal[n] < p]
    categories["θh"]["rev_ventral"] = [n for n in 1:max_n if rev_θh_ventral[n] < p]
    categories["θh"]["rev_dorsal"] = [n for n in 1:max_n if rev_θh_dorsal[n] < p]
    categories["θh"]["rect_ventral"] = [n for n in 1:max_n if θh_p_vals_rect_ventral[n] < p]
    categories["θh"]["rect_dorsal"] = [n for n in 1:max_n if θh_p_vals_rect_dorsal[n] < p]
    categories["θh"]["dorsal"] = [n for n in 1:max_n if θh_dorsal[n] < p]
    categories["θh"]["ventral"] = [n for n in 1:max_n if θh_ventral[n] < p]
    categories["θh"]["all"] = [n for n in 1:max_n if θh_p_vals_all[n] < p]

    corrected_p_vals["θh"]["fwd_ventral"] .= fwd_θh_ventral
    corrected_p_vals["θh"]["fwd_dorsal"] .= fwd_θh_dorsal
    corrected_p_vals["θh"]["rev_ventral"] .= rev_θh_ventral
    corrected_p_vals["θh"]["rev_dorsal"] .= rev_θh_dorsal
    corrected_p_vals["θh"]["rect_ventral"] .= θh_p_vals_rect_ventral
    corrected_p_vals["θh"]["rect_dorsal"] .= θh_p_vals_rect_dorsal
    corrected_p_vals["θh"]["dorsal"] .= θh_dorsal
    corrected_p_vals["θh"]["ventral"] .= θh_ventral
    corrected_p_vals["θh"]["all"] .= θh_p_vals_all


    fwd_P_act = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["fwd_P_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
    fwd_P_inh = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["fwd_P_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
    rev_P_act = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["rev_P_encoding_act"]) for n in 1:max_n], BenjaminiHochberg())
    rev_P_inh = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["rev_P_encoding_inh"]) for n in 1:max_n], BenjaminiHochberg())
    P_act = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["P_pos"]) for n in 1:max_n], BenjaminiHochberg())
    P_inh = adjust([compute_AND(P_p_rel_enc_str[n], neuron_cats[n]["P_neg"]) for n in 1:max_n], BenjaminiHochberg())

    categories["P"]["fwd_inh"] = [n for n in 1:max_n if fwd_P_inh[n] < p]
    categories["P"]["fwd_act"] = [n for n in 1:max_n if fwd_P_act[n] < p]
    categories["P"]["rev_inh"] = [n for n in 1:max_n if rev_P_inh[n] < p]
    categories["P"]["rev_act"] = [n for n in 1:max_n if rev_P_act[n] < p]
    categories["P"]["rect_act"] = [n for n in 1:max_n if P_p_vals_rect_pos[n] < p]
    categories["P"]["rect_inh"] = [n for n in 1:max_n if P_p_vals_rect_neg[n] < p]
    categories["P"]["act"] = [n for n in 1:max_n if P_act[n] < p]
    categories["P"]["inh"] = [n for n in 1:max_n if P_inh[n] < p]
    categories["P"]["all"] = [n for n in 1:max_n if P_p_vals_all[n] < p]

    categories["all"] = [n for n in 1:max_n if p_vals_all[n] < p]

    corrected_p_vals["P"]["fwd_inh"] .= fwd_P_inh
    corrected_p_vals["P"]["fwd_act"] .= fwd_P_act
    corrected_p_vals["P"]["rev_inh"] .= rev_P_inh
    corrected_p_vals["P"]["rev_act"] .= rev_P_act
    corrected_p_vals["P"]["rect_act"] .=  P_p_vals_rect_pos
    corrected_p_vals["P"]["rect_inh"] .= P_p_vals_rect_neg
    corrected_p_vals["P"]["act"] .= P_act
    corrected_p_vals["P"]["inh"] .= P_inh
    corrected_p_vals["P"]["all"] .= P_p_vals_all

    corrected_p_vals["all"] .= p_vals_all

    return categories, corrected_p_vals, neuron_cats
end

"""
Categorizes all neurons in all datasets at all time ranges. Returns the neuron categorization, the p-values for it,
and the raw (not multiple-hypothesis corrected) p-values.

# Arguments:
- `fit_results`: Gen fit results.
- `deconvolved_activity`: Dictionary of deconvolved activity values at lattice points.
- `P_ranges`: Ranges of feeding values
- `p`: Significant `p`-value.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
- `threshold_artifact`: Deconvolved activity must differ by at least this much, to filter out motion artifacts
- `threshold_weak`: Non-deconvolved activity must differ by at least this much, to filter out weak overfitting encodings
- `P_diff_thresh` (optional, default `0`): Minimum feeding variance in a time range necessary for trying to compute feeding info.
- `rel_enc_str_correct` (optional, default `0`): Threshold for deleting neurons with too low relative encoding strength.
"""
function categorize_all_neurons(fit_results, deconvolved_activity, P_ranges, p, θh_pos_is_ventral, threshold_artifact, threshold_weak, relative_encoding_strength; P_diff_thresh=0.0, rel_enc_str_correct=0.0)
    neuron_categorization = Dict()
    neuron_p = Dict()
    neuron_cats = Dict()
    @showprogress for dataset = keys(deconvolved_activity)
        neuron_categorization[dataset] = Dict()
        neuron_p[dataset] = Dict()
        neuron_cats[dataset] = Dict()
        for rng = 1:length(fit_results[dataset]["ranges"])
            empty_cat = Dict()
            for n = 1:fit_results[dataset]["num_neurons"]
                empty_cat[n] = zeros(size(deconvolved_activity[dataset][rng][n]))
            end
            compute_feeding = (P_ranges[dataset][rng][2] - P_ranges[dataset][rng][1]) > P_diff_thresh
            rel_enc_str = (isnothing(relative_encoding_strength) ? nothing : relative_encoding_strength[dataset][rng])
            neuron_categorization[dataset][rng], neuron_p[dataset][rng], neuron_cats[dataset][rng] = categorize_neurons(deconvolved_activity[dataset][rng], empty_cat, p, θh_pos_is_ventral[dataset], fit_results[dataset]["trace_original"], 
                    threshold_artifact, threshold_weak, relative_encoding_strength[dataset][rng], compute_feeding=compute_feeding, rel_enc_str_correct=rel_enc_str_correct)
        end
    end
    return neuron_categorization, neuron_p, neuron_cats
end

function subcategorize_all_neurons!(fit_results, analysis_dict, datasets)
    v_keys = ["fwd_slope_pos", "fwd_slope_neg", "rev_slope_pos", "rev_slope_neg", "rect_pos", "rect_neg"]
    θh_keys = ["fwd_ventral", "fwd_dorsal", "rev_ventral", "rev_dorsal", "rect_ventral", "rect_dorsal"]
    P_keys = ["fwd_act", "fwd_inh", "rev_act", "rev_inh", "rect_act", "rect_inh"]

    analysis_dict["v_enc"] = Dict()
    for k in v_keys
        analysis_dict["v_enc"][k] = []
    end
    analysis_dict["θh_enc"] = Dict()
    for k in θh_keys
        analysis_dict["θh_enc"][k] = []
    end
    analysis_dict["P_enc"] = Dict()
    for k in P_keys
        analysis_dict["P_enc"][k] = []
    end
    num_possible_encodings = length(v_keys) + length(θh_keys) + length(P_keys)
    analysis_dict["joint_encoding"] = zeros(num_possible_encodings, num_possible_encodings)
    tot = 0
    analysis_dict["v_enc_matrix"] = zeros(3,3)
    analysis_dict["θh_enc_matrix"] = zeros(3,3)
    analysis_dict["P_enc_matrix"] = zeros(3,3)
    analysis_dict["neuron_subcategorization"] = Dict()
    subcategories = ["analog_pos", "analog_neg", "fwd_slope_pos_rect_pos", "rev_slope_pos_rect_neg", "fwd_slope_neg_rect_neg", "rev_slope_neg_rect_pos", "fwd_pos_rev_neg", "rev_pos_fwd_neg", "unknown_enc", "nonencoding"]
    analysis_dict["joint_subencoding"] = zeros(3*length(subcategories), 3*length(subcategories))

    # rect fwd inh, rect fwd, unknown
    # slow, linear fwd, rect rev inh
    # linear rev, speed, rect rev
    for dataset in datasets
        analysis_dict["neuron_subcategorization"][dataset] = Dict()
        for rng in 1:length(fit_results[dataset]["ranges"])
            analysis_dict["neuron_subcategorization"][dataset][rng] = Dict()
            for beh in ["v", "θh", "P"]
                analysis_dict["neuron_subcategorization"][dataset][rng][beh] = Dict()
                for cat in subcategories
                    analysis_dict["neuron_subcategorization"][dataset][rng][beh][cat] = []
                end
            end
                
            count = 0
            for neuron in 1:fit_results[dataset]["num_neurons"]
                encs_all = zeros(Bool, num_possible_encodings)
                idx=1
                
                for (beh, beh_enc, beh_keys, beh_enc_matrix) in [("v", "v_enc", v_keys, "v_enc_matrix"), 
                            ("θh", "θh_enc", θh_keys, "θh_enc_matrix"), ("P", "P_enc", P_keys, "P_enc_matrix")]

                    encs = zeros(Bool,length(beh_keys))
                    for (i,k) in enumerate(beh_keys)
                        if neuron in analysis_dict["neuron_categorization"][dataset][rng][beh][k]
                            push!(analysis_dict[beh_enc][k], (dataset, neuron))
                            encs[i] = 1
                        end
                    end
                    
                    @assert(!(encs[1] && encs[2]))
                    @assert(!(encs[3] && encs[4]))
                    @assert(!(encs[5] && encs[6]))
                    if encs[1] && encs[4] # speed neuron
                        @assert(encs[5]) # speed neurons must be rectified
                        analysis_dict[beh_enc_matrix][2,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_pos_rev_neg"], neuron)
                    elseif encs[2] && encs[3] # slow neuron
                        @assert(encs[6]) # slow neurons must be rectified
                        analysis_dict[beh_enc_matrix][1,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_pos_fwd_neg"], neuron)
                    elseif encs[1] && encs[5] # forward positively-rectified
                        analysis_dict[beh_enc_matrix][3,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_pos_rect_pos"], neuron)
                    elseif encs[4] && encs[5] # reversal positively-rectified
                        analysis_dict[beh_enc_matrix][2,3] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_neg_rect_pos"], neuron)
                    elseif encs[2] && encs[6] # reversal negatively-rectified
                        analysis_dict[beh_enc_matrix][3,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_neg_rect_neg"], neuron)
                    elseif encs[3] && encs[6] # forward negatively-rectified
                        analysis_dict[beh_enc_matrix][1,3] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_pos_rect_neg"], neuron)
                    elseif encs[2] && encs[4] # linear reversal
                        analysis_dict[beh_enc_matrix][2,2] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_neg"], neuron)
                    elseif encs[1] && encs[3] # linear forward
                        analysis_dict[beh_enc_matrix][1,1] += 1
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_pos"], neuron)
                    elseif any(encs[1:6])
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["unknown_enc"], neuron)
                        analysis_dict[beh_enc_matrix][3,3] += 1
                        @assert(sum(encs[1:6]) == 1)
                    else
                        @assert(sum(encs[1:6]) == 0)
                        push!(analysis_dict["neuron_subcategorization"][dataset][rng][beh]["nonencoding"], neuron)
                    end
                    
                    encs_all[idx:idx+length(beh_keys)-1] .= encs
                    idx += length(beh_keys)
                end
                
                for i=1:length(encs_all)
                    for j=i+1:length(encs_all)
                        if encs_all[i] && encs_all[j]
                            analysis_dict["joint_encoding"][i,j] += 1
                            analysis_dict["joint_encoding"][j,i] += 1
                        end
                    end
                    if encs_all[i]
                        analysis_dict["joint_encoding"][i,i] += 1
                    end
                end
                
                for (i, beh1) in enumerate(["v", "θh", "P"])
                    for (j, cat1) in enumerate(subcategories)
                        for (x, beh2) in enumerate(["v", "θh", "P"])
                            for (y, cat2) in enumerate(subcategories)
                                if neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh1][cat1] && neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh2][cat2] 
                                    analysis_dict["joint_subencoding"][length(subcategories)*(i-1)+j,length(subcategories)*(x-1)+y] += 1
                                end
                            end
                        end
                    end
                end
                        
                
                tot += 1
            end
        end
    end
end

function get_neuron_category(dataset, rng, neuron, fit_results, neuron_categorization, relative_encoding_strength)
    encoding = []
    relative_enc_str = []
    for beh in ["v", "θh", "P"]
        for k in keys(neuron_categorization[dataset][rng][beh])
            if neuron in neuron_categorization[dataset][rng][beh][k]
                push!(encoding, (beh, k))
            end
        end
        push!(relative_enc_str, median(relative_encoding_strength[dataset][rng][neuron][beh]))
    end

    s = compute_s(median(fit_results[dataset]["sampled_trace_params"][rng,neuron,:,7]))
    τ = log.(s ./ (s .+ 1), 0.5) .* fit_results[dataset]["avg_timestep"]
    return encoding, relative_enc_str, τ
end

function get_enc_stats(fit_results, neuron_p, P_ranges; encoding_changes=nothing, P_diff_thresh=0.5, p=0.05, rngs_valid=nothing)
    result = Dict{String,Dict}()
    list_uid_invalid = String[] # no pumping
    for dataset in keys(fit_results)
        if rngs_valid == nothing
            rngs = 1:length(fit_results[dataset]["ranges"])
        else
            rngs = rngs_valid[dataset]
        end

        dict_ = Dict{String,Any}()
        n_neuron = fit_results[dataset]["num_neurons"]
        n_b = 3 # number of behaviors
        enc_array = zeros(Int, n_neuron, n_b, length(rngs))

        n_neurons_tot_all = 0
        n_neurons_fit_all = 0
        n_neurons_beh = [0,0,0]
        n_neurons_npred = [0,0,0,0]

        P_ranges_valid = [r for r=rngs if P_ranges[dataset][r][2] - P_ranges[dataset][r][1] > P_diff_thresh]
        n_neurons_tot_all += n_neuron
        neurons_fit = [n for n in 1:fit_results[dataset]["num_neurons"] if sum(adjust([neuron_p[dataset][i]["all"][n] for i=rngs], BenjaminiHochberg()) .< p) > 0]
        n_neurons_fit_all += length(neurons_fit)
        if length(P_ranges_valid) == 0
            @warn("Dataset $(dataset) has no time ranges with valid pumping information")
            push!(list_uid_invalid, dataset)
            continue
        end
        for n=1:fit_results[dataset]["num_neurons"]
            if !isnothing(encoding_changes) && (!(n in keys(encoding_changes)) || !(n in encoding_changes[dataset][(1,2)]["neurons"]))
                continue
            end
            max_npred = 0

            v_p = adjust([neuron_p[dataset][r]["v"]["all"][n] for r=rngs], BenjaminiHochberg())
            θh_p = adjust([neuron_p[dataset][r]["θh"]["all"][n] for r=rngs], BenjaminiHochberg())
            P_p_valid = adjust([neuron_p[dataset][r]["P"]["all"][n] for r=P_ranges_valid], BenjaminiHochberg())
            P_p = []
            idx=1
            for r=rngs
                if r in P_ranges_valid
                    push!(P_p, P_p_valid[idx])
                    idx += 1
                else
                    push!(P_p, 1.0)
                end
            end
            if any(v_p .< p)
                n_neurons_beh[1] += 1
            end
            if any(θh_p .< p)
                n_neurons_beh[2] += 1
            end
            if any(P_p .< p)
                n_neurons_beh[3] += 1   
            end

            for r=1:length(rngs)
                enc = adjust([v_p[r], θh_p[r], P_p[r]], BenjaminiHochberg())
                max_npred = max(max_npred, sum(enc .< p))
            end
            n_neurons_npred[max_npred+1] += 1

            enc_array[n,1,:] .= v_p .< p
            enc_array[n,2,:] .= θh_p .< p
            enc_array[n,3,:] .= P_p .< p
        end

        dict_["n_neurons_beh"] = n_neurons_beh
        dict_["n_neurons_npred"] = n_neurons_npred
        dict_["n_neurons_fit_all"] = n_neurons_fit_all
        dict_["n_neurons_tot_all"] = n_neurons_tot_all
        dict_["enc_array"] = enc_array

        result[dataset] = dict_
    end

    result, list_uid_invalid
end


function get_enc_stats_pool(fit_results, neuron_p, P_ranges; P_diff_thresh=0.5, p=0.05, rngs_valid=nothing)
    n_neurons_tot_all = 0
    n_neurons_fit_all = 0
    n_neurons_beh = [0,0,0]
    n_neurons_npred = [0,0,0,0]
   
    dict_enc_stat, list_uid_invalid = get_enc_stats(fit_results, neuron_p,
        P_ranges; P_diff_thresh=P_diff_thresh, p=p, rngs_valid=rngs_valid)
    
    for (k,v) = dict_enc_stat
        if !(k in list_uid_invalid)
            dict_ = dict_enc_stat[k]
            n_neurons_beh .+= dict_["n_neurons_beh"]
            n_neurons_npred .+= dict_["n_neurons_npred"]
            n_neurons_fit_all += dict_["n_neurons_fit_all"]
            n_neurons_tot_all += dict_["n_neurons_tot_all"]
        end
    end
    
    n_neurons_beh, n_neurons_npred, n_neurons_fit_all, n_neurons_tot_all
end

        
function get_consistent_neurons(datasets, fit_results, neuron_categorization, dict_enc, rngs_use; rngs_valid=[5,6], ewma_skip=200, err_thresh=0.9)
    consistent_neurons = Dict()
    inconsistent_neurons = Dict()
    parameters = Dict()
    fits = Dict()
    @assert(length(rngs_valid) == 2)
    for dataset in datasets
        consistent_neurons[dataset] = []
        inconsistent_neurons[dataset] = []
        parameters[dataset] = Dict()
        fits[dataset] = Dict()
        rng = rngs_use[dataset]
        @assert(rng in rngs_valid)
        rng_other = [r for r in rngs_valid if r != rng][1]
        
        for n in dict_enc[dataset]

            ps_keep = []
            if (n in neuron_categorization[dataset][rng]["v"]["all"])
                append!(ps_keep, [1,2,5])
            end
            if (n in neuron_categorization[dataset][rng]["θh"]["all"])
                append!(ps_keep, 3)
            end
            if (n in neuron_categorization[dataset][rng]["P"]["all"])
                append!(ps_keep, 4)
            end
            ps_delete = [p for p in 1:5 if !(p in ps_keep)]
            ps = project_posterior(fit_results[dataset]["sampled_trace_params"][rng, n, :, :], ps_delete)
            ps[6] = 0
            parameters[dataset][n] = ps
            
            rng_test = fit_results[dataset]["ranges"][rng_other]
            rng_len = length(fit_results[dataset]["v"])
            
            rng_fit = fit_results[dataset]["ranges"][rng]
            fit = zscore(model_nl8(rng_len, ps[1:8]..., fit_results[dataset]["v"], fit_results[dataset]["θh"], fit_results[dataset]["P"]))
            
            fits[dataset][n] = fit
            err = mean((fit[rng_test] .- fit_results[dataset]["trace_array"][n,rng_test])[ewma_skip+1:end] .^ 2)
            err_control = mean(fit_results[dataset]["trace_array"][n,rng_test][ewma_skip+1:end].^2)
            
            if err > err_thresh * err_control || length(ps_keep) == 0
                push!(inconsistent_neurons[dataset], n)
            else
                push!(consistent_neurons[dataset], n)
            end
        end
    end
    return consistent_neurons, inconsistent_neurons, parameters, fits
end

function add_weighted_subencoding_matrix!(fit_results, analysis_dict, datasets; use_relative=true)
    analysis_dict["weighted_v_enc_matrix_baseline"] = zeros(3,3)
    analysis_dict["weighted_θh_enc_matrix_baseline"] = zeros(3,3)
    analysis_dict["weighted_P_enc_matrix_baseline"] = zeros(3,3)
    for dataset in datasets
        for rng=1:length(fit_results[dataset]["ranges"])
            for beh = ["v", "θh", "P"]
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_pos"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][1,1] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["analog_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][2,2] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_pos_rev_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][2,1] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_pos_fwd_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][1,2] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_pos_fwd_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][1,2] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_pos_rect_pos"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][3,1] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["fwd_slope_neg_rect_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][3,2] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_pos_rect_neg"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][1,3] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
                for neuron in analysis_dict["neuron_subcategorization"][dataset][rng][beh]["rev_slope_neg_rect_pos"]
                    analysis_dict["weighted_$(beh)_enc_matrix_baseline"][2,3] += use_relative ? median(analysis_dict["relative_encoding_strength"][dataset][rng][neuron][beh]) : 1
                end
            end
        end
    end
end
