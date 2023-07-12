Gen.@load_generated_functions

n_params = 11
μ_vT = 0.0
σ_vT = vT_STD

fit_uid = "2021-05-26-07"
output_path_gt = "/om2/user/aaatanas/gen_sbc_nl10c/h5/$(ARGS[1])_gt.h5"
# output_path_smc5000_1 = "/om2/user/aaatanas/gen_sbc_nl9/h5/$(ARGS[1])_smc5000_1.h5"
output_path = "/om2/user/aaatanas/gen_sbc_nl10c/h5/$(ARGS[1])_mcmc2000.h5"
# output_path_mcmc = "/om2/user/aaatanas/gen_sbc_nl9/h5/$(ARGS[1])_mcmc.h5"
# output_path_mcmc_init = "/om2/user/aaatanas/gen_sbc_nl9/h5/$(ARGS[1])_mcmc_init.h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-data.h5"
dict = import_data(path_h5)

n_obs = 400
max_t = n_obs
model = :nl10c
v = dict["velocity"][201:200+n_obs]
θh = dict["θh"][201:200+n_obs]
P = dict["pumping"][201:200+n_obs]

(trace, _) = Gen.generate(nl10c, (n_obs, v, θh, P))

h5open(output_path_gt, "w") do f
    f["ground_truth"] = get_free_params(trace, model)
end

n_init = 100000
n_track = 5
n_iters = 2000

traces_fit = Array{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_track, n_iters+1)
accept = zeros(Bool,n_track,n_iters+1,7)
δ_vals = fill(1.0,n_track,n_iters+1,7)

traces_init = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_init)
scores_init = zeros(n_init)
cmap = Gen.choicemap()
cmap[:ys] = trace[:ys]
    
@time for i=1:n_init
    traces_init[i], _ = generate(nl10c, (max_t,v,θh,P), cmap)
    scores_init[i] = get_score(traces_init[i])
end
    
traces_sorted = traces_init[sort(1:n_init, by=x->scores_init[x])]
for i=1:n_track
    traces_fit[i,1] = traces_sorted[n_init-n_track+i]
end

@time for i=1:n_iters
    for j=1:n_track
        traces_fit[j,i+1], accept[j,i+1,1] = mh(traces_fit[j,i], drift_ℓ, (max_t, δ_vals[j,i,1]))
        traces_fit[j,i+1], accept[j,i+1,2] = mh(traces_fit[j,i+1], drift_σ_SE, (max_t, δ_vals[j,i,2]))
        traces_fit[j,i+1], accept[j,i+1,3] = mh(traces_fit[j,i+1], drift_σ_noise, (max_t, δ_vals[j,i,3]))
        traces_fit[j,i+1], accept[j,i+1,4] = hmc(traces_fit[j,i+1], select(:c_vT, :c_v, :c_θh, :c_P, :c, :b, :y0, :s0), eps=δ_vals[j,i,4])
        for k=1:4
            δ_vals[j,i+1,k] = (accept[j,i+1,k]) ? δ_vals[j,i,k] * 1.1 : δ_vals[j,i,k] / 1.1
        end  
        neg = rand([-1,1])
        (traces_fit[j,i+1], accept[j,i+1,5]) = mh(traces_fit[j,i+1], jump_c_vT, (neg,))
        neg = rand([-1,1])
        (traces_fit[j,i+1], accept[j,i+1,6]) = mh(traces_fit[j,i+1], jump_c, (neg,))
        neg_c_vT = rand([-1,1])
        neg_c_v = rand([-1,1])
        (traces_fit[j,i+1], accept[j,i+1,7]) = mh(traces_fit[j,i+1], jump_c_vvT, (neg_c_vT, neg_c_v, μ_vT, σ_vT))
    end
end

trace_params_fit = zeros(n_track, n_iters+1, n_params)
for i=1:n_iters+1
    for j=1:n_track
        trace_params_fit[j,i,:] .= get_free_params(traces_fit[j,i], :nl10c)
    end
end

h5open(output_path, "w") do f
    f["traces_params_fit"] = trace_params_fit
    f["accept"] = accept
    f["delta_vals"] = δ_vals
end
