module CePNEM

using Gen
using FlavellBase
using HDF5
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using MultivariateStats
using FlavellConstants
using ANTSUNData

include("fit.jl")
include("model.jl")
include("sbc_tests.jl")

export
    # model.jl
    s_MEAN,
    σ_MEAN,
    ℓ_MEAN,
    ℓ_STD,
    α_MEAN,
    α_STD,
    σ_RQ_MEAN,
    σ_RQ_STD,
    σ_SE_MEAN,
    σ_SE_STD,
    σ_NOISE_MEAN,
    σ_NOISE_STD,
    compute_cov_matrix_vectorized_RQ,
    compute_cov_matrix_vectorized_SE,
    unfold_nl7b,
    model_nl8,
    nl8,
    nl9,
    nl10,
    nl10c,
    nl10d,
    unfold_v_noewma,
    unfold_v,
    get_free_params,
    compute_s,
    compute_σ,

    # fit.jl
    jump_c_vT,
    jump_c,
    jump_c_vvT,
    drift_params,
    drift_α,
    drift_ℓ,
    drift_σ_RQ,
    drift_σ_SE,
    drift_σ_noise,
    hmc_jump_update,
    particle_filter_incremental,
    output_state,
    mcmc,
    run_mcmc_10,
    nl10c_traces_to_params,
    nl10d_traces_to_params,

    # sbc_tests.jl
    rank_test,
    χ2_uniformtest
end
