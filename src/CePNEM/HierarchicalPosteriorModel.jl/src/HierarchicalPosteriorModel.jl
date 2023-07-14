module HierarchicalPosteriorModel

using Distributions, Optim, Statistics, StatsBase, LinearAlgebra, ForwardDiff,
    CePNEM, SpecialFunctions

include("model.jl")
include("util.jl")
include("variability.jl")
include("fit.jl")

export
    # model.jl
    HBParams,
    angular_log_probability,
    joint_logprob_flat,
    joint_logprob,
    joint_logprob_flat_negated,
    # util.jl
    cart2spher,
    spher2cart,
    angle_diff,
    fit_multivariate_normals,
    bic_multivariate_normals,
    get_Ps,
    get_corrected_r,
    exp_r,
    params_to_spher,
    get_datasets,
    convert_hbparams_to_ps,
    convert_hbdatasets_to_behs,
    angle_mean,
    angle_std,
    hbparams_to_cart,
    mean_direction,
    estimate_kappa,
    fit_vmf,
    compute_cartesian_average,
    # variability.jl
    get_variability,
    get_variability_subtypes,
    # fit.jl
    initialize_params,
    optimize_MAP
end # module
