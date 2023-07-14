module CePNEMAnalysis

using FlavellBase
using DataFrames
using EncoderModel
using CePNEM
using HDF5
using Gen
using GLM
using GLMNet
using JSON
using Statistics
using StatsBase
using ProgressMeter
using PyPlot
using Plots
using Plots.PlotMeasures
using TSne
using UMAP
using ColorSchemes
using MultipleTesting
using MultivariateStats
using AnalysisBase
using FlavellConstants
using ANTSUNData
using Clustering
using DataStructures
using HypothesisTests

include("bgi.jl")
include("data.jl")
include("dist.jl")
include("encoding-categorization.jl")
include("encoding-subcategorization.jl")
include("encoding-change.jl")
include("mse.jl")
include("decode.jl")
include("state.jl")
include("plot.jl")
include("umap.jl")
include("cluster.jl")
include("pca.jl")
include("strength.jl")
include("tuning.jl")
include("util.jl")

export
    # bgi.jl
    get_CePNEM_fit_score,
    get_CePNEM_prior_score,
    get_CePNEM_full_posterior_score,
    compute_BGI,
    compute_cv_accuracy_priorcompare,
    # data.jl
    load_CePNEM_output,
    get_pumping_ranges,
    add_to_analysis_dict!,
    compute_signal,
    neuropal_data_to_dict,
    export_to_json,
    # dist.jl
    update_cmap!,
    evaluate_pdf_xgiveny!,
    compute_dists,
    kl_div,
    overlap_index,
    prob_P_greater_Q,
    project_posterior,
    # encoding-categorization.jl
    VALID_V_COMPARISONS,
    deconvolved_model_nl8,
    compute_range,
    get_deconvolved_activity,
    make_deconvolved_lattice,
    neuron_p_vals,
    categorize_neurons,
    categorize_all_neurons,
    get_neuron_category,
    get_enc_stats,
    get_enc_stats_pool,
    # encoding-subcategorization.jl
    subcategorize_all_neurons!,
    sum_subencodings,
    normalize_subencodings,
    # encoding-change.jl
    detect_encoding_changes,
    correct_encoding_changes,
    get_enc_change_stats,
    encoding_summary_stats,
    get_enc_change_cat_p_vals,
    get_enc_change_cat_p_vals_dataset,
    get_enc_change_category,
    MSE_correct_encoding_changes,
    compute_feeding_encoding_changes,
    # mse.jl
    idx_splitify_rng,
    generate_reg_L2_nl10d,
    fit_model,
    fit_mse_models,
    compute_CePNEM_MSE,
    # umap.jl
    make_distance_matrix,
    find_subset_idx,
    compute_tsne,
    extrapolate_behaviors,
    compute_extrapolated_CePNEM_posterior_stats,
    append_median_CePNEM_fits,
    project_CePNEM_to_UMAP,
    make_umap_rgb,
    compute_umap_subcategories!,
    # plot.jl
    CET_L13,
    make_deconvolved_heatmap,
    plot_deconvolved_heatmap,
    plot_deconvolved_neural_activity!,
    plot_tsne,
    plot_tau_histogram,
    plot_neuron,
    plot_posterior_heatmap!,
    plot_posterior_rgb,
    plot_arrow!,
    color_to_rgba,
    plot_colorbar,
    get_color_from_palette,
    # pca.jl
    extrapolate_neurons,
    make_distance_matrix,
    invert_id,
    invert_array,
    # cluster.jl
    cluster_dist,
    dendrocolor!,
    # strength.jl
    get_relative_encoding_strength!,
    get_relative_encoding_strength_mt,
    get_relative_encoding_strength,
    # tuning.jl
    get_forwardness,
    get_dorsalness,
    get_feedingness,
    get_tuning_strength,
    calculate_tuning_strength_per_neuron,
    # decode.jl
    fit_decoder,
    compute_variance_explained,
    average_dict_qualities,
    # util.jl
    gaussian_kernel,
    convolve,
    fit_state_classifier,
    get_all_neurons_with_feature,
    correct_name,
    find_peaks,
    parse_tuning_strength,
    get_random_sample_without_feature,
    get_frac_responding,
    get_subcats,
    # state.jl
    fit_state_classifier,
    compute_state_neuron_candidates
end # module
