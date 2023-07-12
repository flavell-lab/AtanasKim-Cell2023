module CaAnalysis

using Statistics, PyPlot, HDF5, Dierckx, ProgressMeter, MultivariateStats,
    ImageDataIO, ExtractRegisteredData, Interpolations, FFTW, NRRDIO

import Images:centered, imfilter

include("init.jl")
include("data.jl")
include("heatmap.jl")
include("single_unit.jl")
include("multivar.jl")
include("noise_correction.jl")
include("util/util.jl")
include("util/bleach.jl")
include("util/processing.jl")
include("util/plot.jl")
include("util/denoise.jl")
include("util/baseline.jl")

export import_data,
    # bleach.jl
    fit_bleach,
    fit_bleach!,

    # heatmap.jl
    plot_cluster_cost,
    plot_heatmap,
    order_by_kmeans,
    order_by_cor,

    # unit.jl
    plot_unit_cor,
    get_idx_unit,
    get_idx_stim,
    get_idx_t,
    get_data,
    get_stim,

    # single_unit.jl
    compute_unit_cor,
    plot_unit_cor,

    # multivar.jl
    pca,
    multivar_fit,
    plot_pca_var,
    plot_statespace_component,
    plot_statespace_3d,
    plot_statespace_2d,
    plot_statespace_2d_stim,
    plot_loading,
    highlight_loading,

    # util/processing.jl
    derivative,
    integrate,
    standardize,
    discrete_diff,
    chain_process,

    # util/plot.jl
    highlight_stim,


    # util/denoise.jl
    preview_denoise,
    denoise,
    denoise!,
    DenoiserTrendfilter,
    DenoiserGSTV,

    # util/baseline.jl
    estimate_baseline,
    
    # noise_correction.jl
    divide_by_marker_signal,
    get_background,
    bkg_subtract,
    normalize_traces,
    interpolate_traces,
    zscore_traces,
    process_traces,
    get_all_values,
    get_laser_intensity

end # module
