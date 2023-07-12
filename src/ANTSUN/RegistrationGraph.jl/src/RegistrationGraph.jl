module RegistrationGraph

using NRRDIO, Images, Statistics, LinearAlgebra, Graphs, FlavellBase,
    SegmentationTools, SLURMManager, SimpleWeightedGraphs, ProgressMeter,
    Interact, Plots, Dates, Printf, ImageDataIO, PyPlot

include("make_elastix_difficulty.jl")
include("assess_registration_quality.jl")
include("run_elastix.jl")
include("make_registration_graph.jl")
include("registration_visualization.jl")
include("metrics.jl")

export
    generate_elastix_difficulty,
    make_quality_dict,
    write_sbatch_graph,
    run_registration,
    load_graph,
    to_dict,
    remove_frame,
    optimize_subgraph,
    output_graph,
    update_graph,
    remove_previous_registrations,
    make_voting_subgraph,
    sync_registered_data,
    fix_param_paths,
    average_am_registrations,
    plot_centroid_match,
    view_roi_regmap,
    gen_regmap_rgb,
    visualize_roi_predictions,
    make_rgb_arr,
    make_diff_pngs,
    make_diff_pngs_base,
    nrrd_to_png,
    calculate_ncc,
    metric_tfm,
    run_elastix_openmind,
    squeue_submit_sbatch_remote,
    get_squeue_status,
    wait_for_elastix,
    get_lock,
    release_lock,
    average_registered_images
end # module
