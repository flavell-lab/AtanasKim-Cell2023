module AnalysisBase

using InformationMeasures, GLMNet, StatsBase, Statistics

include("decoder.jl")
include("mutual_info.jl")
include("cv.jl")
include("cost.jl")

export
    # decoder.jl
    train_decoder,

# mutual_info.jl
    compute_mutual_information_dict,

# metric.jl
    cost_rss,
    cost_abs,
    cost_mse,
    cost_cor,
    reg_L1,
    reg_L2,
    reg_var_L1,
    reg_var_L2,
    reg_L1_nl7,

    # cv.jl
    add_sampling,
    generate_cv_splits,
    rm_dataset_begin,
    rm_low_variation,
    kfold_cv_split,
    trim_idx_splits,
    continuous_cv_split

end # module
