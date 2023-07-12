module EncoderModel

using FlavellBase, GalacticOptim, NLopt, ForwardDiff, StatsBase, Statistics
include("model.jl")
include("fit.jl")

export 
    # fit.jl
    fit_model_glopt_bound,
    fit_model_glopt_bound_reg,
    fit_model_nlopt_bound,
    fit_model_nlopt_bound_reg,

    # model.jl
    ewma,
    ModelEncoder,
    n_ps,
    generate_model_f!,
    init_model_ps!,
    lesser,

    # nl5
    generate_model_nl5,
    generate_model_nl5_partial,
    init_ps_model_nl5,
    ModelEncoderNL5,
    
    # nl6
    generate_model_nl6,
    generate_model_nl6_partial,
    init_ps_model_nl6,
    generate_model_nl6a,
    init_ps_model_nl6a,
    generate_model_nl6b,
    init_ps_model_nl6b,
    generate_model_nl6c,
    init_ps_model_nl6c,
    generate_model_nl6d,
    init_ps_model_nl6d,
    init_ps_model_nl6e,
    generate_model_nl6f,
    init_ps_model_nl6e,
    ModelEncoderNL6,
    ModelEncoderNL6a,
    ModelEncoderNL6b,
    ModelEncoderNL6c,
    ModelEncoderNL6d,
    ModelEncoderNL6e,
    ModelEncoderNL6f,

    # nl7
    generate_model_nl7,
    generate_model_nl7_partial,
    init_ps_model_nl7,
    init_ps_model_nl7_component,
    ModelEncoderNL7,
    ModelEncoderNL7a,

    # nl10
    generate_model_nl10c,
    generate_model_nl10c_partial,
    init_ps_model_nl10c,
    init_ps_model_nl10c_component,
    ModelEncoderNL10c,
    generate_model_nl10d,
    generate_model_nl10d_partial,
    init_ps_model_nl10d,
    init_ps_model_nl10d_component,
    ModelEncoderNL10d,
    generate_model_nl10e,
    generate_model_nl10e_partial,
    init_ps_model_nl10e,
    init_ps_model_nl10e_component,
    ModelEncoderNL10e
end
