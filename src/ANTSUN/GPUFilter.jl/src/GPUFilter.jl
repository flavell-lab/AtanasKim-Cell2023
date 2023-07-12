module GPUFilter

using Statistics, CUDA, Distributions

include("kernel.jl")
include("caller.jl")
include("util.jl")

export
    # util.jl,
    compute_λ_filt,
    filter_rof,

    # kernel.jl
    kernel_rof,

    # caller.jl
    gpu_imROF
    
end # module
