module FFTRegGPU
using CUDA, AbstractFFTs

include("dftreg_translate.jl")

export dftreg_gpu!,
	subpix_shift_gpu!,
	dftreg_resample_gpu!,
	reg_stack_translate!

end # module
