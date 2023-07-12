"""
    shear_correction_nrrd!(
        param_path::Dict, param::Dict, ch::Int, shear_params_dict::Dict;
        vmax::Int=1600, nrrd_in_key::String="path_dir_nrrd", nrrd_out_key::String="path_dir_nrrd_shearcorrect",
        MIP_out_key::String="path_dir_MIP_shearcorrect"
    )

Applies shear correction to a dataset.
# Arguments
 - `param_path::Dict`: Dictionary containing locations of input and output directories with NRRD files
 - `param::Dict`: Dictionary containing image dimension parameters `x_range`, `y_range`, `z_range`, and `t_range`
 - `ch`: Channel to apply shear correction to
 - `shear_params_dict::Dict`: Dictionary of shear correction parameters. If nonempty, those parameters will be used.
    If empty, it will be filled with the computed paramers.
 - `vmax::Int` (optional, default 1600): Contrast parameter for png files
 - `nrrd_in_key::String` (optional): Key in `param_path` containing input NRRD directory
 - `NRRD_out_key::String` (optional): Key in `param_path` containing output NRRD directory
 - `MIP_out_key::String` (optional): Key in `param_path` containing output MIP directory
"""
function shear_correction_nrrd!(param_path::Dict, param::Dict, ch::Int, shear_params_dict::Dict;
        vmax::Int=1600, nrrd_in_key::String="path_dir_nrrd", nrrd_out_key::String="path_dir_nrrd_shearcorrect",
        MIP_out_key::String="path_dir_MIP_shearcorrect")
    create_dir(param_path[nrrd_out_key])
    create_dir(param_path[MIP_out_key])
    size_x, size_y, size_z = length(param["x_range"]), length(param["y_range"]), length(param["z_range"])
    
    img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
    img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
    CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
    N_g = CuArray{Float32}(undef, size_x, size_y)
    img_stack_reg = zeros(Float32, size_x, size_y, size_z)
    img_stack_reg_g = CuArray{Float32}(undef, size_x, size_y, size_z)

    @showprogress for t = param["t_range"]
        # check if parameters exist at the given time point
        if !haskey(shear_params_dict, t)
            shear_params_dict[t] = Dict()
        end
        basename = param_path["get_basename"](t, ch)
        path_nrrd_in = joinpath(param_path[nrrd_in_key], basename * ".nrrd")
        path_nrrd_out = joinpath(param_path[nrrd_out_key], basename * ".nrrd")
        path_MIP_out = joinpath(param_path[MIP_out_key], basename * ".png")
        
        nrrd_in = NRRD(path_nrrd_in)
        copyto!(img_stack_reg_g, Float32.(read_img(nrrd_in)))
        reg_stack_translate!(img_stack_reg_g, img1_f_g, img2_f_g, CC2x_g, N_g,
            reg_param=shear_params_dict[t])
        copyto!(img_stack_reg, img_stack_reg_g)
        
        write_nrrd(path_nrrd_out, floor.(UInt16, clamp.(img_stack_reg, typemin(UInt16), typemax(UInt16))),
            spacing(nrrd_in))
        imsave(path_MIP_out, maxprj(img_stack_reg, dims=3) / vmax, cmap="gray")
    end
 
    CUDA.unsafe_free!(img_stack_reg_g)
    CUDA.unsafe_free!(img1_f_g)
    CUDA.unsafe_free!(img2_f_g)
    CUDA.unsafe_free!(CC2x_g)
    CUDA.unsafe_free!(N_g)    
    GC.gc(true)
    CUDA.reclaim()
 
    return shear_params_dict
end