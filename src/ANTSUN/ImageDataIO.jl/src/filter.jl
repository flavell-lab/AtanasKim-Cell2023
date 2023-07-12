"""
    get_λ(img_list::Array; verbose=true)

Gets total variation noise filtering parameters from a list of images `img_list`. Can set `verbose` to `false` to suppress output.
"""
function get_λ(img_list::Array; verbose=true)
    # img_array dim: x, y, z, t
    n_t = length(img_list)

    θ_list = []
    for t = 1:n_t
        img_MIP = Float32.(maxprj(img_list[t], dims=3))
        push!(θ_list, compute_λ_filt(img_MIP))
    end
    
    λ_μ = mean(θ_list)
    λ_σ = std(θ_list)
    verbose && println("mean(λ): $(λ_μ) std(λ): $(λ_σ)")
    
    λ_μ
end

"""
    filter_nrrd_gpu(
        param_path::Dict, path_dir_nrrd::String, t_range, list_ch,
        f_basename::Function; nrrd_filt_dir_key::String="path_dir_nrrd_filt", 
        mip_filt_dir_key::String="path_dir_MIP_filt", vmax=1600
    )

Runs total variation filtering on a set of images.

# Arguments
 - `param_path::Dict`: Dictionary of locations of data
 - `path_dir_nrrd::String`: Location of NRRD files to filter
 - `t_range`: Time points to filter
 - `list_ch`: Channels to filter
 - `f_basename::Function`: Function that returns NRRD filename given time point and channel
 - `nrrd_filt_dir_key::String` (optional): Key in `param_path` that maps to the location to store
the output NRRD files. Default `path_dir_nrrd_filt`
 - `mip_filt_dir_key::String` (optional): Key in `param_path` that maps to the location to store
the output MIP files. Default `path_dir_MIP_filt`
 - `vmax`: Contrast setting for png files
"""
function filter_nrrd_gpu(param_path::Dict, path_dir_nrrd::String, t_range, list_ch,
        f_basename::Function; nrrd_filt_dir_key::String="path_dir_nrrd_filt", 
        mip_filt_dir_key::String="path_dir_MIP_filt", vmax=1600)
    path_dir_nrrd_filt = param_path[nrrd_filt_dir_key]
    path_dir_MIP_filt = param_path[mip_filt_dir_key]
    create_dir.([path_dir_nrrd_filt, path_dir_MIP_filt])
    
    # getting image size
    path_nrrd = joinpath(path_dir_nrrd, f_basename(t_range[1], list_ch[1]) * ".nrrd")
    img = read_img(NRRD(path_nrrd))
    type_img = eltype(img)
    n_t = length(t_range)
    
    # getting the filter parameter
    list_λ = []
    for ch = list_ch
        list_img_λ = []
        for t = [round(Int, n_t * i / 10) for i = 1:10]
            if !(t in t_range)
                continue
            end
            path_nrrd = joinpath(path_dir_nrrd, f_basename(t, ch) * ".nrrd")
            push!(list_img_λ, read_img(NRRD(path_nrrd)))
        end
        println("ch$ch parameter:")
        push!(list_λ, get_λ(list_img_λ))
    end

    # filtering
    @showprogress for t = t_range
        for (i_ch, ch) = enumerate(list_ch)
            λ_ch = list_λ[i_ch]
            basename = f_basename(t, ch)
            path_nrrd = joinpath(path_dir_nrrd, basename * ".nrrd")
            path_nrrd_filt = joinpath(path_dir_nrrd_filt, basename * ".nrrd")
            path_MIP = joinpath(path_dir_MIP_filt, basename * ".png")
            
            nrrd_in = NRRD(path_nrrd)
            img = Float32.(read_img(nrrd_in))
            n_x, n_y, n_z = size(img)
            img_filt = zeros(type_img, n_x, n_y, n_z)
            for z = 1:n_z
                img_filt[:,:,z] .= round.(type_img, gpu_imROF(img[:,:,z], λ_ch, 100))
            end

            write_nrrd(path_nrrd_filt, img_filt, spacing(nrrd_in))
            imsave(path_MIP, maxprj(img_filt, dims=3) / vmax, cmap="gray");
        end
    end
end