"""
    nd2read(path_nd2; ch=1, t=1, n_bin=nothing, n_z=nothing)

Read the image at ch and t

Arguments
---------
* `path_nd2`: .nd2 file to read
* `ch`: ch to read. First ch: 1
* `t`: time point to read. Can be multiple (e.g. [1,2,10], 1:20). First t: 1
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack for a continuous timestream data series
"""
function nd2read(path_nd2; ch=1, t=1, n_bin=nothing, n_z=nothing)
    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end
    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    stack_ = (length(t) == 1) ? zeros(UInt16, x_size, y_size, z_size) : zeros(
        UInt16, x_size, y_size, z_size, length(t))

    if typeof(t) == Int t = [t] end

    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        for (n_, t_) = enumerate(t)
            for z_ = 1:z_size
                if isnothing(n_z)
                    img_ = transpose(images.get_frame_2D(c=ch-1,
                        t=t_-1, z=z_-1))
                else
                    img_ = transpose(images.get_frame_2D(c=ch-1,
                        t=0, z=n_z*(t_-1)+z_-1))
                end
                stack_[:,:,z_,n_] = !isnothing(n_bin) ? round.(UInt16,
                    bin_img(img_, n_bin)) : img_
            end
        end
    end

    stack_
end

"""
    nd2dim(path_nd2)

Returns dim of the file in (x, y, z, t, c)

Arguments
---------
* `path_nd2`: .nd2 file to read
* `verbose`: if true, print out the dimensions
"""
function nd2dim(path_nd2, verbose=false)
    if !isfile(path_nd2) error(".nd2 file does not exist.") end

    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        @assert eltype(images.get_frame_2D(c=0,t=0,z=0)) in [Float64, UInt16]
        x_size, y_size, c_size, t_size, z_size = [get(images.sizes, k, 1) for
            k = ["x", "y", "c", "t", "z"]]

        if verbose
            println("x:$x_size, y:$y_size, c:$c_size, t:$t_size, z:$z_size")
        end

        return (x_size, y_size, z_size, t_size, c_size)
    end
end

function rotate_img(img, θ)
    tfm = recenter(Rotations.RotMatrix(θ), center(img))
    ImageTransformations.warp(img, tfm)
end

"""
    nd2preview(path_nd2; ch=1, return_data=false, z_crop=nothing,
        n_bin=nothing, n_z=nothing)

Preview MIP of first, middle, and last time points in the .nd2 file

Arguments
---------
* `path_nd2`: .nd2 file to read
* `ch`: ch to use. Default: 1 (first ch)
* `return_data`: if true returns the 3 images as array
* `z_crop`: selecting z range to use. e.g. `3:15` then only use slice 3 to 15
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack for a continuous timestream data series
"""
function nd2preview(path_nd2; ch=1, return_data=false, z_crop=nothing,
    n_bin=nothing, n_z=nothing)
    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end
    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    # first, middle, last time point
    t_list = t_size > 3 ? [1, round(Int, t_size / 2), t_size] : [1]
    z_size_use = Int(0)
    if z_crop == nothing
        z_size_use = z_size
        z_crop = 1:z_size
    else
        z_size_use = length(z_crop)
    end
    stack_ = zeros(UInt16, x_size, y_size, z_size_use, length(t_list))
    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        for (i_t, t_) = enumerate(t_list)
            for (i_z, z_) = enumerate(z_crop)
                if isnothing(n_z)
                    img_ = transpose(images.get_frame_2D(c=ch-1,
                        t=t_-1, z=z_-1))
                else
                    img_ = transpose(images.get_frame_2D(c=ch-1,
                        t=0, z=n_z*(t_-1)+z_-1))
                end
                # perform binning if requested and save to stack
                stack_[:,:,i_z,i_t] = !isnothing(n_bin) ? round.(eltype(img_),
                    bin_img(img_, n_bin)) : img_
            end
        end
    end

    stack_MIP = Float64.(maxprj(stack_, dims=3))

    for i = 1:length(t_list)
        subplot(1,length(t_list),i)
        imshow(stack_MIP[:,:,i])
        title("t=$(t_list[i]+1)")
    end
    tight_layout()


    return_data ? (return return_data) : (return nothing)
end

"""
    nd2preview_crop(stack::Array; θ, x_crop=nothing, y_crop=nothing,
        z_crop=nothing)

Preview MIP of rotatation and x, y, z cropping

Arguments
---------
* `stack`: array (e.g. returned from `nd2preview` with `return_data=true`)
containing 3 stacks; a 3D volume; or a 2D image.
* `θ`: yaw angle (lateral rotation)
* `x_crop`: range of x to use
* `y_crop`: range of y to use
* `z_crop`: range of z to use
"""
function nd2preview_crop(stack::Array; θ=nothing, x_crop=nothing,
    y_crop=nothing, z_crop=nothing)

    if z_crop == nothing
        z_crop = 1:size(stack, 3)
    end
    if y_crop == nothing
        y_crop = 1:size(stack, 2)
    end
    if x_crop == nothing
        x_crop = 1:size(stack, 1)
    end

    local stack_MIP_proc

    if size(stack, 4) == 3 # nd2preview returned 3 time points stack
        stack_MIP = Float64.(maxprj(stack[:,:,z_crop,:], dims=3))
        stack_MIP_proc = zeros(eltype(stack_MIP), length(x_crop),
            length(y_crop), 3)

        for i = 1:3
            if θ == nothing
                stack_MIP_proc[:,:,i] = stack_MIP[x_crop,y_crop,i]
            else
                stack_MIP_proc[:,:,i] = rotate_img(stack_MIP[:,:,i],
                    θ)[x_crop, y_crop]
            end
        end

        for i = 1:3
            subplot(1,3,i)
            imshow(stack_MIP_proc[:,:,i])
        end
        tight_layout()
    elseif size(stack, 3) != 1 # 3D volume
        stack_MIP = Float64.(maxprj(stack[:,:,z_crop], dims=3))
        if θ == nothing
            stack_MIP_proc = stack_MIP[x_crop,y_crop]
        else
            stack_MIP_proc = rotate_img(stack_MIP, θ)[x_crop, y_crop]
        end
        imshow(stack_MIP_proc)
    else # 2D image
        if θ == nothing
            stack_MIP_proc = stack[x_crop,y_crop]
        else
            stack_MIP_proc = rotate_img(Float64.(stack), θ)[x_crop, y_crop]
        end
        imshow(stack_MIP_proc)
    end

    if true in isnan.(stack_MIP_proc)
        @warn "transformed data contains NaN due to rotation."
    end

    println(size(stack_MIP_proc, 1) ./ [2,4,8,16])
    println(size(stack_MIP_proc, 2) ./ [2,4,8,16])
end
