function get_basename(bname, t, c)
    bname * "_t" * lpad(string(t), 4, "0") * "_ch" * string(c)
end

"""
    nd2_to_nrrd(path_nd2, path_save,
        spacing_lat, spacing_axi, generate_MIP::Bool;
        θ=nothing, x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
        NRRD_dir_name="NRRD", MIP_dir_name="MIP", n_bin=nothing, n_z=nothing, label_offset::Int=0)

Saves nd2 into NRRD files after rotating and cropping. Rotation is skipped if
θ is set to `nothing`.

Arguments
---------
* `path_nd2`: path of .nd2 file to use
* `path_save`: path of .h5 file to save
* `spacing_lat`: lateral spacing (for logging)
* `spacing_axi`: axial spacing (for logging)
* `generate_MIP`: if true, save MIP in as preview
* `θ`: yaw angle (lateral rotation, radian). nothing if no rotation
* `x_crop`: x range to use. Full range if nothing
* `y_crop`: y range to use. Full range if nothing
* `z_crop`: z range to use. Full range if nothing
* `chs`: ch to use
* `NRRD_dir_name`: name of the subfolder to save NRRD files
* `MIP_dir_name`: name of the subfolder to save MIP files
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack for a continuous timestream data series
* `label_offset::Int`: offset to save images
"""
function nd2_to_nrrd(path_nd2, path_save,
    spacing_lat, spacing_axi, generate_MIP::Bool;
    θ=nothing, x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
    NRRD_dir_name="NRRD", MIP_dir_name="MIP", n_bin=nothing, n_z=nothing, label_offset::Int=0)

    mhd_paths = []
    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)

    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end

    # directories
    bname = splitext(basename(path_nd2))[1]

    path_dir_NRRD = joinpath(path_save, NRRD_dir_name)
    path_dir_MIP = joinpath(path_save, MIP_dir_name)

    create_dir(path_save)
    create_dir(path_dir_NRRD)
    generate_MIP && create_dir(path_dir_MIP)

    x_size_save = Int(0)
    y_size_save = Int(0)
    z_size_save = Int(0)

    if z_crop == nothing
        z_size_save = z_size
        z_crop = 1:z_size
    else
        z_size_save = length(z_crop)
    end
    if x_crop == nothing
        x_size_save = x_size
        x_crop = 1:x_size
    else
        x_size_save = length(x_crop)
    end
    if y_crop == nothing
        y_size_save = y_size
        y_crop = 1:y_size
    else
        y_size_save = length(y_crop)
    end


    img = zeros(Float64, x_size, y_size)
    vol = zeros(UInt16, x_size_save, y_size_save, z_size_save)

    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        @showprogress for t = 1:t_size
            for c = chs
                for (n, z) = enumerate(z_crop)
                    # load
                    if isnothing(n_z)
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=t-1, z=z-1)))
                    else
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=0, z=n_z*(t-1)+z-1)))
                    end
                    # binning
                    if !isnothing(n_bin)
                        img .= bin_img(img, n_bin)
                    end

                    # rotate, crop, convert to UInt16
                    if isnothing(θ)
                        vol[:,:,n] = round.(UInt16, img[x_crop, y_crop])
                    else
                        vol[:,:,n] = round.(UInt16,
                            rotate_img(img, θ)[x_crop, y_crop])
                    end
                end

                save_basename = get_basename(bname, t + label_offset, c)
                path_file_nrrd = joinpath(path_dir_NRRD, save_basename * ".nrrd")

                # save NRRD
                write_nrrd(path_file_nrrd, vol, (spacing_lat, spacing_lat, spacing_axi))

                # save MIP
                if generate_MIP
                    path_file_MIP = joinpath(path_dir_MIP, save_basename * ".png")
                    imsave(path_file_MIP, dropdims(maximum(vol, dims=3), dims=3),
                        cmap="gray")
                end
            end # for c
        end # for t
    end # pywith
end # function

"""
    nd2_to_mhd(path_nd2, path_save,
        spacing_lat, spacing_axi, generate_MIP::Bool;
        θ=nothing, x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
        MHD_dir_name="MHD", MIP_dir_name="MIP", n_bin=nothing, n_z=nothing)

Saves nd2 into MHD files after rotating and cropping. Rotation is skipped if
θ is set to `nothing`.

Arguments
---------
* `path_nd2`: path of .nd2 file to use
* `path_save`: path of .h5 file to save
* `spacing_lat`: lateral spacing (for logging)
* `spacing_axi`: axial spacing (for logging)
* `generate_MIP`: if true, save MIP in as preview
* `θ`: yaw angle (lateral rotation, radian). nothing if no rotation
* `x_crop`: x range to use. Full range if nothing
* `y_crop`: y range to use. Full range if nothing
* `z_crop`: z range to use. Full range if nothing
* `chs`: ch to use
* `MHD_dir_name`: name of the subfolder to save MHD files
* `MIP_dir_name`: name of the subfolder to save MIP files
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack for a continuous timestream data series
* `label_offset::Int`: offset to save images
"""
function nd2_to_mhd(path_nd2, path_save,
    spacing_lat, spacing_axi, generate_MIP::Bool;
    θ=nothing, x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
    MHD_dir_name="MHD", MIP_dir_name="MIP", n_bin=nothing, n_z=nothing, label_offset::Int=0)

    mhd_paths = []
    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)

    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end

    # directories
    bname = splitext(basename(path_nd2))[1]

    path_dir_MHD = joinpath(path_save, MHD_dir_name)
    path_dir_MIP = joinpath(path_save, MIP_dir_name)

    create_dir(path_save)
    create_dir(path_dir_MHD)
    generate_MIP && create_dir(path_dir_MIP)

    x_size_save = Int(0)
    y_size_save = Int(0)
    z_size_save = Int(0)

    if z_crop == nothing
        z_size_save = z_size
        z_crop = 1:z_size
    else
        z_size_save = length(z_crop)
    end
    if x_crop == nothing
        x_size_save = x_size
        x_crop = 1:x_size
    else
        x_size_save = length(x_crop)
    end
    if y_crop == nothing
        y_size_save = y_size
        y_crop = 1:y_size
    else
        y_size_save = length(y_crop)
    end


    img = zeros(Float64, x_size, y_size)
    vol = zeros(UInt16, x_size_save, y_size_save, z_size_save)

    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        @showprogress for t = 1:t_size
            for c = chs
                for (n, z) = enumerate(z_crop)
                    # load
                    if isnothing(n_z)
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=t-1, z=z-1)))
                    else
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=0, z=n_z*(t-1)+z-1)))
                    end
                    # binning
                    if !isnothing(n_bin)
                        img .= bin_img(img, n_bin)
                    end

                    # rotate, crop, convert to UInt16
                    if isnothing(θ)
                        vol[:,:,n] = round.(UInt16, img[x_crop, y_crop])
                    else
                        vol[:,:,n] = round.(UInt16,
                            rotate_img(img, θ)[x_crop, y_crop])
                    end
                end

                save_basename = get_basename(bname, t + label_offset, c)
                path_file_MHD = joinpath(path_dir_MHD, save_basename * ".mhd")
                path_file_raw = joinpath(path_dir_MHD, save_basename * ".raw")

                # save MHD
                write_raw(path_file_raw, vol)
                write_MHD_spec(path_file_MHD, spacing_lat, spacing_axi,
                        x_size_save, y_size_save, z_size_save,
                            save_basename * ".raw")

                # save MIP
                if generate_MIP
                    path_file_MIP = joinpath(path_dir_MIP, save_basename * ".png")
                    imsave(path_file_MIP, dropdims(maximum(vol, dims=3), dims=3),
                        cmap="gray")
                end
            end # for c
        end # for t
    end # pywith
end # function

"""
    nd2_to_h5(path_nd2, path_save, spacing_lat, spacing_axi; θ=nothing,
        x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
        z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
        n_bin=nothing, n_z=nothing)

Saves nd2 into HDF5 file after rotating and cropping. Rotation is skipped if
θ is set to `nothing`. Note: indexing is 1 based. Array axis is in the following
order: [x, y, z, t, c]. HDF5 file is chunked with:
(x size, y size, z size, 1, 1, 1)

Arguments
---------
* `path_nd2`: path of .nd2 file to use
* `path_save`: path of .h5 file to save
* `spacing_lat`: lateral spacing (for logging)
* `spacing_axi`: axial spacing (for logging)
* `θ`: yaw angle (lateral rotation, rauab). nothing if no rotation
* `x_crop`: x range to use. Full range if nothing
* `y_crop`: y range to use. Full range if nothing
* `z_crop`: z range to use. Full range if nothing
* `chs`: ch to use
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack for a continuous timestream data series
"""
function nd2_to_h5(path_nd2, path_save, spacing_lat, spacing_axi; θ=nothing,
    x_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    y_crop::Union{Nothing, UnitRange{Int64}}=nothing,
    z_crop::Union{Nothing, UnitRange{Int64}}=nothing, chs::Array{Int}=[1],
    n_bin=nothing, n_z=nothing)

    if splitext(path_save)[2] != ".h5"
        error("path_save must end with .h5")
    end

    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)

    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end

    x_size_save = Int(0)
    y_size_save = Int(0)
    z_size_save = Int(0)

    if z_crop == nothing
        z_size_save = z_size
        z_crop = 1:z_size
    else
        z_size_save = length(z_crop)
    end
    if x_crop == nothing
        x_size_save = x_size
        x_crop = 1:x_size
    else
        x_size_save = length(x_crop)
    end
    if y_crop == nothing
        y_size_save = y_size
        y_crop = 1:y_size
    else
        y_size_save = length(y_crop)
    end


    im_ = zeros(UInt16, x_size, y_size)

    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin

        h5open(path_save, "w") do f
            dset = create_dataset(f, "data", datatype(UInt16),
            dataspace(x_size_save, y_size_save, z_size_save, t_size,
                length(chs)), chunk=(x_size_save, y_size_save, 1, 1, 1), compress=5)

            @showprogress for t = 1:t_size
                for (i_c, c) = enumerate(chs)
                    for (i_z, z) = enumerate(z_crop)
                        # load
                        if isnothing(n_z)
                            img = Float64.(transpose(images.get_frame_2D(
                                c=c-1, t=t-1, z=z-1)))
                        else
                            img = Float64.(transpose(images.get_frame_2D(
                                c=c-1, t=0, z=n_z*(t-1)+z-1)))
                        end

                        # binning
                        if !isnothing(n_bin)
                            img = bin_img(img, n_bin)
                        end

                        # rotate, crop, convert to UInt16, and save
                        if isnothing(θ)
                            dset[:, :, i_z, t, i_c] = round.(UInt16,
                            img[x_crop, y_crop])
                        else
                            dset[:, :, i_z, t, i_c] = round.(UInt16,
                                rotate_img(img_, θ)[x_crop, y_crop])
                        end
                    end # for
                end # for
            end # for
        end # h5open

        if isnothing(θ) θ = 0.0 end
        dict_attr = Dict("x_crop"=>[x_crop.start, x_crop.stop],
            "y_crop"=>[y_crop.start, y_crop.stop],
            "z_crop"=>[z_crop.start, z_crop.stop], "θ"=>θ,
            "spacing_lat"=>spacing_lat, "spacing_axi"=>spacing_axi)
        h5writeattr(path_save, "data", dict_attr)
    end # pywith
end

"""
    function write_nd2_preview(path_nd2; prjdim=3, chs=[1], z_crop=:drop_first)

Saves maximum intensity projection (MIP) of nd2 file and make movies of the
time series.

Arguments
---------
* `path_nd2`: path of .nd2 file to use
* `prjdim`: MIP projection dimension. Default: 3 (across z slices)
* `z_crop`: z range to use. Full range if nothing. `:drop_first` drops 1st frame
* `chs`: ch to use
* `dir_save`: directory to save MIP images and movies
* `n_bin`: number of rounds to bin. e.g. `n_bin=2` results in 4x4 binning
* `n_z`: number of frames per z-stack, if using a continuous timestream data series
* `vmax`: vmax for img export
* `list_fps`: list of fps for the videos
"""
function write_nd2_preview(path_nd2; prjdim=3, chs=[1], z_crop=nothing,
    dir_save=nothing, n_bin=nothing, n_z=nothing, vmax=1000, list_fps=[30,60,120])
    x_size, y_size, z_size, t_size, c_size = nd2dim(path_nd2)
    # binning
    if !isnothing(n_z)
        z_size = n_z
        t_size = t_size ÷ n_z
    end
    if !isnothing(n_bin)
        x_size = floor(Int, x_size / (2 ^ n_bin))
        y_size = floor(Int, y_size / (2 ^ n_bin))
    end

    # save dir
    if isnothing(dir_save)
        dir_save = dirname(path_nd2)
    end
    dir_MIP = joinpath(dir_save, "MIP_original")
    dir_movie = joinpath(dir_save, "movie_original")
    bname = splitext(basename(path_nd2))[1]
    create_dir(dir_MIP)
    create_dir(dir_movie)

    # size determination
    if z_crop == nothing
        z_size_save = z_size
        z_crop = 1:z_size
    else
        z_size_save = length(z_crop)
    end

    vol = zeros(Float64, x_size, y_size, z_size)
    img = zeros(Float64, x_size, y_size)
    
    @pywith py_nd2reader.ND2Reader(path_nd2) as images begin
        @showprogress for t = 1:t_size
            for c = chs
                for (n, z) = enumerate(z_crop)
                    # load
                    if isnothing(n_z)
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=t-1, z=z-1)))
                    else
                        img .= Float64.(transpose(images.get_frame_2D(c=c-1,
                            t=0, z=n_z*(t-1)+z-1)))
                    end
                    # binning
                    if !isnothing(n_bin)
                        img .= bin_img(img, n_bin)
                    end
                    vol[:,:,n] .= img
                    
                    img_MIP = Gray{N0f8}.(clamp01.(maxprj(vol, dims=prjdim) ./ vmax))
                    
                    # write MIP
                    path_png = joinpath(dir_MIP, get_basename(bname, t, c) * ".png")
                    save(path_png, img_MIP)
                end # z
            end # ch
        end # t
    end # py nd2
    
    encoder_options = (crf="10", preset="veryslow")
    target_pix_fmt = VideoIO.AV_PIX_FMT_YUV420P
    for c = chs, fps_ = list_fps
        path_vid =  joinpath(dir_movie, bname * "_ch" * lpad(string(c), 2, "0") *
            "_original_$(fps_)fps.mp4")
        path_png = joinpath(dir_MIP, get_basename(bname, 1, c) * ".png")
        open_video_out(path_vid, load(path_png), framerate=fps_,
            encoder_options=encoder_options, codec_name="libx264",
            target_pix_fmt=target_pix_fmt) do vidf
            for t = 2:t_size
                path_png = joinpath(dir_MIP, get_basename(bname, t, c) * ".png")
                write(vidf, load(path_png))
            end # t
        end # vid
    end # c
end
