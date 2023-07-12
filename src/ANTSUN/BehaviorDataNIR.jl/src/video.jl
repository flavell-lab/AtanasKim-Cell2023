"""
    write_behavior_video(path_h5, path_vid=nothing; fps=20, encoder_options=nothing, downsample=true)
Writes a video of the behavior data
Arguments
---------
* `path_h5`: path of the HDF5 file  
* `path_vid`: path of the video to be generated. if `nothing` (default), automatically generates it  
* `fps`: frame rate of the video (default: 20 to match the acquisition rate)  
* `downsample`: if true downsample by factor of 2   
* `encoder_options`: in named tuple (e.g. `(crf="23", preset="slow")`)
* `vars`: variables to display in the video represented as a tuple `(varname, value_arr, color)`. Default `nothing` which does not display any text
* `text_pos`: position of variables text (default `(5,5)`)
* `text_size`: size of variables text (default `20`)
* `text_font`: font of variables text (default `Futura`)
* `text_spacing`: vertical spacing of variables text (default `1.05`)
### Notes on the encoder options  
`preset`: possible options are: ultrafast, superfast, veryfast, faster, fast, medium – default preset, slow, slower, veryslow.  
`crf`: The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible.  
A lower value generally leads to higher quality, and a subjectively sane range is 17–28.  
Consider 17 or 18 to be visually lossless or nearly so; it should look the same or nearly the same as the input but it isn't technically lossless.  
"""
function write_behavior_video(path_h5, path_vid=nothing; fps=20, encoder_options=nothing, downsample=true,
        vars=nothing, text_pos=(5,5), text_size=20, text_font="Futura", text_spacing=1.05)
    path_vid = isnothing(path_vid) ? splitext(path_h5)[1] * "_$(fps)fps.mp4" : path_vid
    if splitext(path_vid)[2] !== ".mp4"
        error("`path_vid` extension must be .mp4")
    end

    encoder_options = (crf="23", preset="slow")
    target_pix_fmt = VideoIO.AV_PIX_FMT_YUV420P

    h5open(path_h5, "r") do h5f
        img_nir = h5f["img_nir"]
        n_x, n_y, n_t = size(img_nir)

        img_t1 = (downsample ? floor.(ds(img_nir[:,:,1])) : img_nir[:,:,1])'
        if !isnothing(vars)
            text_arr = ["$(var[1]): $(var[2][1])" for var in vars]
            text_color = [var[3] for var in vars]
            img_t1 = add_text_to_image(img_t1, text_arr, text_pos, text_color, text_size, text_font, text_spacing)
        end
            
        
        open_video_out(path_vid, img_t1, framerate=fps,
            encoder_options=encoder_options, codec_name="libx264",
            target_pix_fmt=target_pix_fmt) do vidf
            @showprogress for t = 2:n_t
                img_ = (downsample ? floor.(ds(img_nir[:,:,t])) : img_nir[:,:,t])'
                if !isnothing(vars)
                    text_arr = ["$(var[1]): $(var[2][t])" for var in vars]
                    img_ = add_text_to_image(img_, text_arr, text_pos, text_color, text_size, text_font, text_spacing)
                end
                write(vidf, img_)
            end # t
        end # vid
    end #h5open
end

function add_text_to_image(img, text_arr, position, colors, fs, font, spacing)
    len, wid = size(img)
    word_img = zeros(ARGB32,len,wid)
    new_img = uint8_to_rgb(img)
    pos = collect(position)
    n0f8_colors = RGB{N0f8}.(colors)
    idx = 1
    for text in text_arr
        word_img_add = @imagematrix begin
            Drawing(wid,len)
            sethue(colors[idx])
            fontsize(fs)
            fontface(font)
            Luxor.text(text, Point(pos[1], pos[2]), halign=:left, valign=:top)
        end
        word_img .+= word_img_add
        idx += 1
        pos .+= [0, ceil(spacing*fs)]
    end
  
    
    
    for i=1:len
        for j=1:wid
            if word_img[i,j].color != 0
                new_img[i,j] = RGB{N0f8}(word_img[i,j]) * alpha(word_img[i,j]) + new_img[i,j] * (1 - alpha(word_img[i,j]))
            end
        end
    end
    
    return new_img
end

function uint8_to_rgba(img, alpha=1.0)
    img_ = reinterpret.(N0f8, img)
    RGBA.(img_, img_, img_, fill(N0f8(alpha), size(img_)...))
end

function uint8_to_rgb(img)
    img_ = reinterpret.(N0f8, img)
    RGB.(img_, img_, img_)
end

function ds(img)
    img = Float64.(img)
    round.(UInt8, (img[1:2:end-1,1:2:end-1] .+ img[2:2:end,2:2:end] .+
            img[1:2:end-1,2:2:end] .+ img[2:2:end,1:2:end-1]) ./ 4)
end


function encode_movie(input, output; fps=30)
    run(`ffmpeg -hide_banner -loglevel panic -y -framerate $fps -i $input -c:v libx264 -pix_fmt yuv420p -preset slow -b:v 16M $output`)
    nothing
end;


"""
Writes maximum-intensity projection video of MHD data.

# Arguments:
- `param_path::Dict`: Parameter path dictionary containing `path_dir_mhd` and `get_basename` keys
- `num_timepts`: Number of timepoints to create video of
- `ch`: Confocal channel to use
- `path_vid`: Path to output video
- `fps` (default 20): frames per second
- `encoder_options`: in named tuple (e.g. `(crf="23", preset="slow")`)
"""
function write_mip_video(param_path, num_timepts, ch, path_vid=nothing; fps=20, encoder_options=nothing)
    path_vid = isnothing(path_vid) ? splitext(path_h5)[1] * "_$(fps)fps.mp4" : path_vid
    if splitext(path_vid)[2] !== ".mp4"
        error("`path_vid` extension must be .mp4")
    end

    encoder_options = (crf="23", preset="slow")
    target_pix_fmt = VideoIO.AV_PIX_FMT_YUV420P

    img_t1 = maxprj(read_img(MHD(joinpath(param_path["path_dir_mhd"], param_path["get_basename"](1, ch)*".mhd"))),dims=3)

    open_video_out(path_vid, img_t1, framerate=fps,
        encoder_options=encoder_options, codec_name="libx264",
        target_pix_fmt=target_pix_fmt) do vidf
        @showprogress for t = 2:num_timepts
            img_ = maxprj(read_img(MHD(joinpath(param_path["path_dir_mhd"], param_path["get_basename"](t, ch)*".mhd"))),dims=3)
            write(vidf, img_)
        end # t
    end # vid
end
