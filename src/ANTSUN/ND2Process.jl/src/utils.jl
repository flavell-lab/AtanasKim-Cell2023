function encode_movie(input, output; fps=10)
    input = `-i $input`
    options = `-hide_banner -loglevel panic -y -framerate $fps`
    codec = `-c:v libx264 -crf 16 -pix_fmt yuv420p -preset veryslow`
    run(`ffmpeg $options $input $codec $output`)
    nothing
end

function bin_img(data)
    (data[1:2:end-1,1:2:end-1] .+ data[2:2:end,2:2:end] .+
        data[1:2:end-1,2:2:end] .+ data[2:2:end,1:2:end-1])
end

function bin_img(data,n)
    foldl((x,y)->bin_img(x), 1:n, init=data)
end
