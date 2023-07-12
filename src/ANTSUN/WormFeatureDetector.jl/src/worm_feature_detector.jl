"""
    find_gut_granules_img(img::Array{UInt16, 3}, threshold::Real, density::Real, radius)

Generates a mask containing predicted gut granules from an image. The mask has the same dimensions as the image,
with entry 1 for a pixel that is a gut granule, and 0 for a pixel that isn't.

# Arguments
- `img`: 3-dimensional volume of imaging data
- `threshold::Real`: pixel intensity brightness value. Pixels below this intensity are excluded
- `density::Real`: density of nearby pixels that must meet the threshold for the original pixel to be counted
- `radius`: distances in each dimension (in pixels) away from the original pixel that are counted as nearby.
    For example, `radius = [3, 2, 1]` would allow a distance of three pixels in the x-direction, two pixels in the y-direction,
    and one pixel in the z-direction.
"""
function find_gut_granules_img(img::Array{UInt16, 3}, threshold::Real, density::Real, radius)
    sx, sy, sz = size(img)
    dx,dy,dz = radius
    result_img = zeros(UInt16, size(img))
    m = median(img) * threshold
    thresh_img = img .> m
    density_threshold = density * sum(ones(2*dx+1, 2*dy+1, 2*dz+1))
    for x in 1:sx
        for y in 1:sy
            for z in 1:sz
                if thresh_img[x,y,z]
                    result_img[x,y,z] = (sum(thresh_img[max(1,x-dx):min(x+dx,sx),max(1,y-dy):min(y+dy,sy),max(1,z-dz):min(z+dz,sz)]) > density_threshold)
                end
            end
        end
    end
    return result_img
end

"""
    find_gut_granules(path::String, names::Array{String, 1}, threshold, density, radius; dir_nrrd="NRRD", out="")

Finds, returns, and outputs gut granule masks for a set of images. The masks filter out the gut granules, so
1 is not a gut granule, and 0 is a gut granule.

# Arguments
- `path::String`: working directory path; all other directory inputs are relative to this
- `names::Array{String,1}`: filenames of images to process
- `threshold::Real`: pixel intensity brightness value. Pixels below this intensity are excluded
- `density::Real`: density of nearby pixels that must meet the threshold for the original pixel to be counted
- `radius`: distances in each dimension (in pixels) away from the original pixel that are counted as nearby.
    For example, `radius = [3, 2, 1]` would allow a distance of three pixels in the x-direction, two pixels in the y-direction,
    and one pixel in the z-direction.

## Optional keyword arguments
- `nrrd::String`: path to NRRD directory, where the image will be found. Default "NRRD".
- `out::String`: path to output directory to store the masks. If left blank (default), images will not be written.
"""
function find_gut_granules(path::String, names::Array{String, 1}, threshold,
        density, radius; dir_nrrd="NRRD", out="")
    gut_imgs = []
    @showprogress for name in names
        nrrd = NRRD(joinpath(path, dir_nrrd, name * ".nrrd"))
        img = read_img(nrrd)
        gut_img = 0x0001 .- find_gut_granules_img(img, threshold, density, radius)
        if out != ""
            out_path = joinpath(path, out)
            create_dir(out_path)
            write_nrrd(joinpath(out_path, name*".nrrd"), gut_img, spacing(nrrd))
        end
        push!(gut_imgs, gut_img)
    end
    return gut_imgs
end


"""
    find_hsn(
        path::String, frames, dir_nrrd::String, img_prefix::String, channel::Integer,
        threshold_outer::Real, density_outer::Real, radius_outer, 
        threshold_inner::Real, density_inner::Real, radius_inner, radius_detection; outfile::String=""
    )

Finds location of the HSN soma in a frame. First threshold to remove densely-packed regions that might
correspond to gut fluorescence or neuropil (by excluding too-dense regions), then threshold again to ensure high local density
(by excluding not-dense regions). If multiple regions are still included, the larger region is chosen.
Can optionally choose to output data to a file, for use with heuristics.

# Arguments:
- `path::String`: working directory path; all other directory inputs are relative to this
- `frames`: frames of images to process
- `dir_nrrd::String`: path to NRRD directory, where the image will be found.
- `img_prefix::String`: image prefix not including the timestamp. It is assumed that each frame's filename
    will be, eg, `img_prefix_t0123_ch2.nrrd` for frame 123 with channel=2.
- `channel::Integer`: channel being used.
- `threshold_outer::Real`: pixel intensity brightness value. Pixels below this intensity are excluded
- `density_outer::Real`: density of nearby pixels that must meet the outer threshold for the original pixel to **NOT** be counted
- `radius_outer`: distances in each dimension (in pixels) away from the original pixel that are counted as nearby.
For example, `radius = [3, 2, 1]` would allow a distance of three pixels in the x-direction, two pixels in the y-direction,
and one pixel in the z-direction.
- `threshold_inner::Real`: pixel intensity brightness value. Pixels below this intensity are excluded
- `density_inner::Real`: density of nearby pixels that must meet the inner threshold for the original pixel to be counted
- `radius_inner`: distances in each dimension (in pixels) away from the original pixel that are counted as nearby.
For example, `radius = [3, 2, 1]` would allow a distance of three pixels in the x-direction, two pixels in the y-direction,
and one pixel in the z-direction.
- `radius_detection`: If multiple locations remain as possible HSN locations after both thresholding steps,
    the location with the most other such points within `radius_detection` of it is chosen. 

## Optional keyword arguments
- `outfile::String`: path to HSN output file. If left blank (default), no output will be written.
"""
function find_hsn(path::String, frames, dir_nrrd::String, img_prefix::String, channel::Integer, threshold_outer::Real, density_outer::Real, radius_outer, 
        threshold_inner::Real, density_inner::Real, radius_inner, radius_detection; outfile::String="")
    result_imgs = []
    hsn_locs_all = []
    best_hsn_locs = []
    n = length(frames)
    @showprogress for i=1:n
        frame = frames[i]
        img = read_nrrd(path, img_prefix, dir_nrrd, frame, channel)
        sx, sy, sz = size(img)
        dx,dy,dz = radius_outer
        result_img = zeros(UInt16, size(img))
        m = median(img) * threshold_outer
        thresh_img = img .> m
        density_threshold = density_outer * sum(ones(2*dx+1, 2*dy+1, 2*dz+1))
        for x in 1:sx
            for y in 1:sy
                for z in 1:sz
                    if thresh_img[x,y,z]
                        result_img[x,y,z] = (sum(thresh_img[max(1,x-dx):min(x+dx,sx),max(1,y-dy):min(y+dy,sy),max(1,z-dz):min(z+dz,sz)]) < density_threshold)
                    end
                end
            end
        end
        dx,dy,dz = radius_inner
        m = median(img) * threshold_inner
        thresh_img = img .> m
        density_threshold = density_inner * sum(ones(2*dx+1, 2*dy+1, 2*dz+1))
        for x in 1:sx
            for y in 1:sy
                for z in 1:sz
                    if !thresh_img[x,y,z]
                        result_img[x,y,z] = 0
                        continue
                    end
                    if result_img[x,y,z] > 0
                        result_img[x,y,z] = (sum(thresh_img[max(1,x-dx):min(x+dx,sx),max(1,y-dy):min(y+dy,sy),max(1,z-dz):min(z+dz,sz)]) > density_threshold)
                    end
                end
            end
        end
        dx,dy,dz = radius_detection
        hsn_locs = Dict()
        best_hsn_loc = nothing
        best_hsn_score = 0
        for x in 1:sx
            for y in 1:sy
                for z in 1:sz
                    if result_img[x,y,z] > 0
                        hsn_locs[(x,y,z)] = sum(result_img[max(1,x-dx):min(x+dx,sx),max(1,y-dy):min(y+dy,sy),max(1,z-dz):min(z+dz,sz)])
                        if hsn_locs[(x,y,z)] > best_hsn_score
                            best_hsn_loc = (x,y,z)
                            best_hsn_score = hsn_locs[(x,y,z)]
                        end
                    end
                end
            end
        end
        push!(result_imgs, result_img)
        push!(hsn_locs_all, hsn_locs)
        push!(best_hsn_locs, best_hsn_loc)
    end
    if outfile != ""
        open(joinpath(root, outfile), "w") do f
            for (i,name) in frames
                write(f, string(name)*" "*string(best_hsn_locs[i][1])*" "*string(best_hsn_locs[i][2])*" "*string(best_hsn_locs[i][3])*"\n")
            end
        end
    end

    return result_imgs, hsn_locs_all, best_hsn_locs
end


"""
    find_nerve_ring(path::String, frames, dir_nrrd::String, img_prefix::String, channel::Integer, threshold::Real, region, radius; outfile::String="")

Finds location of the nerve ring in a frame. Can optionally output nerve ring locations to a file for use with heuristics.

# Arguments:
- `path::String`: working directory path; all other directory inputs are relative to this
- `frames`: frames of images to process
- `dir_nrrd::String`: path to NRRD directory, where the image will be found.
- `img_prefix::String`: image prefix not including the timestamp. It is assumed that each frame's filename
    will be, eg, `img_prefix_t0123_ch2.nrrd` for frame 123 with channel=2.
- `channel::Integer`: channel being used.
- `threshold::Real`: pixel intensity brightness value. Pixels below this intensity are excluded
- `region`: region of the image that will be searched for the nerve ring. Generically, you should try to include the nerve ring
    and exclude any other regions (such as gut granules, HSN soma, etc)
- `radius`: The location with the most other points that meet the threshold within `radius` of it is chosen
    as the nerve ring location.

## Optional keyword arguments

- `outfile::String`: path to nerve ring output file. If left blank (default), no output will be written.
"""
function find_nerve_ring(path::String, frames, dir_nrrd::String, img_prefix::String, channel::Integer, threshold::Real, region, radius; outfile::String="")
    result_imgs = []
    nr_locs_all = []
    best_nr_locs = []
    nr_markers = []
    n = length(frames)
    @showprogress for i=1:n
        frame = frames[i]
        img = read_nrrd(path, img_prefix, dir_nrrd, frame, channel)
        result_img = zeros(UInt16, size(img))
        sx, sy, sz = size(img)
        dx, dy, dz = radius
        rx, ry, rz = region
        mx, my, mz = map(minimum, region)
        ax, ay, az = map(maximum, region)
        nr_locs = Dict()
        m = median(img[rx, ry, rz]) * threshold
        result_img = img .> m
        best_nr_score = 0
        best_nr_loc = nothing
        for x in rx
            for y in ry
                for z in rz
                    if result_img[x,y,z] > 0
                        nr_locs[(x,y,z)] = sum(result_img[max(mx,x-dx):min(x+dx,ax),max(my,y-dy):min(y+dy,ay),max(mz,z-dz):min(z+dz,az)])
                        if nr_locs[(x,y,z)] > best_nr_score
                            best_nr_loc = (x,y,z)
                            best_nr_score = nr_locs[(x,y,z)]
                        end
                    end
                end
            end
        end
        result_img_marker = zeros(size(result_img))
        x,y,z = best_nr_loc
        for dx in -3:3
            for dy in -3:3
                result_img_marker[max(mx,min(ax,x+dx)),max(my,min(ay,y+dy)),z] = 1
            end
        end
        push!(result_imgs, result_img)
        push!(nr_locs_all, nr_locs)
        push!(best_nr_locs, best_nr_loc)
        push!(nr_markers, result_img_marker)
    end
    if outfile != ""
        open(joinpath(root, outfile), "w") do f
            for (i,name) in frames
                write(f, string(name)*" "*string(best_nr_locs[i][1])*" "*string(best_nr_locs[i][2])*" "*string(best_nr_locs[i][3])*"\n")
            end
        end
    end
    return result_imgs, nr_locs_all, best_nr_locs, nr_markers
end


