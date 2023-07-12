"""
    run_transformix_centroids(path, output, centroids, parameters, transformix_dir)

Runs transformix to map centroids from the fixed volume to the moving volume.
Note that this is the opposite of `run_transformix_roi`, because of the way `transformix` is implemented.

# Arguments

- `path::String`: Path to directory containing elastix registration
- `output::String`: Output file directory
- `centroids`: File containing centroids to transform
- `parameters::String`: Transform parameter file
- `transformix_dir::String`: Path to `transformix` executable
"""
function run_transformix_centroids(path, output, centroids, parameters, transformix_dir)
    cmd = Cmd(Cmd([transformix_dir, "-out $output", "-def $centroids", "-tp $parameters"]), dir=path)
    result = read(cmd)
    return (read_img(NRRD(joinpath(output, "result.nrrd"))), result)
end

"""
    run_transformix_img(path::String, output::String, input::String, parameters::String, parameters_roi::String, transformix_dir::String; interpolation_degree::Int=1)

Runs transformix to map an image from the moving volume to the fixed volume. 
Returns the resulting transformed image.
Note that this is the opposite of `run_transformix_centroids`, because of the way `transformix` is implemented.

# Arguments
- `path::String`: Path to directory containing elastix registration
- `output::String`: Output file directory
- `input::String`: Input file path
- `parameters::String`: Transform parameter file
- `transformix_dir::String`: Path to `transformix` executable
- `interpolation_degree::Int` (optional): Interpolation degree. Default 1 (linear interpolation).
"""
function run_transformix_img(path::String, output::String, input::String, parameters::String, parameters_roi::String, transformix_dir::String; interpolation_degree::Int=1)
    create_dir(output)
    modify_parameter_file(parameters, parameters_roi, Dict("FinalBSplineInterpolationOrder" => interpolation_degree))
    cmd = Cmd(Cmd([transformix_dir, "-out", output, "-in", input, "-tp", parameters_roi]), dir=path)
    result = read(cmd)
    return (read_img(NRRD(joinpath(output, "result.nrrd"))), result)
end


"""
    run_transformix_roi(path::String, input::String, output::String, parameters::String, parameters_roi::String, transformix_dir::String)    

Runs transformix to map an ROI image from the moving volume to the fixed volume. 
Modifies various transform parameters to force nearest-neighbors interpolation.
Returns the resulting transformed image.
Note that this is the opposite of `run_transformix_centroids`, because of the way `transformix` is implemented.

# Arguments
- `path::String`: Path to directory containing elastix registration. Other paths should be absolute; they are NOT relative to this.
- `input::String`: Input file path
- `output::String`: Output file directory path
- `parameters::String`: Path to transform parameter file
- `parameters_roi::String`: Filename to save modified parameters.
- `transformix_dir::String`: Path to `transformix` executable
"""
function run_transformix_roi(path::String, input::String, output::String, parameters::String, parameters_roi::String, transformix_dir::String)
    create_dir(output)
    modify_parameter_file(parameters, parameters_roi, Dict("FinalBSplineInterpolationOrder" => 0, "DefaultPixelValue" => 0))
    cmd = Cmd(Cmd([transformix_dir, "-out", output, "-in", input, "-tp", parameters_roi]), dir=path)
    result = read(cmd)
    return (read_img(NRRD(joinpath(output, "result.nrrd"))), result)
end

