"""
    cart2spher(x_cart::Vector)

Converts Cartesian coordinates to spherical coordinates.

# Arguments
- `x_cart::Vector`: A 3-element vector representing Cartesian coordinates (x, y, z).

# Returns
- A 3-element vector representing spherical coordinates (r, theta, phi).
"""
function cart2spher(x_cart::Vector)
    r = norm(x_cart)
    theta = acos(x_cart[3] / r)
    phi = atan(x_cart[2], x_cart[1])
    return [r, theta, phi]
end

"""
    spher2cart(x_spher::Vector)

Converts spherical coordinates to Cartesian coordinates.

# Arguments
- `x_spher::Vector`: A 3-element vector representing spherical coordinates (r, theta, phi).

# Returns
- A 3-element vector representing Cartesian coordinates (x, y, z).
"""
function spher2cart(x_spher::Vector)
    r, theta, phi = x_spher
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return [x, y, z]
end

"""
    angle_diff(theta1, theta2)

Calculates the signed difference between two angles.

# Arguments
- `theta1`: First angle in radians.
- `theta2`: Second angle in radians.

# Returns
- The signed difference between the two angles, in the range (-π, π].
"""
function angle_diff(theta1, theta2)
    diff = mod(theta1 - theta2 + π, 2π) - π
    return diff
end


"""
    fit_multivariate_normals(Ps::Vector{Matrix{Float64}})

Fits multivariate normal distributions to a set of input matrices.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices, each representing a set of data points.

# Returns
- A vector of fitted `MvNormal` objects.
"""
function fit_multivariate_normals(Ps::Vector{Matrix{Float64}})
    mvns = Vector{MvNormal}(undef, length(Ps))

    for i in 1:length(Ps)
        P_i = Ps[i]
        μ_i = mean(P_i, dims=1)[:]
        Σ_i = cov(P_i)
        mvn_i = MvNormal(μ_i, Σ_i)
        mvns[i] = mvn_i
    end

    return mvns
end

"""
    bic_multivariate_normals(Ps::Vector{Matrix{Float64}}, mvns::Vector{MvNormal})

Computes the Bayesian Information Criterion (BIC) for each fitted multivariate normal distribution.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices, each representing a set of data points.
- `mvns::Vector{MvNormal}`: A vector of fitted `MvNormal` objects.

# Returns
- A vector of BIC values for the corresponding fitted `MvNormal` objects.
"""
function bic_multivariate_normals(Ps::Vector{Matrix{Float64}}, mvns::Vector{MvNormal})
    bics = Vector{Float64}(undef, length(Ps))

    for i in 1:length(Ps)
        P_i = Ps[i]
        n = size(P_i, 1)
        k = size(P_i, 2)
        log_likelihood_i = sum(logpdf.(mvns[i], eachrow(P_i)))
        bic_i = -2 * log_likelihood_i + k * log(n)
        bics[i] = bic_i
    end

    return bics
end

"""
    get_Ps(fit_results, matches, θh_pos_is_ventral; idx_use=[1,2,3,4,7], datasets_use=nothing, rngs_use=nothing, max_rng=2)

Return a vector of matrices containing the parameter sets for given `matches` and conditions.

# Arguments
- `fit_results`: A dictionary containing the CePNEM fit results for each dataset.
- `matches`: A list of tuples corresponding to a given neuron, with dataset and the corresponding index in the traces.
- `θh_pos_is_ventral`: A dictionary mapping dataset to a boolean indicating if positive θh corresponds to ventral head curvature (as opposed to dorsal).
- `idx_use`: An optional array of indices of the parameters to be used. Default is `[1,2,3,4,7]`.
- `datasets_use`: An optional list of dataset indices to be used. If not provided, all datasets in `matches` are used.
- `rngs_use`: An optional dictionary mapping dataset indices to lists of range indices to be used. If not provided, all ranges are used.
- `max_rng`: An optional integer specifying the maximum number of ranges to be used. Default is `2`.

# Returns
- `Ps`: A vector of matrices containing the parameter sets for the given `matches` and conditions.
"""
function get_Ps(fit_results, matches, θh_pos_is_ventral; idx_use=[1,2,3,4,7], datasets_use=nothing, rngs_use=nothing, max_rng=2)
    Ps = Vector{Matrix{Float64}}(undef, max_rng*length(matches))
    count = 1
    for (dataset, n) in matches
        if !isnothing(datasets_use) && !(dataset in datasets_use)
            continue
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            if !isnothing(rngs_use) && !(rng in rngs_use[dataset])
                continue
            end
            Ps[count] = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,n,:,:])
            Ps[count][:,3] = Ps[count][:,3] .* (1 - 2*θh_pos_is_ventral[dataset])
            Ps[count] = Ps[count][:,idx_use]
            count += 1
        end
    end
    return Ps[1:count-1]
end

"""
    get_corrected_r(analysis_dict::Dict, mu::Vector{Float64})

Compute the corrected r value based on extrapolating the given mu array to the set of extrapolated behaviors.

# Arguments
- `analysis_dict::Dict`: Dictionary containing the `extrapolated_behaviors` array.
- `mu::Vector{Float64}`: An array containing the mu values of the parameters.

# Returns
- `corrected_r`: The computed corrected r value.
"""
function get_corrected_r(analysis_dict, mu)
    extrapolated_behaviors = analysis_dict["extrapolated_behaviors"]

    get_corrected_r(extrapolated_behaviors, mu)
end

function get_corrected_r(extrapolated_behaviors, mu)
    mu_cart = [mu[1], spher2cart(exp_r(mu[2:4]))..., 0, 0, mu[5], 0]
    extrap = model_nl8(length(extrapolated_behaviors[:,1]), mu_cart..., extrapolated_behaviors[:,1], extrapolated_behaviors[:,2], extrapolated_behaviors[:,3])

    return std(extrap)
end

"""
    params_to_spher(params::Vector{Float64}) -> Vector{Float64}

Convert a vector of parameters from Cartesian to spherical coordinates. This function expects
a vector of 5 parameters, where the first element is `c_vT`, the next three elements represent
`c_v`, `c_θh`, and `c_P` in Cartesian coordinates, and the last element is `s0`. The output will
have the same structure, but with the middle three elements in spherical coordinates.

# Arguments
- `params::Vector{Float64}`: A vector of 5 parameters with the structure [c_vT, c_v, c_θh, c_P, s0].

# Returns
- `Vector{Float64}`: A vector of 5 parameters with the structure [c_vT, log(r), θ, φ, s0] in spherical coordinates.
"""
function params_to_spher(params::Vector{Float64})
    log_r = x->[log(x[1]), x[2], x[3]]
    return [params[1], log_r(cart2spher(params[2:4]))..., params[5]]
end

"""
    exp_r(vec)

Compute the exponential of the first element of the input vector and concatenate it with the rest of the vector.

# Arguments
- `vec`: A vector of values.

# Returns
- `exp_r_vec`: A new vector with the first element exponentiated and the rest unchanged.
"""
function exp_r(vec)
    return [exp(vec[1]), vec[2:end]...]
end


"""
    get_datasets(matches; datasets_use=nothing, rngs_use=nothing)

Retrieve datasets based on the given matches and optional filtering criteria.

# Arguments
- `fit_results::Dict`: A dictionary containing the CePNEM fit results for each dataset.
- `matches::Vector`: A list of tuples containing the dataset identifier and the number of samples.
- `datasets_use::Union{Vector{String}, Nothing}`: An optional list of dataset identifiers to include. If not provided, all datasets in `matches` will be used.
- `rngs_use::Union{Dict, Nothing}`: An optional dictionary of dataset identifiers with a list of range indices to include. If not provided, all ranges will be used.

# Returns
- `datasets`: A list of tuples containing the selected dataset identifier, range index, and number of samples.
"""
function get_datasets(fit_results::Dict, matches::Vector; datasets_use::Union{Vector{String}, Nothing}=nothing, rngs_use::Union{Dict, Nothing}=nothing)
    datasets = []
    for (dataset, n) in matches
        if !isnothing(datasets_use) && !(dataset in datasets_use)
            continue
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            if !isnothing(rngs_use) && !(rng in rngs_use[dataset])
                continue
            end
            push!(datasets, (dataset, rng, n))
        end
    end
    return datasets
end

"""
    convert_hbparams_to_ps(hbparams::Vector{Float64})

Convert the given hbparams 5-vector of `mu` or `x` from `HBParams` into a new `ps`` vector with 8 elements compatible with `model_nl8`.

# Arguments
- `hbparams::Vector{Float64}`: A vector of hierarchical model parameters.

# Returns
- `ps`: A new 8-element vector containing the converted hbparams.
"""
function convert_hbparams_to_ps(hbparams::Vector{Float64})
    ps = zeros(8)
    ps[1] = hbparams[1] # c_vT
    ps[2:4] = spher2cart(exp_r(hbparams[2:4])) # c_v, c_θh, c_P
    ps[3] = -ps[3] # original model parameters have opposite sign for c_θh
    ps[5] = 0.0
    ps[6] = 0.0
    ps[7] = hbparams[5] # τ
    ps[8] = 0.0
    return deepcopy(ps)
end


"""
    convert_hbdatasets_to_behs(fit_results::Dict, hbdatasets::Vector; max_len::Int=800)

Convert the given hbdatasets into separate v, θh, and P arrays based on fit_results.

# Arguments
- `fit_results::Dict`: A dictionary containing the results of a model fitting process.
- `hbdatasets::Vector`: A list of tuples containing the selected dataset identifier, range index, and number of samples.
- `θh_pos_is_ventral::Dict{String, Bool}`: Whether the θh values in the datasets are in ventral or dorsal coordinates.
- `max_len::Int` (optional, default 800): The maximum length of each dataset.

# Returns
- `v`: A vector containing the concatenated v values from the datasets.
- `θh`: A vector containing the concatenated θh values from the datasets.
- `P`: A vector containing the concatenated P values from the datasets.
"""
function convert_hbdatasets_to_behs(fit_results::Dict, hbdatasets::Vector, θh_pos_is_ventral::Dict{String, Bool}; max_len::Int=800)
    v = zeros(max_len*length(hbdatasets))
    θh = zeros(max_len*length(hbdatasets))
    P = zeros(max_len*length(hbdatasets))
    count = 1
    for (dataset, rng, n) in hbdatasets
        rng_t = fit_results[dataset]["ranges"][rng]
        len = length(rng_t)
        v[count:count+len-1] .= fit_results[dataset]["v"][rng_t]
        θh[count:count+len-1] .= fit_results[dataset]["θh"][rng_t] .* (2*θh_pos_is_ventral[dataset] - 1)
        P[count:count+len-1] .= fit_results[dataset]["P"][rng_t]
        count += len
    end
    v = v[1:count-1]
    θh = θh[1:count-1]
    P = P[1:count-1]
    return v, θh, P
end

"""
    angle_mean(angles::Vector{Float64})

Compute the mean angle of the given angles vector.

# Arguments
- `angles::Vector{Float64}`: A vector of angles in radians.

# Returns
- `mean_angle`: The mean angle in radians.
"""
function angle_mean(angles::Vector{Float64})
    if isempty(angles)
        error("Input list cannot be empty")
    end
    
    x_sum = 0.0
    y_sum = 0.0

    for angle in angles
        x_sum += cos(angle)
        y_sum += sin(angle)
    end

    mean_x = x_sum / length(angles)
    mean_y = y_sum / length(angles)

    mean_angle = atan(mean_y, mean_x)

    if mean_angle < 0
        mean_angle += 2 * pi
    end

    return mean_angle
end

"""
    angle_mean(matrix::Array{Float64, 2}, dim::Int)

Compute the mean angle along the specified dimension of the given matrix.

# Arguments
- `matrix::Array{Float64, 2}`: A 2D array containing angles in radians.
- `dim::Int`: The dimension along which to compute the mean angle (1 for row-wise or 2 for column-wise).

# Returns
- `result`: A vector containing the mean angles computed along the specified dimension.
"""
function angle_mean(matrix::Array{Float64, 2}, dim::Int)
    if isempty(matrix)
        error("Input matrix cannot be empty")
    end
    
    if dim < 1 || dim > 2
        error("Invalid dimension. The dimension must be 1 or 2")
    end

    if dim == 1
        result_size = size(matrix, 2)
    else
        result_size = size(matrix, 1)
    end
    
    result = zeros(Float64, result_size)

    for i in 1:result_size
        if dim == 1
            angles = matrix[:, i]
        else
            angles = matrix[i, :]
        end

        x_sum = 0.0
        y_sum = 0.0

        for angle in angles
            x_sum += cos(angle)
            y_sum += sin(angle)
        end

        mean_x = x_sum / length(angles)
        mean_y = y_sum / length(angles)

        mean_angle = atan(mean_y, mean_x)

        result[i] = mean_angle
    end

    return result
end


"""
    angle_std(angles::Vector{Float64})

Compute the standard deviation of the given angles vector.

# Arguments
- `angles::Vector{Float64}`: A vector of angles in radians.

# Returns
- `std_deviation`: The standard deviation of the given angles, in radians.
"""
function angle_std(angles::Vector{Float64})
    if isempty(angles)
        error("Input list cannot be empty")
    end

    n = length(angles)
    
    if n == 1
        return 0.0
    end

    x_sum = 0.0
    y_sum = 0.0

    for angle in angles
        x_sum += cos(angle)
        y_sum += sin(angle)
    end

    mean_x = x_sum / n
    mean_y = y_sum / n

    sum_of_squared_differences = 0.0

    for angle in angles
        x_diff = cos(angle) - mean_x
        y_diff = sin(angle) - mean_y
        sum_of_squared_differences += x_diff^2 + y_diff^2
    end

    variance = sum_of_squared_differences / (n - 1)
    std_deviation = sqrt(variance)

    return std_deviation
end

function hbparams_to_cart(params)
    return [params[1], spher2cart(exp_r(params[2:4]))..., params[5]]
end


# Function to compute the mean direction of data points
function mean_direction(data)
    n = length(data)
    mean_dir = sum(data, dims=1) ./ n
    return mean_dir ./ norm(mean_dir)
end

# Function to estimate kappa from data points and their mean direction
function estimate_kappa(data, mean_dir)
    R_bar = mean([dot(mean_dir, x) for x in data])

    if R_bar < 0.9
        kappa = R_bar * (2 - R_bar) / (1 - R_bar^2)
    else
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    end

    return kappa
end

# Function to fit a vMF distribution to the data points
function fit_vmf(data)
    mean_dir = mean_direction(data)
    kappa = estimate_kappa(data, mean_dir)
    return VonMisesFisher(mean_dir, kappa)
end

"""
compute_cartesian_average(hbparams::HBParams)

Compute the Cartesian average of the given hierarchical Bayesian parameters.

# Arguments
- `hbparams::HBParams`: A hierarchical Bayesian parameter object.

# Returns
- `ps_tot::Vector{Float64}`: The Cartesian average of the given hierarchical Bayesian parameters.
"""
function compute_cartesian_average(hbparams::HBParams)
    ps = zeros(length(hbdatasets), 8)

    for i in 1:length(hbparams.x)
        ps[i,:] .= convert_hbparams_to_ps(hbparams.x[i])
    end
    ps_tot = deepcopy(convert_hbparams_to_ps(hbparams.mu))
    ps_tot[2:4] .= mean(ps, dims=1)[1,2:4] # use HB model estimates directly for parameters other than spherical ones
    # use individual dataset means for spherical parameters in case of bimodal distributions

    return ps_tot
end
