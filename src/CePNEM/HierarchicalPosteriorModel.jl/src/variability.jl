"""
    kappa_to_circular_sd(kappa::Float64)

Converts the concentration parameter `kappa` of a von Mises distribution to the circular standard deviation.

# Arguments
- `kappa::Float64`: The concentration parameter of the von Mises distribution.

# Returns
- `circular_sd::Float64`: The circular standard deviation.

"""
function kappa_to_circular_sd(kappa::Float64)
    R_bar = besseli(1, kappa) / besseli(0, kappa)
    circular_sd = sqrt(2 * (1 - R_bar))
    return circular_sd
end


"""
    normalize_cart_xi(cart_xi::Vector{Float64})

Normalizes a 1x5 Cartesian parameters vector.

# Arguments
- `cart_xi::Vector{Float64}`: A 1x5 Cartesian parameters vector.

# Returns
- `normalized_cart_xi::Vector{Float64}`: A 1x5 Cartesian parameter vector with the second through fourth parameters normalized.
"""
function normalize_cart_xi(cart_xi::Vector{Float64})
    return [cart_xi[1], cart_xi[2:4] ./ norm(cart_xi[2:4])..., cart_xi[5]]
end

"""
    normalized_cartesian_distance(xi_cart_1::Array{Float64}, xi_cart_2::Array{Float64})

Computes the normalized Cartesian distance between two Cartesian parameteters in 3D space.

# Arguments
- `xi_cart_1::Array{Float64}`: A 1x5 array representing the first point in Cartesian parameters.
- `xi_cart_2::Array{Float64}`: A 1x5 array representing the second point in Cartesian parameters.

# Returns
- `distance::Array{Float64}`: A 1x3 array representing the normalized Cartesian distance between the two points.
    The first element is the absolute difference between the first parameters,
    the second element is the normalized Euclidean distance between the second through fourth parameters, and the third element is the absolute difference between the fifth parameters.
"""
function normalized_cartesian_distance(xi_cart_1::Array{Float64}, xi_cart_2::Array{Float64})
    return [abs(xi_cart_1[1] - xi_cart_2[1]), sqrt(sum((xi_cart_1[2:4] .- xi_cart_2[2:4]).^2)) / (norm(xi_cart_1[2:4]) / 2 + norm(xi_cart_2[2:4]) / 2), abs(xi_cart_1[5] - xi_cart_2[5])]
end


"""
    get_variability(sigma::Array{Float64})

Compute the variability of the parameters based on the given `sigma` array from a hierarchical posterior model fit.

# Arguments
- `sigma::Array{Float64}`: An array containing the `sigma` values of the parameters.

# Returns
- `variability`: The computed variability of the parameters.
"""
function get_variability(sigma::Array{Float64})
    exp_sigma = exp.(sigma) .+ 2e-4

    # use inverse sqrt approximation for large kappa to prevent overflow
    return exp_sigma[1] + (exp_sigma[3] > 400 ? 1/sqrt(exp_sigma[3]) : kappa_to_circular_sd(exp_sigma[3]))
end

"""
    get_variability(model_params::HBParams)

Compute the variability metric based on the provided model parameters.

# Arguments
- `model_params`: A `HBParams` instance containing the standard deviation (sigma) parameter.

# Returns
- `variability`: A scalar value representing the computed variability metric.
"""
function get_variability(model_params::HBParams)
    sigma = model_params.sigma

    return get_variability(sigma)
end

"""
    average_xis(xis::Matrix{Float64})

Computes the average of the given `xis` matrix of parameter values.

# Arguments
- `xis::Matrix{Float64}`: A matrix of size (`n \times 5`) containing the Cartesian parameters of `n` points.

# Returns
- `spher_mean::Vector{Float64}`: A 1x3 vector representing the mean of the given `xis` matrix in spherical coordinates.
- `cart_mean::Vector{Float64}`: A 1x5 vector representing the mean of the given `xis` matrix in Cartesian coordinates.
"""
function average_xis(xis::Matrix{Float64})
    n = size(xis, 1)
    xis_cart = zeros(size(xis)...)
    for i=1:n
        xis_cart[i, :] .= hbparams_to_cart(xis[i, :])
    end
    mean_cartesian_xi = mean(xis_cart, dims=1)[1,:]

    return params_to_spher(mean_cartesian_xi), mean_cartesian_xi
end


"""
    get_variability_subtypes(hierarchical_datasets::Vector, hierarchical_params::HBParams; neuron::String="")

Calculate the variability of parameters between datasets for different subtypes of variability:
inter-dataset variability, intra-dataset variability, and left vs right (LR) variability.

# Arguments
- `hierarchical_datasets::Vector`: A vector containing the datasets used to fit the hierarchical model.
- `hierarchical_params::HBParams`: An instance of HBParams containing the hierarchical model parameters.
- `neuron::String`: (optional) The name of the neuron (default: empty string).

# Returns
- `inter_variability`: The calculated inter-dataset variability.
- `intra_variability`: The calculated intra-dataset variability.
- `LR_variability`: The calculated left-right variability.
- `inter_dataset_variability`: A list of values from which inter-dataset variability is computed.
- `intra_dataset_variability`: A list of values from which intra-dataset variability is computed.
- `LR_dataset_variability`: A list of values from which left-right variability is computed.
"""
function get_variability_subtypes(hierarchical_datasets::Vector, hierarchical_params::HBParams; neuron::String="")
    intra_dataset_variability = []
    inter_dataset_variability = []
    LR_dataset_variability = []

    sigma_diff = (sigma1, sigma2) -> [sigma1[1] - sigma2[1], sigma1[2] - sigma2[2], angle_diff(sigma1[3], sigma2[3]), angle_diff(sigma1[4], sigma2[4]), sigma1[5] - sigma2[5]]
    for dataset in unique([x[1] for x in hierarchical_datasets])
        all_obs = [(i,x...) for (i,x) in enumerate(hierarchical_datasets) if x[1] == dataset]
        x_dataset = zeros(length(all_obs), length(hierarchical_params.x[1]))
        for i=1:length(all_obs)
            x_dataset[i,:] .= hierarchical_params.x[all_obs[i][1]]
        end
        avg_dataset = normalize_cart_xi(average_xis(x_dataset)[2]) # mean over all observations in the dataset; note that this will be log(r) not r
        push!(inter_dataset_variability, avg_dataset)
        datasets_rng_1 = [x for x in all_obs if x[3] == 1]
        datasets_rng_2 = [x for x in all_obs if x[3] == 2]
        if length(datasets_rng_1) > 0 && length(datasets_rng_2) > 0
            x_dataset_rng_1 = zeros(length(datasets_rng_1), length(hierarchical_params.x[1]))
            x_dataset_rng_2 = zeros(length(datasets_rng_2), length(hierarchical_params.x[1]))
            for i=1:length(datasets_rng_1)
                x_dataset_rng_1[i,:] .= hierarchical_params.x[datasets_rng_1[i][1]]
            end
            for i=1:length(datasets_rng_2)
                x_dataset_rng_2[i,:] .= hierarchical_params.x[datasets_rng_2[i][1]]
            end
            avg_dataset_rng_1 = average_xis(x_dataset_rng_1)[2]
            avg_dataset_rng_2 = average_xis(x_dataset_rng_2)[2]
            push!(intra_dataset_variability, normalized_cartesian_distance(avg_dataset_rng_1, avg_dataset_rng_2) ./ sqrt(2))
        end

        neuron_obs = unique([x[4] for x in all_obs])

        if length(neuron_obs) > 2
            @warn("Cannot have more than 2 detections of the same neuron $neuron in a dataset $dataset")
            continue
        end
        if length(neuron_obs) == 1
            continue
        end
        obs_1 = [x for x in all_obs if x[4] == neuron_obs[1]]
        obs_2 = [x for x in all_obs if x[4] == neuron_obs[2]]

        x_dataset_obs_1 = zeros(length(obs_1), length(hierarchical_params.x[1]))
        x_dataset_obs_2 = zeros(length(obs_2), length(hierarchical_params.x[1]))
        for i=1:length(obs_1)
            x_dataset_obs_1[i,:] .= hierarchical_params.x[obs_1[i][1]]
        end
        for i=1:length(obs_2)
            x_dataset_obs_2[i,:] .= hierarchical_params.x[obs_2[i][1]]
        end
        avg_dataset_obs_1 = average_xis(x_dataset_obs_1)[2]
        avg_dataset_obs_2 = average_xis(x_dataset_obs_2)[2]
        push!(LR_dataset_variability, normalized_cartesian_distance(avg_dataset_obs_1, avg_dataset_obs_2) ./ sqrt(2))
    end

    n_params = length(hierarchical_params.x[1])

    inter_len = length(inter_dataset_variability)

    inter_data_spher = [inter_dataset_variability[i][2:4] for i=1:inter_len]

    inter_sigma = (length(inter_dataset_variability) > 1) ? [std([inter_dataset_variability[i][1] for i=1:inter_len]), 1.0, estimate_kappa(inter_data_spher, mean_direction(inter_data_spher)[1]), 1.0, std([inter_dataset_variability[i][5] for i=1:inter_len])] : fill(NaN, n_params)
    sigma_to_mean = sigma -> (length(sigma) > 1) ? [mean([sigma[i][j] for i=1:length(sigma)]) for j=1:length(sigma[1])] : fill(NaN, 3)

    intra_sigma = (length(intra_dataset_variability) >= 1) ? sigma_to_mean(intra_dataset_variability) : fill(NaN, n_params)
    LR_sigma = (length(LR_dataset_variability) >= 1) ? sigma_to_mean(LR_dataset_variability) : fill(NaN, n_params)

    return get_variability(log.(inter_sigma)), intra_sigma[1] + intra_sigma[2], LR_sigma[1] + LR_sigma[2], inter_dataset_variability, intra_dataset_variability, LR_dataset_variability
end
