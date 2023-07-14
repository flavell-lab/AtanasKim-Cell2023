"""
    gaussian_kernel(x::Real; sigma::Float64=1.0)

Evaluates a Gaussian kernel at position `x`` with standard deviation `sigma`.
"""
function gaussian_kernel(x::Real; sigma::Float64=1.0)
    return exp(-x^2 / (2 * sigma^2)) / sqrt(2 * pi * sigma^2)
end

"""
    convolve(x::Vector{Float64}, k::Vector{Float64})

Convolves a vector `x` with a kernel vector `k`.
"""
function convolve(x::Vector{Float64}, k::Vector{Float64})
    n = length(x)
    m = length(k)
    y = zeros(n + m - 1)
    for i in 1:n
        for j in 1:m
            y[i + j - 1] += x[i] * k[j]
        end
    end
    return y
end

"""
    correct_name(neuron_name)

Corrects the name of a neuron by removing the "0" substring from the name if it contains "DB0" or "VB0".
This causes the neuron name to be compatible with the connectome files.

# Arguments:
- `neuron_name::String`: The name of the neuron to be corrected.

# Returns:
- `neuron_name::String`: The corrected name of the neuron.
"""
function correct_name(neuron_name::String)
    if occursin("DB0", neuron_name)
        neuron_name = neuron_name[1:2]*neuron_name[4:end]
    end
    if occursin("VB0", neuron_name)
        neuron_name = neuron_name[1:2]*neuron_name[4:end]
    end
    return neuron_name
end

"""
    find_peaks(neural_activity::Vector{Float64}, threshold::Float64)

Find the peaks in a vector of neural activity above a given threshold.
This method is most useful for finding spikes in the activity of spiking neurons.

# Arguments:
- `neural_activity::Vector{Float64}`: A vector of neural activity.
- `threshold::Float64`: The threshold above which to consider a value a peak.

# Returns:
- `peaks::Vector{Int}`: A vector of indices of the peaks in the neural activity.
- `peak_heights::Vector{Float64}`: A vector of the heights of the peaks in the neural activity.
"""
function find_peaks(neural_activity::Vector{Float64}, threshold::Float64)
    peaks = Int[]
    peak_heights = Float64[]
    over_threshold = false
    curr_peak = (-1, -Inf)
    for i in 1:length(neural_activity)
        if neural_activity[i] > threshold && !over_threshold
            over_threshold = true
            curr_peak = (i, neural_activity[i])
        elseif neural_activity[i] < threshold && over_threshold
            over_threshold = false
            push!(peaks, curr_peak[1])
            push!(peak_heights, curr_peak[2])
            curr_peak = (-1, -Inf)
        elseif neural_activity[i] > curr_peak[2] && over_threshold
            curr_peak = (i, neural_activity[i])
        end
    end
    return peaks, peak_heights
end

"""
    parse_tuning_strength(tuning_strength::Dict{String, Any}, beh::String, encoding::String)

Parses the tuning strength of a neuron for a given behavior and encoding.

# Arguments:
- `tuning_strength::Dict{String, Any}`: A dictionary containing the tuning strength of a neuron.
- `beh::String`: The behavior for which to find the tuning strength.
- `encoding::String`: The encoding for which to find the tuning strength.

# Returns:
- `tuning_strength::Any`: The tuning strength of the neuron for the given behavior encoding.
"""
function parse_tuning_strength(tuning_strength::Dict{String, Any}, beh::String, encoding::String)
    if beh == "v"
        if encoding in ["rev_slope_neg", "rev_slope_pos"]
            return tuning_strength["v_encoding"][1,2,1,1]
        elseif encoding in ["fwd_slope_neg", "fwd_slope_pos"]
            return tuning_strength["v_encoding"][3,4,1,1]
        elseif encoding in ["fwd", "rev"]
            return tuning_strength["v_fwd"]
        elseif encoding in ["rect_neg", "rect_pos"]
            return tuning_strength["v_rect_pos"]
        end
    elseif beh == "θh"
        if encoding in ["dorsal", "ventral"]
            return tuning_strength["θh_dorsal"]
        elseif encoding in ["fwd_ventral", "fwd_dorsal"]
            return tuning_strength["fwd_θh_encoding_dorsal"]
        elseif encoding in ["rev_ventral", "rev_dorsal"]
            return tuning_strength["rev_θh_encoding_dorsal"]
        elseif encoding in ["rect_dorsal", "rect_ventral"]
            return tuning_strength["θh_rect_dorsal"]
        end
    end
    return tuning_strength
end

"""
    get_all_neurons_with_feature(fit_results::Dict{String, Any}, analysis_dict::Dict{String, Any}, beh::String, sub_behs::Union{Nothing, Vector{String}}, p::Float64=0.05)

Get all neurons that encode a given behavior or sub-behavior feature.

# Arguments:
- `fit_results::Dict{String, Any}`: A dictionary containing the results of fitting CePNEM to neural data.
- `analysis_dict::Dict{String, Any}`: A dictionary containing the results of analyzing the neural data.
- `beh::String`: Find neurons that encode this behavior and the corresponding sub-behavior.
- `sub_behs::Union{Nothing, Vector{String}}`: Find neurons that encode this sub-behavior. If `nothing`, only encodings of the main behavior are considered.
- `p::Float64=0.05`: The p-value threshold for considering a neuron to have the given feature.

# Returns:
- `neurons_use::Dict{String, Vector{Int32}}`: A dictionary containing the indices of the neurons with the given feature for each dataset.
"""
function get_all_neurons_with_feature(fit_results::Dict{String, Any}, analysis_dict::Dict{String, Any}, beh::String, sub_behs::Union{Nothing, Vector{String}}, p::Float64=0.05)
    traces_use = Dict()
    neurons_use = Dict()
    for dataset in keys(fit_results)
        neurons_use[dataset] = Int32[]

        for neuron in 1:size(fit_results[dataset]["trace_array"],1)
            use = false
            if isnothing(sub_behs)
                pval = minimum(adjust([analysis_dict["neuron_p"][dataset][rng][beh][neuron] for rng = 1:length(fit_results[dataset]["ranges"])], BenjaminiHochberg()))
                if pval < p
                    use = true
                end
            else
                for sub_beh in sub_behs
                    pval = length(sub_behs)*minimum(adjust([analysis_dict["neuron_p"][dataset][rng][beh][sub_beh][neuron] for rng = 1:length(fit_results[dataset]["ranges"])], BenjaminiHochberg()))
                    if pval < p
                        use = true
                        break
                    end
                end
            end
            if use
                push!(neurons_use[dataset], neuron)
            end
        end

        traces_use[dataset] = fit_results[dataset]["trace_array"][neurons_use[dataset], :]
    end
    return neurons_use, traces_use
end

"""
    get_random_sample_without_feature(fit_results::Dict{String, Any}, analysis_dict::Dict{String, Any}, neurons_with_feature::Dict{String, Vector{Int32}})

Get a random sample of neurons that do not encode a given feature.

# Arguments:
- `fit_results::Dict{String, Any}`: A dictionary containing the results of fitting CePNEM to neural data.
- `analysis_dict::Dict{String, Any}`: A dictionary containing the results of analyzing the neural data.
- `neurons_with_feature::Dict{String, Vector{Int32}}`: A dictionary containing the indices of the neurons that encode the feature for each dataset.

# Returns:
- `neurons_use::Dict{String, Vector{Int32}}`: A dictionary containing the indices of the neurons without the given feature for each dataset.
- `traces_use::Dict{String, Array{Float64, 2}}`: A dictionary containing the traces of the neurons without the given feature for each dataset.
"""
function get_random_sample_without_feature(fit_results::Dict{String, Any}, analysis_dict::Dict{String, Any}, neurons_with_feature::Dict{String, Vector{Int32}})
    traces_use = Dict{String, Array{Float64, 2}}()
    neurons_use = Dict{String, Vector{Int32}}()
    for dataset in keys(fit_results)
        neurons_available = [i for i in 1:size(fit_results[dataset]["trace_array"],1) if !(i in neurons_with_feature[dataset])]
        neurons_use[dataset] = sample(neurons_available, length(neurons_with_feature[dataset]), replace=false)
        traces_use[dataset] = fit_results[dataset]["trace_array"][neurons_use[dataset], :]
    end
    return neurons_use, traces_use
end

"""
    get_frac_responding(neurons::Vector{Int}, responses::Dict{String, Any}, response_keys::Vector{String}, decide_fn::Function; response_keys_2::Union{Nothing, Vector{String}}=nothing, decide_fn_2::Union{Nothing, Function}=nothing)

Calculate the fraction of neurons that respond to the heat stimulus in the given way.

# Arguments:
- `neurons::Vector{String}`: A vector of neurons to consider.
- `responses::Dict{String, Any}`: A dictionary containing the responses of the neurons to the stimulus.
- `response_keys::Vector{String}`: A vector of keys corresponding to the response types to consider in the `responses` dictionary.
- `decide_fn::Function`: A function that takes a response and returns the strength of that response.
- `response_keys_2::Union{Nothing, Vector{String}}=nothing`: A vector of keys to access a second set of responses in the `responses` dictionary. If `nothing`, only the first set of responses is used. If set, this is used as a denominator to normalize the responses.
- `decide_fn_2::Union{Nothing, Function}=nothing`: A function that takes a response from the second set of responses and returns a boolean indicating whether the neuron responded. If `nothing`, all neurons are considered to have responded to the second set of responses.

# Returns:
- `frac_responding::Float64`: The fraction of neurons that responded to the stimulus.
"""
function get_frac_responding(neurons::Vector{String}, responses::Dict{String, Any}, response_keys::Vector{String}, decide_fn::Function; response_keys_2::Union{Nothing, Vector{String}}=nothing, decide_fn_2::Union{Nothing, Function}=nothing)
    responses_use = responses
    for k in response_keys
        responses_use = responses_use[k]
    end
    responses_use2 = responses
    if !isnothing(response_keys_2)
        for k in response_keys_2
            responses_use2 = responses_use2[k]
        end
    end
    n_responding = 0
    n_tot = 0
    for neuron in neurons
        for response in responses_use[neuron]
            n_responding += decide_fn(response)
            n_tot += (isnothing(decide_fn_2) ? 1 : decide_fn_2(responses_use2[neuron]))
        end
    end
    return (n_responding == 0) ? 0.0 : n_responding / n_tot
end

"""
    get_subcats(beh)

Get the subcategories of a given behavior.

# Arguments:
- `beh::String`: A string representing the behavior.

# Returns:
- `subcats::Vector{Tuple{String, String}}`: A vector of tuples representing the subcategories of the behavior.
"""
function get_subcats(beh)
    if beh == "v"
        subcats = [("rev_slope_neg", "rev_slope_pos"), ("rev", "fwd"), ("rect_neg", "rect_pos"), ("fwd_slope_neg", "fwd_slope_pos")]
    elseif beh == "θh"
        subcats = [("rev_dorsal", "rev_ventral"), ("dorsal", "ventral"), ("rect_dorsal", "rect_ventral"), ("fwd_dorsal", "fwd_ventral")]
    elseif beh == "P"
        subcats = [("rev_inh", "rev_act"), ("inh", "act"), ("rect_inh", "rect_act"), ("fwd_inh", "fwd_act")]
    end 
    return subcats
end
