"""
    fit_state_classifier(x::Matrix{Float64}, t_stim::Int)

Fit a logistic regression model to classify whether a set of neurons contains "state" information about the indicator function of `t_stim`,
    based on neural data `x` and the time `t_stim` at which the hypothesized state change occurred. Returns the trained model and its accuracy.

# Arguments
- `x::Matrix{Float64}`: Input data matrix of size `(n_features, n_timepoints)`.
- `t_stim::Int`: Time of the hypothesized state change.

# Returns
- `model`: Trained logistic regression model.
- `accuracy`: Accuracy of the trained model.
"""
function fit_state_classifier(x::Matrix{Float64}, t_stim::Int)
    y = zeros(Int64, size(x,2))
    y[1:t_stim] .= 0
    y[t_stim+1:end] .= 1

    w = zeros(Float64, size(x,2))
    w[1:t_stim] .= length(w) - t_stim
    w[t_stim+1:end] .= t_stim

    w = w ./ mean(w)

    # Convert the input data and target vector to a DataFrame

    # if size(x, 1) > 1
    #     println(DataFrame(convert(Matrix{Float64}, transpose(x)), :auto))
    # end
    df = hcat(DataFrame(y=y, w=w), DataFrame(convert(Matrix{Float64}, transpose(x)), :auto))

    # Create the logistic regression model
    model = glm(term(:y) ~ sum(term(t) for t in names(df) if (t != "y") && (t != "w")), df, Binomial(), LogitLink(), wts=df.w)

    # Use the model to make predictions for new games
    predictions = GLM.predict(model, df)

    # Converting probability score to classes
    prediction_class = [if x < 0.5 0 else 1 end for x in predictions];

    accuracy_0 = mean([prediction_class[i] == df.y[i] for i in 1:length(df.y) if df.y[i] == 0])
    accuracy_1 = mean([prediction_class[i] == df.y[i] for i in 1:length(df.y) if df.y[i] == 1])
    accuracy = mean([accuracy_0, accuracy_1])

    return (model, accuracy)
end


"""
    compute_state_neuron_candidates(
        analysis_dict::Dict, fit_results::Dict, datasets::Vector{String}, strategy::String,
        stim_times::Dict; stim_try=200:50:1400, delta_t::Int=190
    )::Dict

Computes the state neuron candidates for each dataset in `datasets`. They are found by fitting a logistic regression model to the neural data and
determining whether the model can predict the state change better than it can predict control time points.

The strategy for selecting the neurons is determined by the `strategy` argument, which can be one of "increase", "decrease", or "findall":
- "increase": Start from 0 neurons and add state changing neurons until performance no longer improves
- "decrease": Start from all neurons and remove non-state changing neurons until performance starts to degrade
- "findall": Find all neurons with positive performance

The `stim_try` argument is a range of timepoints to try for the controls.

# Arguments
- `analysis_dict::Dict`: A dictionary containing the CePNEM analysis results.
- `fit_results::Dict`: A dictionary containing the CePNEM fit results.
- `datasets::Vector{String}`: A vector of datasets to use. They must all be heat-stimulation datasets.
- `strategy::String`: The strategy for selecting the neurons.
- `stim_times::Dict`: A dictionary containing the heat stimulus times for each dataset.
- `stim_try`: A range of timepoints to try for the state change controls.
- `delta_t::Int`: The minimum distance between the state change and control timepoints. (Control points closer than this to the real stim are excluded.)

# Returns
- `state_neuron_candidates::Dict`: A dictionary containing the state neuron candidates for each dataset.
"""
function compute_state_neuron_candidates(analysis_dict::Dict, fit_results::Dict, datasets::Vector{String}, strategy::String,
        stim_times::Dict; stim_try=200:50:1400, delta_t::Int=190)::Dict
    
    @assert(strategy in ["increase", "decrease", "findall"], "strategy must be one of increase, decrease, findall")

    state_neuron_candidates = Dict()

    @showprogress for dataset = datasets
        t_stim = stim_times[dataset]        
        
        stim_accuracy = zeros(Int32, length(stim_try))

        curr_stim_accuracy = 0
        curr_rand_accuracy = 0
        stop = false
        if strategy == "increase"
            curr_neurons = Int32[]
        elseif strategy == "decrease"
            curr_neurons = 1:size(fit_results[dataset]["trace_array"],1)
        elseif strategy == "findall"
            curr_neurons = Int32[]
        end

        relative_accuracy = zeros(Float64, size(fit_results[dataset]["trace_array"],1))
        acc = Dict()

        while !stop
            stop = true
            next_neurons = curr_neurons

            for neuron in 1:size(fit_results[dataset]["trace_array"],1)
                if (neuron in curr_neurons) ⊻ (strategy == "decrease")
                    continue
                end

                curr_neurons_test = copy(curr_neurons)
                if strategy == "increase"
                    push!(curr_neurons_test, neuron)
                elseif strategy == "decrease"
                    curr_neurons_test = [i for i in curr_neurons_test if i != neuron]
                elseif strategy == "findall"
                    curr_neurons_test = [neuron]
                end

                X = fit_results[dataset]["trace_array"][curr_neurons_test,:]

                accuracy_vals = Float64[]

                accuracy_true = 0
                try
                    _, accuracy_true = fit_state_classifier(X, t_stim)
                catch e
                    @warn("Dataset $dataset, neuron $neuron: $e")
                    continue
                end
                
                for fake_stim = stim_try
                    if abs(fake_stim - t_stim) < delta_t
                        continue
                    end

                    accuracy_fake = 0
                    try
                        _, accuracy_fake = fit_state_classifier(X, fake_stim)
                    catch e
                        @warn("Dataset $dataset, neuron $neuron: $e")
                        continue
                    end

                    push!(accuracy_vals, accuracy_fake)
                end

                if strategy == "findall"
                    relative_accuracy[neuron] = accuracy_true - maximum(accuracy_vals)
                    acc[neuron] = (relative_accuracy[neuron], accuracy_true, accuracy_vals)
                elseif accuracy_true - mean(accuracy_vals) > curr_stim_accuracy - curr_rand_accuracy || (strategy == "decrease" && (accuracy_true - mean(accuracy_vals) == curr_stim_accuracy - curr_rand_accuracy))
                    curr_stim_accuracy = accuracy_true
                    curr_rand_accuracy = mean(accuracy_vals)
                    next_neurons = curr_neurons_test
                    if strategy == "decrease" # to avoid excessive computation power, immediately delete any neurons that degrade performance
                        curr_neurons = next_neurons
                    end
                    stop = false
                end
            end
            curr_neurons = next_neurons
        end
        if strategy == "findall"
            state_neuron_candidates[dataset] = acc
        else
            state_neuron_candidates[dataset] = curr_neurons
        end
    end
    return state_neuron_candidates
end
