"""
    get_tuning(activity, behavior, conv_fn; angle=nothing)

Gets the tuning of a neuron's activity pattern to a behavior.
Effectively, this function makes a scatterplot of activity vs behavior,
and for each value of behavior, computes what the typical neuron activity is by
doing a convolution with the neuron activity.

# Arguments:
- `activity`: Neuron activity traces
- `behavior`: Behavior in question
- `conv_fn`: Convolution function to apply to the behavior (eg: `x->pdf(Normal(0,pi/4),x)`)
- `angle` (optional, default `nothing`): If set, the behavior is in terms of an angle. In that case,
    the convolution function will be applied to the difference between the behavioral angle and the `angle` parameter.
"""
function get_tuning(activity, behavior, conv_fn; angle=nothing)
    if !isnothing(angle)
        tuning = median(activity, weights(conv_fn.(recenter_angle.(behavior .- angle))))
        uniform_tuning = median(activity)
    else
        tuning = median(activity, weights(conv_fn.(behavior)))
        uniform_tuning = median(activity)
    end
    return tuning - uniform_tuning
end
