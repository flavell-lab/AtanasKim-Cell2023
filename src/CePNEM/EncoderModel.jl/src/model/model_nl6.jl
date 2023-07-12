"""
    generate_model_nl6(xs_s, idx_splits)

model nl6 function
predictors: velocity, θh, pumping, ang vel, curvature

pumping threshold term is removed

Arguments
---------
* `xs_s`: standardized predictors array of `(n, t)`
* `idx_splits`: list of index range for time points splits.
e.g. `[1:800, 801:1600] for 2 videos merged with each 800 time points`
"""
function generate_model_nl6(xs_s, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps)
        ps[16] .+ ps[15] .* ewma((sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*

            (sin(ps[8]) .* x4 .+ cos(ps[8])) .*
            (sin(ps[9]) .* (1 .- 2 .* lesser.(x4, ps[10])) .+ cos(ps[9])) .*

            (sin(ps[11]) .* x5 .+ cos(ps[11])) .*
            (sin(ps[12]) .* (1 .- 2 .* lesser.(x5, ps[13])) .+ cos(ps[12])), ps[14], ewma_trim, idx_splits)
    end
end

"""
    generate_model_nl6_partial(xs_s, idx_splits, idx_valid)

model nl6 partial function
predictors: velocity, θh, pumping, ang vel, curvature

pumping threshold term is removed


Arguments
---------
* `xs_s`: standardized predictors array of `(n, t)`
* `idx_valid`: list of valid parameter index. e.g. model with pumping only should be [7,8,9,16,17,18]
* `ewma_trim`: amount to trim EWMA
* `idx_splits`: list of index range for time points splits.
e.g. `[1:800, 801:1600] for 2 videos merged with each 800 time points`
"""
function generate_model_nl6_partial(xs_s, idx_valid, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps_::AbstractVector{T}) where T
        ps = zeros(T, 16)
        ps[idx_valid] .= ps_
        ps[16] .+ ps[15] .* ewma((sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*

            (sin(ps[8]) .* x4 .+ cos(ps[8])) .*
            (sin(ps[9]) .* (1 .- 2 .* lesser.(x4, ps[10])) .+ cos(ps[9])) .*

            (sin(ps[11]) .* x5 .+ cos(ps[11])) .*
            (sin(ps[12]) .* (1 .- 2 .* lesser.(x5, ps[13])) .+ cos(ps[12])), ps[14], ewma_trim, idx_splits)
    end
end

"""
    init_ps_model_nl6(xs, idx_predictor)

initializes the model nl6 parameters
predictors: velocity, θh, pumping, ang vel, curvature

Arguments
---------
* `xs`: predictors array of `(n, t)` (not standardized)
* `idx_predictor`: list of index for predictors. for full model, [1,2,3,4,5]. model w/o velocity: [2,3,4,5]
"""
function init_ps_model_nl6(xs, idx_predictor=[1,2,3,4,5])
    n_xs = length(idx_predictor)
  
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = [[1,2,3], [4,5,6], [7], [8,9,10], [11,12,13]]
    list_idx_ps_reg = [[1,2], [4,5], [7], [8,9], [11,12]]

    for (i,b) = enumerate(idx_predictor)
        if b == 3
            push!(ps_0, [0.])
            push!(ps_min, [-pi/2])
            push!(ps_max, [pi/2])
        else
            θ = b == 5 ? 0 : -mean(xs[b,:])/std(xs[b,:])
            push!(ps_0, [0., 0., θ])
            push!(ps_min, [-pi/2, -pi/2, percentile(zstd(xs[b,:]), 5)])
            push!(ps_max, [pi/2, pi/2, percentile(zstd(xs[b,:]), 95)])
        end
    end
    ps_0 = vcat(ps_0..., [0.1, 1., 0.])
    ps_min = vcat(ps_min..., [0.03, -10., -10.])
    ps_max = vcat(ps_max..., [1., 10., 10.])
    
    
    list_idx_ps = vcat(list_idx_ps[idx_predictor]..., [14,15,16])
    list_idx_ps_reg = vcat(list_idx_ps_reg[idx_predictor]..., [15]) # 14: \gamma, 16: bias

    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

mutable struct ModelEncoderNL6 <: ModelEncoder
    xs
    xs_s
    ewma_trim::Int
    idx_splits::Union{Vector{UnitRange{Int64}}, Vector{Int64}}
    ps_0
    ps_min
    ps_max
    idx_predictor
    f
    list_idx_ps
    list_idx_ps_reg
    
    function ModelEncoderNL6(xs, xs_s, ewma_trim, idx_splits)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            [1,2,3,4,5], nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6)
    model.f = generate_model_nl6(model.xs_s, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL6)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6(model.xs,
        model.idx_predictor)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end
