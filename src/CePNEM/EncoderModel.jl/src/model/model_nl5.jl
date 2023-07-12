"""
    generate_model_nl5(xs_s, idx_splits)

model nl5 function
predictors: velocity, θh, pumping, ang vel, curcature

Arguments
---------
* `xs_s`: standardized predictors array of `(n, t)`
* `idx_splits`: list of index range for time points splits.
e.g. `[1:800, 801:1600] for 2 videos merged with each 800 time points`
"""
function generate_model_nl5(xs_s, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps)
        ps[18] .+ ps[17] .* ewma((sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*
            (sin(ps[8]) .* (1 .- 2 .* lesser.(x3, ps[9])) .+ cos(ps[8])) .*

            (sin(ps[10]) .* x4 .+ cos(ps[10])) .*
            (sin(ps[11]) .* (1 .- 2 .* lesser.(x4, ps[12])) .+ cos(ps[11])) .*

            (sin(ps[13]) .* x5 .+ cos(ps[13])) .*
            (sin(ps[14]) .* (1 .- 2 .* lesser.(x5, ps[15])) .+ cos(ps[14])),
                ps[16], ewma_trim, idx_splits)
    end
end

"""
    generate_model_nl5_partial(xs_s, idx_splits, idx_valid)

model nl5 partial function
predictors: velocity, θh, pumping, ang vel, curcature

Arguments
---------
* `xs_s`: standardized predictors array of `(n, t)`
* `idx_splits`: list of index range for time points splits.
e.g. `[1:800, 801:1600] for 2 videos merged with each 800 time points`
* `idx_valid`: list of valid parameter index. e.g. model with pumping only should be [7,8,9,16,17,18]
"""
function generate_model_nl5_partial(xs_s, idx_splits, idx_valid)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps_::AbstractVector{T}) where T
        ps = zeros(T, 18)
        ps[idx_valid] .= ps_
        ps[18] .+ ps[17] .* ewma(ps[16],
            (sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*
            (sin(ps[8]) .* (1 .- 2 .* lesser.(x3, ps[9])) .+ cos(ps[8])) .*

            (sin(ps[10]) .* x4 .+ cos(ps[10])) .*
            (sin(ps[11]) .* (1 .- 2 .* lesser.(x4, ps[12])) .+ cos(ps[11])) .*

            (sin(ps[13]) .* x5 .+ cos(ps[13])) .*
            (sin(ps[14]) .* (1 .- 2 .* lesser.(x5, ps[15])) .+ cos(ps[14])), idx_splits)
    end
end

"""
    init_ps_model_nl5(xs, idx_predictor)

initializes the model nl5 parameters
predictors: velocity, θh, pumping, ang vel, curcature

Arguments
---------
* `xs`: predictors array of `(n, t)` (not standardized)
* `idx_predictor`: list of index for predictors. for full model, [1,2,3,4,5]. model w/o velocity: [2,3,4,5]
"""
function init_ps_model_nl5(xs, idx_predictor=[1,2,3,4,5])
    n_xs = length(idx_predictor)
    
    ps_0 = vcat(repeat([0.0, 0.0, 0.0], n_xs), [0.1, 1., 0.])
    ps_min = vcat(repeat([-pi/2], n_xs * 3), [0.03, -10, -10])
    ps_max = vcat(repeat([pi/2], n_xs * 3), [1.0, 10., 10.])
    
    list_idx_ps_reg = []
    list_idx_ps = []
    for (i, b) = enumerate(idx_predictor)
        ps_min[3*i], ps_max[3*i] = percentile(zstd(xs[b,:]), [5,95])
        if b in [3,5] # pumping, ang_vel threshold
            ps_0[3*i] = 0. # mean before zstd
        else
            ps_0[3*i] = -mean(xs[b,:])/std(xs[b,:]) # 0 before zstd
        end
        push!(list_idx_ps, 3*b-2)
        push!(list_idx_ps, 3*b-1)
        push!(list_idx_ps, 3*b)

        push!(list_idx_ps_reg, 3*i-2)
        push!(list_idx_ps_reg, 3*i-1)
    end
    append!(list_idx_ps, [16,17,18])
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

mutable struct ModelEncoderNL5 <: ModelEncoder
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
    
    function ModelEncoderNL5(xs, xs_s, ewma_trim, idx_splits)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            [1,2,3,4,5], nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL5)
    model.f = generate_model_nl5(model.xs_s, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL5)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl5(model.xs,
        model.idx_predictor)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end