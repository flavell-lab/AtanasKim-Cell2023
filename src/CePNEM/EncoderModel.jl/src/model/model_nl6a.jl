function generate_model_nl6a(xs_s)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature

    return function (ps)
        ps[7] .+ ps[6] .*
            (sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* x2 .+ cos(ps[2])) .*
            (sin(ps[3]) .* x3 .+ cos(ps[3])) .*
            (sin(ps[4]) .* x4 .+ cos(ps[4])) .*
            (sin(ps[5]) .* x5 .+ cos(ps[5]))
    end
end

function init_ps_model_nl6a(xs)
    ps_0 = vcat(repeat([0.], 5), [1., 0.])
    ps_min = vcat(repeat([-π/2.], 5), [-10., -10.])
    ps_max = vcat(repeat([π/2.], 5), [10., 10.])
    
    list_idx_ps = [1,2,3,4,5,6,7]
    list_idx_ps_reg = [1,2,3,4,5,6]
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

mutable struct ModelEncoderNL6a <: ModelEncoder
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
    
    function ModelEncoderNL6a(xs, xs_s, ewma_trim, idx_splits)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            [1,2,3,4,5], nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6a)
    model.f = generate_model_nl6a(model.xs_s)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL6a)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6a(model.xs)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg

    nothing
end