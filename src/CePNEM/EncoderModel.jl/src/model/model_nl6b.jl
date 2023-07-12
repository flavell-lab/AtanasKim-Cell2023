function generate_model_nl6b(xs_s, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps)
        ps[8] .+ ps[7] .* ewma((sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* x2 .+ cos(ps[2])) .*
            (sin(ps[3]) .* x3 .+ cos(ps[3])) .*
            (sin(ps[4]) .* x4 .+ cos(ps[4])) .*
            (sin(ps[5]) .* x5 .+ cos(ps[5])), ps[6], ewma_trim, idx_splits)
    end
end

# TODO: return valid idx and reg idx
function init_ps_model_nl6b(xs)
    ps_0 = vcat(repeat([0.], 5), [0.1, 1., 0.])
    ps_min = vcat(repeat([-pi/2.], 5), [0.03, -10., -10.])
    ps_max = vcat(repeat([pi/2.], 5), [1., 10., 10.])
    
    list_idx_ps = [1,2,3,4,5,6,7,8]
    list_idx_ps_reg = [1,2,3,4,5,7]
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg

end

mutable struct ModelEncoderNL6b <: ModelEncoder
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
    
    function ModelEncoderNL6b(xs, xs_s, ewma_trim, idx_splits)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            [1,2,3,4,5], nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6b)
    model.f = generate_model_nl6b(model.xs_s, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL6b)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6b(model.xs)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg

    nothing
end