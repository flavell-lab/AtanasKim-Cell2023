# nl6d: fixed threshold
function generate_model_nl6d(xs_s, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps)
        ps[12] .+ ps[11] .* ewma((sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, list_θ[1])) .+ cos(ps[2])) .*

            (sin(ps[3]) .* x2 .+ cos(ps[3])) .*
            (sin(ps[4]) .* (1 .- 2 .* lesser.(x2, list_θ[2])) .+ cos(ps[4])) .*

            (sin(ps[5]) .* x3 .+ cos(ps[5])) .*

            (sin(ps[6]) .* x4 .+ cos(ps[6])) .*
            (sin(ps[7]) .* (1 .- 2 .* lesser.(x4, list_θ[4])) .+ cos(ps[7])) .*

            (sin(ps[8]) .* x5 .+ cos(ps[8])) .*
            (sin(ps[9]) .* (1 .- 2 .* lesser.(x5, list_θ[5])) .+ cos(ps[9])), ps[10], ewma_trim, idx_splits)
    end
end

function init_ps_model_nl6d(xs, idx_predictor=[1,2,3,4,5])    
    n_xs = length(idx_predictor)
  
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = [[1,2],[3,4],[5],[6,7],[8,9]]
    list_idx_ps_reg = [[1,2],[3,4],[5],[6,7],[8,9]]

    for (i,b) = enumerate(idx_predictor)
        if b == 3
            push!(ps_0, [0.])
            push!(ps_min, [-pi/2])
            push!(ps_max, [pi/2])
        else
            push!(ps_0, [0., 0.])
            push!(ps_min, [-pi/2, -pi/2])
            push!(ps_max, [pi/2, pi/2])
        end
    end
    ps_0 = vcat(ps_0..., [0.1, 1., 0.])
    ps_min = vcat(ps_min..., [0.03, -10., -10.])
    ps_max = vcat(ps_max..., [1., 10., 10.])
    
    
    list_idx_ps = vcat(list_idx_ps[idx_predictor]..., [10,11,12])
    list_idx_ps_reg = vcat(list_idx_ps_reg[idx_predictor]..., [11]) # 10: \gamma, 12: bias

    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

mutable struct ModelEncoderNL6d <: ModelEncoder
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
    
    function ModelEncoderNL6d(xs, xs_s, ewma_trim, idx_splits)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            [1,2,3,4,5], nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6d)
    list_θ = zeros(eltype(model.xs), 5)
    for i = 1:5
        u = mean(model.xs[i,:])
        s = std(model.xs[i,:])
        
        list_θ[i] = i == 5 ? 0 : -u/s # 5: curvature. set to mean in real scale
    end
    model.f = generate_model_nl6d(model.xs_s, list_θ, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL6d)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6d(model.xs,
        model.idx_predictor)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg

    nothing
end