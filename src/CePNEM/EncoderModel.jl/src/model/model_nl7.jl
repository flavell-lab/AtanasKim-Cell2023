function generate_model_nl7(xs_s, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps)
        ewma((sin(ps[1]) .* (1 .- 2 .* lesser.(x1, list_θ[1])) .+ cos(ps[1])) .*
            (ps[2] .* x1 .+
                ps[3] .* x2 .+
                ps[4] .* x2 .* lesser.(x2, list_θ[2]) .+
                ps[5] .* x3 .+
                ps[6] .* x4 .+
                ps[7] .* x5 .+
                ps[8]), ps[9], ewma_trim, idx_splits) .+ ps[10]
    end
end

"""
generate_model_nl7_partial

list of components:
1: velocity threshold
2: velocity
3: headangle
4: headnagle threshold
5: pumping
6: ang vel
7: curvature
"""
function generate_model_nl7_partial(xs_s, idx_valid, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps_::Vector{T}) where T
        ps = zeros(T, 10)
        ps[idx_valid] .= ps_

        ewma((sin(ps[1]) .* (1 .- 2 .* lesser.(x1, list_θ[1])) .+ cos(ps[1])) .*
            (ps[2] .* x1 .+
                ps[3] .* x2 .+
                ps[4] .* x2 .* lesser.(x2, list_θ[2]) .+
                ps[5] .* x3 .+
                ps[6] .* x4 .+
                ps[7] .* x5 .+
                ps[8]), ps[9], ewma_trim, idx_splits) .+ ps[10]
    end
end

function init_ps_model_nl7_component(xs, idx_component=1:7)
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = []
    list_idx_ps_reg = []
        
    i = 1
    for b = idx_component
        push!(ps_0, 0.)
        push!(ps_min, b == 1 ? -pi/2 : -10)
        push!(ps_max, b == 1 ? pi/2 : 10)
        push!(list_idx_ps, b)
        push!(list_idx_ps_reg, i)
        i += 1
    end

    ps_0 = vcat(ps_0..., [0., 0.1, 0.]) # offset inside, ewma, offset outside
    ps_min = vcat(ps_min..., [-10., 0.03, -10.])
    ps_max = vcat(ps_max..., [10., 1., 10.])
    
    list_idx_ps = vcat(list_idx_ps..., [8,9,10]) # bias inside, ewma, bias outside ewma
    list_idx_ps_reg = vcat(list_idx_ps_reg..., [i]) # 9: ewma, 10: bias
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

function init_ps_model_nl7(xs, idx_predictor=1:5)
    list_comp_group = [[1,2],[3,4],[5],[6],[7]]
    
    init_ps_model_nl7_component(xs, vcat(list_comp_group[idx_predictor]...))
end

mutable struct ModelEncoderNL7 <: ModelEncoder
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
    list_θ
    
    function ModelEncoderNL7(xs, xs_s, ewma_trim, idx_splits, idx_predictor=[1,2,3,4,5])
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            idx_predictor, nothing, nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL7)
    if isnothing(model.list_idx_ps)
        error("list_idx_ps is nothing. run init_model_ps!")
    end
    
    if isnothing(model.list_θ)
        list_θ = zeros(eltype(model.xs), 5)
        for i = 1:5
            u = mean(model.xs[i,:])
            s = std(model.xs[i,:])

            list_θ[i] = i == 5 ? 0 : -u/s # 5: curvature. set to mean in real scale
        end
        model.list_θ = list_θ
    end
    
    model.f = generate_model_nl7(model.xs_s, model.list_θ, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL7)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl7(model.xs,
        model.idx_predictor)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end
