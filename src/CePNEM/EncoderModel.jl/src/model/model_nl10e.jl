function generate_model_nl10e(xs_s, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    
    return function (ps)
        ((ps[1] + 1) / sqrt(ps[1] ^ 2 + 1) .- 2 * ps[1] / sqrt(ps[1] ^ 2 + 1) .*
            lesser.(x1, list_θ[1])) .* (ps[2] .* x1 .+ ps[3] .* x2 .+ ps[4] .* x3) .+ ps[5]
    end
end

function generate_model_nl10e_partial(xs_s, idx_valid, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    
    return function (ps_::Vector{T}) where T
        ps = zeros(T, 5)
        ps[idx_valid] .= ps_

        ((ps[1] + 1) / sqrt(ps[1] ^ 2 + 1) .- 2 * ps[1] / sqrt(ps[1] ^ 2 + 1) .*
            lesser.(x1, list_θ[1])) .* (ps[2] .* x1 .+ ps[3] .* x2 .+ ps[4] .* x3) .+ ps[5]
    end
end


function init_ps_model_nl10e_component(xs, idx_component=1:4)
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = Int[]
    list_idx_ps_reg = Int[]
        
    for (i, c) = enumerate(idx_component)
        push!(ps_0, 0.)
        push!(ps_min, c == 1 ? -3 : -10)
        push!(ps_max, c == 1 ? 3 : 10)
        push!(list_idx_ps, c)
        push!(list_idx_ps_reg, i)
    end

    ps_0 = vcat(ps_0..., [0.]) # offset outside
    ps_min = vcat(ps_min..., [-10.])
    ps_max = vcat(ps_max..., [10.])
    
    list_idx_ps = vcat(list_idx_ps..., [5]) # bias outside
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end

function init_ps_model_nl10e(xs, idx_predictor=1:3)
    list_comp_group = [[1,2],[3],[4]]
    
    init_ps_model_nl10e_component(xs, vcat(list_comp_group[idx_predictor]...))
end

mutable struct ModelEncoderNL10e <: ModelEncoder
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
    
    function ModelEncoderNL10e(xs, xs_s, ewma_trim, idx_splits, idx_predictor=[1,2,3])
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            idx_predictor, nothing, nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL10e)
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
    
    model.f = generate_model_nl10e(model.xs_s, model.list_θ, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL10e)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl10e(model.xs,
        model.idx_predictor)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end