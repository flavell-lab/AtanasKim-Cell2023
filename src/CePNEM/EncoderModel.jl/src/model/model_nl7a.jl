mutable struct ModelEncoderNL7a <: ModelEncoder
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
    
    function ModelEncoderNL7a(xs, xs_s, ewma_trim, idx_splits, idx_predictor=[1,2,3,4,5])
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            idx_predictor, nothing, nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL7a)
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

    model.f = generate_model_nl7_partial(model.xs_s, model.list_idx_ps, model.list_θ, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL7a)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl7_component(model.xs, [2,3,5,6,7])
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end
