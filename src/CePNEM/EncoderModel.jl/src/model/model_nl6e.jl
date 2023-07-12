# nl6e: partial threshol
function init_ps_model_nl6e(xs, idx_threshold=[1,2,4,5])  
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = []
    list_idx_ps_reg = []
    idx_ps = [[1,[2,3]],[4,[5,6]],[7,[]],[8,[9,10]],[11,[12,13]]]
    # idx_ps_reg = [[1,[2,]],[3,[4,]],[5,[]],[6,[7,]],[8,[9,]]]

    i_reg = 1
    
    for b = 1:5
        if b in idx_threshold
            θ = b == 5 ? 0 : -mean(xs[b,:])/std(xs[b,:])
            push!(ps_0, [0., 0., θ])
            push!(ps_min, [-pi/2, -pi/2, percentile(zstd(xs[b,:]), 5)])
            push!(ps_max, [pi/2, pi/2, percentile(zstd(xs[b,:]), 95)])
            push!(list_idx_ps, idx_ps[b][1])
            push!(list_idx_ps, idx_ps[b][2])
            push!(list_idx_ps_reg, [i_reg,i_reg+1])
            i_reg += 3
            # push!(list_idx_ps_reg, idx_ps_reg[b][1])
            # push!(list_idx_ps_reg, idx_ps_reg[b][2])
        else
            push!(ps_0, [0.])
            push!(ps_min, [-pi/2])
            push!(ps_max, [pi/2])
            push!(list_idx_ps, idx_ps[b][1])
            push!(list_idx_ps_reg, i_reg)
            i_reg += 1
            # push!(list_idx_ps_reg, idx_ps_reg[b][1])
        end
    end
    
    ps_0 = vcat(ps_0..., [0.1, 1., 0.])
    ps_min = vcat(ps_min..., [0.03, -10., -10.])
    ps_max = vcat(ps_max..., [1., 10., 10.])

    list_idx_ps = vcat(sort(vcat(list_idx_ps...)), [14,15,16])
    list_idx_ps_reg = vcat(sort(vcat(list_idx_ps_reg...)), [i_reg+1]) # 14: \gamma, 16: bias

    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end


mutable struct ModelEncoderNL6e <: ModelEncoder
    xs
    xs_s
    ewma_trim::Int
    idx_splits::Union{Vector{UnitRange{Int64}}, Vector{Int64}}
    ps_0
    ps_min
    ps_max
    idx_threshold
    f
    list_idx_ps
    list_idx_ps_reg
    
    function ModelEncoderNL6e(xs, xs_s, ewma_trim, idx_splits, idx_threshold)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            idx_threshold, nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6e)
    if isnothing(model.list_idx_ps)
        error("list_idx_ps is nothing. run init_model_ps!")
    end
    model.f = generate_model_nl6_partial(model.xs_s, model.list_idx_ps, model.ewma_trim, model.idx_splits)
    nothing
end

function init_model_ps!(model::ModelEncoderNL6e)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6e(model.xs,
        model.idx_threshold)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end
