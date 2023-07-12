function generate_model_nl6f(xs_s, idx_valid, list_θ, ewma_trim, idx_splits)
    x1 = xs_s[1,:] # velocity
    x2 = xs_s[2,:] # θh
    x3 = xs_s[3,:] # pumping
    x4 = xs_s[4,:] # ang vel
    x5 = xs_s[5,:] #curvature
    
    return function (ps_::AbstractVector{T}) where T
        ps = zeros(T, 16)
        ps[[3,6,10,13]] .= list_θ[[1,2,4,5]]
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

function init_ps_model_nl6f(xs, idx_threshold=[1,2,4,5])  
    ps_0 = []
    ps_min = []
    ps_max = []
    list_idx_ps = []
    list_idx_ps_reg = []
    idx_ps = [[[1,2],[3]], [[4,5],[6]],[[7],[]],[[8,9],[10]],[[11,12],[13]]]
    idx_ps_reg = [[1,2],[4,5],[7],[8,9],[11,12]]

    for b = 1:5
        if b == 3
            push!(ps_0, [0.])
            push!(ps_min, [-pi/2])
            push!(ps_max, [pi/2])
            push!(list_idx_ps, idx_ps[b][1])
            push!(list_idx_ps_reg, idx_ps_reg[b])
        else
            if b in idx_threshold
                θ = b == 5 ? 0 : -mean(xs[b,:])/std(xs[b,:])
                push!(ps_0, [0., 0., θ])
                push!(ps_min, [-pi/2, -pi/2, percentile(zstd(xs[b,:]), 5)])
                push!(ps_max, [pi/2, pi/2, percentile(zstd(xs[b,:]), 95)])
                push!(list_idx_ps, idx_ps[b][1])
                push!(list_idx_ps, idx_ps[b][2])
            else
                push!(ps_0, [0., 0.])
                push!(ps_min, [-pi/2, -pi/2])
                push!(ps_max, [pi/2, pi/2])
                push!(list_idx_ps, idx_ps[b][1])
            end
            
            push!(list_idx_ps_reg, idx_ps_reg[b])
        end
    end
    
    ps_0 = vcat(ps_0..., [0.1, 1., 0.])
    ps_min = vcat(ps_min..., [0.03, -10., -10.])
    ps_max = vcat(ps_max..., [1., 10., 10.])

    list_idx_ps = vcat(sort(vcat(list_idx_ps...)), [14,15,16])
    list_idx_ps_reg = vcat(sort(vcat(list_idx_ps_reg...)), [15]) # 14: \gamma, 16: bias

    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg
end


mutable struct ModelEncoderNL6f <: ModelEncoder
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
    
    function ModelEncoderNL6f(xs, xs_s, ewma_trim, idx_splits, idx_threshold)
        new(xs, xs_s, ewma_trim, idx_splits, nothing, nothing, nothing,
            idx_threshold, nothing, nothing, nothing)
    end
end

function generate_model_f!(model::ModelEncoderNL6f)
    if isnothing(model.list_idx_ps)
        error("list_idx_ps is nothing. run init_model_ps!")
    end
    list_θ = zeros(eltype(model.xs), 5)
    for i = [1,2,4,5]
        u = mean(model.xs[i,:])
        s = std(model.xs[i,:])
        
        list_θ[i] = i == 5 ? 0 : -u/s # 5: curvature. set to mean in real scale
    end

    model.f = generate_model_nl6f(model.xs_s, model.list_idx_ps, list_θ, model.ewma_trim, model.idx_splits)
    
    nothing
end

function init_model_ps!(model::ModelEncoderNL6f)
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6f(model.xs,
        model.idx_threshold)
    
    model.ps_0 = ps_0
    model.ps_min = ps_min
    model.ps_max = ps_max
    model.list_idx_ps = list_idx_ps
    model.list_idx_ps_reg = list_idx_ps_reg
    
    nothing
end