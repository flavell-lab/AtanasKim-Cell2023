#### component functions
zstd = FlavellBase.standardize
logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x - x0)
lesser(x,x0) = leaky_logistic(x0,x,50,1e-3)

function ewma(x, λ::T, trim::Int) where T
    max_t = length(x)
    x_ewma = zeros(T, max_t)
    s = T(0.)
    for t = 1:max_t
        s += exp(-(max_t-t)*λ)
    end
    
    x_ewma[1] = x[1] / s
    @inbounds for t = 2:max_t
        x_ewma[t] = (x_ewma[t-1] * (s-1) / s + x[t] / s)
    end
    x_ewma[1:trim] .= 0
    
    x_ewma
end


function ewma(x, λ::T, trim::Int, idx_splits::Vector{UnitRange{Int}}) where T
    max_t = length(x)
    x_ewma = zeros(T, max_t)

    for split = idx_splits
        ewma!((@view x_ewma[split]), (@view x[split]), λ, trim)
    end
    
    x_ewma
end

function ewma!(result, x, λ, trim::Int)
    max_t = length(x)
    # x_ewma = zeros(max_t)
    s = 0
    for t = 1:max_t
        s += exp(-(max_t-t)*λ)
    end
    
    result[1] = x[1] / s
    @inbounds for t = 2:max_t
        result[t] = (result[t-1] * (s-1) / s + x[t] / s)
    end
    result[1:trim] .= 0
    
    result
end

function ewma!(result, x, λ, trim::Int, idx_splits::Vector{UnitRange{Int}})
    max_t = length(x)

    for split = idx_splits
        ewma!((@view result[split]), (@view x[split]), λ, trim)
    end
    
    result
end

#### Model definitions
abstract type ModelEncoder end

function n_ps(model::ModelEncoder)
    length(model.ps_0)
end

include("model/model_nl5.jl")

include("model/model_nl6.jl")
include("model/model_nl6a.jl")
include("model/model_nl6b.jl")
include("model/model_nl6c.jl")
include("model/model_nl6d.jl")
include("model/model_nl6e.jl")
include("model/model_nl6f.jl")
include("model/model_nl7.jl")
include("model/model_nl7a.jl")
include("model/model_nl10c.jl")
include("model/model_nl10d.jl")
include("model/model_nl10e.jl")