#### cost
function cost_rss(y, y_pred)
    sum((y .- y_pred) .^ 2)
end

function cost_rss(y, y_pred, idx_trace, idx_behavior)
    M = 0.
    @inbounds for i = 1:length(idx_trace)
        M += (y[idx_trace[i]] - y_pred[idx_behavior[i]]) ^ 2
    end
    M
end

function cost_abs(y, y_pred)
    sum(abs.(y .- y_pred))
end

function cost_abs(y, y_pred, idx_trace, idx_behavior)
    M = 0.
    @inbounds for i = 1:length(idx_trace)
        M += abs(y[idx_trace[i]] - y_pred[idx_behavior[i]])
    end
    M / length(idx_trace)
end
    
function cost_mse(y, y_pred)
    mean((y .- y_pred) .^ 2)
end

function cost_mse(y, y_pred, idx_trace, idx_behavior)
    M = 0.
    @inbounds for i = 1:length(idx_trace)
        M += (y[idx_trace[i]] - y_pred[idx_behavior[i]]) ^ 2
    end
    M / length(idx_trace)
end
    
function cost_cor(y, y_pred)
    - cor(y, y_pred)
end

#### regularization
function reg_var_L1(list_θ, list_σ2)
    sum(abs.(list_θ ./ (list_σ2 .+ 1)))
end

function reg_var_L2(list_θ, list_σ2)
    sum((list_θ ./ (list_σ2 .+ 1)) .^ 2)
end

function reg_L1(ps)
    sum(abs.(ps))
end

function reg_L2(ps)
    sum(ps .^ 2)
end
        
function reg_L1_nl7(ps)
    sum(abs.(vcat(sin(ps[1]), ps[2:end])))
end
