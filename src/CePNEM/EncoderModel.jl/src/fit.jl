function fit_model_glopt_bound(trace, model, f_cost, ps_0, ps_min, ps_max,
        idx_trace, idx_behavior, optimizer=ADAM(); max_iter=1000)
    cost_model(ps, x) = f_cost(trace, model(ps), idx_trace, idx_behavior)
    f_opt_cost = OptimizationFunction(cost_model, GalacticOptim.AutoForwardDiff())
    prob_opt = OptimizationProblem(f_opt_cost, ps_0, lb=ps_min, ub=ps_max)
    opt_sol = solve(prob_opt, optimizer, maxiters=max_iter)
    
    opt_sol, opt_sol.u
end

function fit_model_glopt_bound_reg(trace, model, f_cost, ps_0, ps_min, ps_max, 
        idx_trace, idx_behavior, optimizer=ADAM(); 
        λ, f_reg::Function, idx_ps, max_iter=1000)
    cost_model(ps, x) = f_cost(trace, model(ps), idx_trace, idx_behavior) .+ λ * f_reg(ps[idx_ps])
    f_opt_cost = OptimizationFunction(cost_model, GalacticOptim.AutoForwardDiff())
    prob_opt = OptimizationProblem(f_opt_cost, ps_0, lb=ps_min, ub=ps_max)
    opt_sol = solve(prob_opt, optimizer, maxiters=max_iter)
    
    opt_sol, opt_sol.u
end

function fit_model_nlopt_bound(trace, model, f_cost, ps_0, ps_min, ps_max,
        idx_trace, idx_behavior, optimizer_g, optimizer_l;
        max_time=30, xtol=1e-4, ftol=1e-4, max_eval=-1)
    
    f(ps) = f_cost(trace, model(ps), idx_trace, idx_behavior)
    function cost(ps::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, f, ps)
        end
        return f(ps)
    end

    n_ps = length(ps_0)
    opt = Opt(optimizer_g, n_ps)
    opt.min_objective = cost
    opt.maxtime = max_time
    opt.maxeval = max_eval
    opt.lower_bounds = ps_min
    opt.upper_bounds = ps_max
    opt.min_objective = cost
    opt.xtol_rel = xtol
    opt.ftol_rel = ftol
    
    if !isnothing(optimizer_l)
        local_optimizer = Opt(optimizer_l, n_ps)
        local_optimizer.xtol_rel = xtol
        local_optimizer.ftol_rel = ftol
        local_optimizer!(opt, local_optimizer)
    end
    
    NLopt.optimize(opt, ps_0), opt.numevals # ((final cost, u_opt, exit code), num f eval)
end

function fit_model_nlopt_bound_reg(trace, model::Function, f_cost::Function, ps_0, ps_min, ps_max, 
        idx_trace, idx_behavior, optimizer_g::Symbol, optimizer_l::Union{Symbol,Nothing};
        λ, f_reg::Function, idx_ps,
        max_time=30, xtol=1e-4, ftol=1e-4, max_eval=-1)
    
    f(ps) = f_cost(trace, model(ps), idx_trace, idx_behavior) .+ λ * f_reg(ps[idx_ps])
    function cost(ps::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, f, ps)
        end
        return f(ps)
    end

    n_ps = length(ps_0)
    opt = Opt(optimizer_g, n_ps)
    opt.min_objective = cost
    opt.maxtime = max_time
    opt.maxeval = max_eval
    opt.lower_bounds = ps_min
    opt.upper_bounds = ps_max
    opt.min_objective = cost
    opt.xtol_rel = xtol
    opt.ftol_rel = ftol
    
    if !isnothing(optimizer_l)
        local_optimizer = Opt(optimizer_l, n_ps)
        local_optimizer.xtol_rel = xtol
        local_optimizer.ftol_rel = ftol
        local_optimizer!(opt, local_optimizer)
    end
    
    NLopt.optimize(opt, ps_0), opt.numevals # ((final cost, u_opt, exit code), num f eval)
end
