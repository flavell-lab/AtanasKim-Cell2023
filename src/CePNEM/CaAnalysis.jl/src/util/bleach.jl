using Optim

@. exp_mono(t, p, t_med) = max(0, exp((t_med-t) * p[1]))
@. exp_bi(t, p) = max(0, p[1] * exp(-t * p[2]) + (1 - p[1]) * exp(-t * p[3]))

gen_cost_mono(t, f, t_med) = p -> sum((log.(f) .- log.(exp_mono(t, p, t_med))) .^ 2)
gen_cost_bi(t, f) = p -> sum((exp_bi(t, p) .- f) .^ 2)

"""
    fit_bleach(f, t, plot_fit=true, use_mono=false, quantile_norm=0.5)

Fits double exponential bleaching model.

Arguments
---------
* `f`: 1D data to fit the bleaching model
* `t`: Time points
* `plot_fit`: plot fit result if true
* `use_mono`: use mono exponential model instead of double exponential model if true
"""
function fit_bleach(f, t, plot_fit=true, use_mono=false, quantile_norm=0.5)
    y = f ./ quantile(f, quantile_norm)
    t_med = quantile(1:length(y), quantile_norm)

    optim_opts = Optim.Options(g_tol=1e-15, iterations=1000)

    f_cost_mono = gen_cost_mono(t, y, t_med)
    p0_mono = [0.001]
    mono_fitted = optimize(f_cost_mono, p0_mono, Newton(), optim_opts,
        autodiff=:forward)
    p_fit_mono = mono_fitted.minimizer

    if use_mono
        y_hat_mono = exp_mono(t, p_fit_mono, t_med)
        resid_mono = y_hat_mono .- y
    else
        f_cost_bi = gen_cost_bi(t, y)
        p0_bi = [0.75, p_fit_mono[1], 0.001]
        bi_fitted = optimize(f_cost_bi, p0_bi, Newton(), optim_opts,
            autodiff=:forward)
        p_fit_bi = bi_fitted.minimizer
        y_hat_bi = exp_bi(t, p_fit_bi)
        resid_bi = y_hat_bi .- y # residual
    end

    if plot_fit
        subplot(2,1,1)
        title("Fitted")
        plot(t, y, "k", alpha=0.75)
        plot(t, use_mono ? y_hat_mono : y_hat_bi, "r")

        subplot(2,1,2)
        title("Residual")
        plot(use_mono ? resid_mono : resid_bi, "k", alpha=0.75)

        tight_layout()
        println("Fitted parameter:", use_mono ? p_fit_mono : p_fit_bi)
    end
    if use_mono
        return resid_mono, p_fit_mono, y_hat_mono
    end
    return resid_bi, p_fit_bi, y_hat_bi
end

"""
    fit_bleach(f::Array{<:Real,2}, plot_fit=true, use_mono=false; idx_t=:all)

Calculates mean activity across the units and fits the double exponential
bleaching model.

Arguments
---------
* `f`: N x T data array. N: number of units, T: number of time points.
* `plot_fit`: plot fit result if true
* `use_mono`: use mono exponential model instead of double exponential model if true
"""
function fit_bleach(f::Array{<:Real,2}, plot_fit=true, use_mono=false; idx_t=:all)
    y = dropdims(mean(f, dims=1), dims=1)

    d = Dict()
    d["f"] = f
    t = get_idx_t(d, idx_t)

    fit_bleach(y, t, plot_fit, use_mono)
end

"""
    fit_bleach!(data_dict::Dict, plot_fit=true, use_mono=false; data_key="f_denoised", idx_unit=:ok, idx_t=:all)

Calculates mean activity across the units and fits the double exponential.

Arguments
---------
* `data_dict`: data_dictionary.
* `plot_fit`: plot fit result if true
* `idx_unit`: see [`get_idx_unit()`](@ref)
* `idx_t`: see [`get_idx_t()`](@ref)
* `data_key`: key of data_dict to be used for fitting the model
* `use_mono`: use mono exponential model instead of double exponential model if true
"""
function fit_bleach!(data_dict::Dict, plot_fit=true, use_mono=false; data_key="f_denoised",
    idx_unit=:ok, idx_t=:all)
    f = get_data(data_dict::Dict; data_key=data_key, idx_unit=idx_unit,
        idx_t=idx_t)

    resid_bi, p_fit_bi, y_hat_bi = fit_bleach(f, plot_fit, use_mono, idx_t=idx_t)

    data_dict["bleach_param"] = p_fit_bi
    data_dict["bleach_resid"] = resid_bi
    data_dict["bleach_curve"] = y_hat_bi
    data_dict["f_bleach"] = Array((f ./ y_hat_bi'))

    nothing
end
