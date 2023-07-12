function compute_unit_cor(f, stim)
    cor_result = zeros(size(f, 1))

    for i = 1:size(f, 1)
        cor_result[i] = cor(f[i,:], stim)
    end

    cor_result
end

function plot_unit_cor(f, stim, idx_stim, n=10; α_highlight=0.1)
    @assert(n % 2 == 0)

    cor_result = compute_unit_cor(f, stim)
    cor_unit = collect(zip(1:length(cor_result), cor_result))
    sort!(cor_unit, by=x->abs(x[2]), rev=true)

    for i = 1:n
        unit_ = cor_unit[i][1]
        subplot(Int(n/2), 2, i)
        plot(f[unit_,:])
        highlight_stim(idx_stim, α_highlight)
        title("Unit $unit_ Cor: $(round(cor_unit[i][2], digits=3))")
    end
    tight_layout()

    nothing
end

function plot_unit_cor(data_dict, n=10; idx_unit=:ok, idx_t=:all,
        data_key="f_bleach", α_highlight=0.1)
    f = get_data(data_dict, data_key=data_key, idx_unit=idx_unit,
            idx_t=idx_t)
    stim = get_stim(data_dict, idx_t=idx_t)
    idx_stim = get_idx_stim(data_dict, idx_t=idx_t)
    plot_unit_cor(f, stim, idx_stim, n,
        α_highlight=α_highlight)

    nothing
end
