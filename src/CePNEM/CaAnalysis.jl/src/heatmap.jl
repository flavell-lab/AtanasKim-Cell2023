using Clustering

function plot_cluster_cost(f::Array{T,2}, n=20) where T
    cost_list = zeros(n)

    for i = 1:n
        clustered=kmeans(Array(f'), i+1, init=:kmcen, maxiter=200,
            tol=1.0e-8)
        cost_list[i] = clustered.totalcost
    end

    subplot(2,1,1)
    plot(2:n+1, cost_list, "ko-")
    ylabel("Total cost")
    xlabel("Number of clusters")
    xlim(0,n+2)
    xticks(2:2:n+1)

    subplot(2,1,2)
    plot(3:n+1, diff(cost_list), "ko-")
    ylabel("Δ(Total cost)")
    xlabel("Number of clusters")
    xlim(0,n+2)
    xticks(2:2:n+1)

    tight_layout()

    nothing
end

function plot_cluster_cost(data_dict::Dict, n=20; data_key="f_bleach",
    idx_unit=:ok, idx_t=:all)
    f = get_data(data_dict, data_key=data_key, idx_unit=idx_unit,
        idx_t=idx_t)

    plot_cluster_cost(f, n)

    nothing
end

function order_by_kmeans(n=8)
    f_plot -> let
        clustered = kmeans(Array(f_plot'), n, init=:kmcen, maxiter=200,
            tol=1.0e-8)

        sortperm(clustered.assignments)
    end
end

function order_by_cor(stim)
    f_plot -> let
        sortperm(compute_unit_cor(f_plot, stim), rev=true)
    end
end

function plot_heatmap(f::Array{T,2}; f_order=nothing, vmin=0.5, vmax=1.5,
        cmap="magma", vol_rate=0.75, aspect="auto") where T

    f_plot = f ./ mean(f, dims=2)
    plot_order = isnothing(f_order) ? (1:size(f_plot, 1)) : f_order(f_plot)

    imshow(f_plot[plot_order,:], aspect=aspect, vmin=vmin, vmax=vmax,
        cmap=cmap)

    xlabel("Time (min)")
    ylabel("Unit")

    Δ_minute = 5 # minute
    n_vol = size(f, 2)
    t_rg = 0:floor(Int, n_vol * vol_rate / 60 / Δ_minute)
    x_rg = collect(t_rg) * 60 * Δ_minute / vol_rate
    xticks(x_rg, Δ_minute * collect(t_rg))

    nothing
end

function plot_heatmap(data_dict; f_order=nothing, cmap="magma",
    data_key="f_bleach", idx_unit=:ok, idx_t=:all, vmin=0.5, vmax=1.5,
    vol_rate=0.75, plot_stim=false, stim_key="opto_activation")

    f = get_data(data_dict, data_key=data_key, idx_unit=idx_unit,
        idx_t=idx_t)

    if plot_stim
        if !haskey(data_dict, stim_key)
            error("plot_stim is true, but key \"$(stim_key)\"" *
                "is not in data_dict")
        end
        if length(data_dict[stim_key]) != size(data_dict["f"], 2)
            error("length(data_dict[\"$(stim_key)\"]) != # of time points in f")
        end

        stim = get_stim(data_dict, idx_t=idx_t, stim_key=stim_key)

        ax1 = subplot2grid((6,1), (0,0), rowspan=4)
        plot_heatmap(f, f_order=f_order, vmin=vmin, vmax=vmax, cmap=cmap,
            vol_rate=vol_rate)
        xlabel("")

        ax2 = subplot2grid((6,1), (4,0), sharex=ax1, rowspan=2)
        plot(stim, "r")
        xlabel("Time (min)")
        ylabel("Stim")
        xlim(0, (length(stim)-1))
    else
        plot_heatmap(f, f_order=f_order, vmin=vmin, vmax=vmax, cmap=cmap,
            vol_rate=vol_rate)
    end

    nothing
end
