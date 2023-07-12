function pca(X; args...)
    multivar_fit(X, PCA; args...)
end

function multivar_fit(X, f; args...)
    M = fit(f, X; args...)
    Yt = transform(M, X) # transform observations into latent variable
    (M, X, Yt)
end

function plot_pca_var(M, n=10)
    var_list = (M.prinvars / M.tprinvar)[1:n]

    PyPlot.bar(1:n, var_list, color="k", alpha=0.5)
    plot(1:n, cumsum(var_list),"ko-")
    yticks(0.2:0.2:1.0)
    ylim(0,1.0)
    xlabel("PC #")
    ylabel("Var exp %")

    nothing
end

function plot_statespace_component(Y, n=10; idx_stim=nothing, α_highlight=0.1)
    @assert(n % 2 == 0)

    for i = 1:n
        subplot(Int(n/2), 2, i)
        plot(Y[i, :])
        title("Component $i")

        if !isnothing(idx_stim)
            highlight_stim(idx_stim, α_highlight)
        end
    end
    tight_layout()

    nothing
end

function plot_statespace_3d(Y, prjax=[1,2,3])
    if length(prjax) != 3 || !(eltype(prjax) <: Signed)
        error("prjax should be list of 3 integers")
    end

    ax = gcf().add_subplot(111, projection="3d")
    ax.plot(Y[prjax[1],:], Y[prjax[2],:], Y[prjax[3],:], color="k", alpha=0.5)

    ax.view_init(45,15)
    xlabel("Axis 1")
    ylabel("Axis 2")
    zlabel("Axis 3")

    nothing
end

function plot_statespace_2d(Y, prjax=[1,2])
    if length(prjax) != 2 || !(eltype(prjax) <: Signed)
        error("prjax should be list of 2 integers")
    end

    plot(Y[prjax[1],:], Y[prjax[2],:], color="k", alpha=0.5)

    xlabel("Axis 1")
    ylabel("Axis 2")

    nothing
end

function plot_statespace_2d_stim(Y, idx_comp, stim, idx_stim; margin=25,
        align_translate=false, cmap=PyPlot.cm.hot)
    idx_c1, idx_c2 = idx_comp
    max_len = maximum(diff(idx_stim, dims=2) .+ 1)
    n_stim = size(idx_stim, 1)

    stim_events = zeros(n_stim, 2, max_len)

    for i = 1:n_stim
        idx_start, idx_end = idx_stim[i,:]
        stim_length = idx_end - idx_start + 1
        println((idx_start, idx_end))
        c1_0 = align_translate ? Y[idx_c1, idx_start] : 0
        c2_0 = align_translate ? Y[idx_c2, idx_start] : 0

        stim_events[i, 1, 1:stim_length] = Y[idx_c1, idx_start:idx_end] .- c1_0
        stim_events[i, 2, 1:stim_length] = Y[idx_c2, idx_start:idx_end] .- c2_0
    end

    c1_min, c1_max = extrema(stim_events[:,1,:])
    c2_min, c2_max = extrema(stim_events[:,2,:])
    c1_min -= 0.2 * abs(c1_min)
    c2_min -= 0.2 * abs(c2_min)
    c1_max += 0.2 * abs(c1_max)
    c2_max += 0.2 * abs(c2_max)


    for i = 1:n_stim
        subplot(1, n_stim, i)
        idx_start, idx_end = idx_stim[i,:]
        stim_length = idx_end - idx_start + 1

        c1 = stim_events[i, 1, 1:stim_length]
        c2 = stim_events[i, 2, 1:stim_length]

        # scatter(c1, c2, c="k", alpha=0.5, edgecolor="none")
        highlight_stim(hcat(c1,c2)', [1,2], stim[idx_start:idx_end],
            cmap=cmap)
        xlim(c1_min, c1_max)
        ylim(c2_min, c2_max)
        title("Stim #$(i)")
        if i == 1
            xlabel("Component $(idx_c1)")
            ylabel("Component $(idx_c2)")
        end
    end

    tight_layout()
end

function plot_loading(M, i=1; agrs...)
    w_ = projection(M)[:,i]
    w_order = sortperm(w_)
    x = 1:length(w_)

    bar(x, w_[w_order], color="gray"; agrs...)
end

function highlight_loading(M, i=1; idx_roi, idx_use, color="red", args...)
    w_ = projection(M)[:,i]
    x = 1:length(w_)
    w_order = sortperm(w_)
    w_order_rev_hash = Dict(w_order .=> x)

    idx_mark = intersect(idx_use, idx_roi)

    x_mark = [w_order_rev_hash[i] for i = idx_mark]
    bar(x_mark, w_[idx_mark], color=color; args...)
end
