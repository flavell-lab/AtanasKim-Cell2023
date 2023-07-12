function highlight_stim(idx_stim::Array{Int64, 2}, α_highlight=0.1)
    ax = gca()
    for i = 1:size(idx_stim, 1)
        ax.axvspan(idx_stim[i,1], idx_stim[i,2],
            alpha=α_highlight, facecolor="red", lw=0)
    end

    nothing
end

function highlight_stim(Y, prjax, stim; cmap=PyPlot.cm.hot, skip_0=true, s=20)
    if !(length(prjax) in [2,3])
        error("length(prjax) should be 2 or 3")
    end

    idx_plot = skip_0 ? findall(stim .!= 0) : 1:length(stim)

    stim = Float64.(stim)
    stim_rescaled = stim[idx_plot] ./ maximum(stim[idx_plot])

    ax = gca()
    ax.scatter([Y[prjax[i], idx_plot] for i=1:length(prjax)]...,
        c=cmap(stim_rescaled), edgecolor=nothing, s=s)

    nothing
end
