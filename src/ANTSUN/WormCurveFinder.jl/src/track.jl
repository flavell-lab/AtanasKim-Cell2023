softplus(x, scale = 1) = scale * log(1 + exp(x / scale))

function update_xyθ(n_steps::Integer, x::Real, y::Real, θ::Real, dθ::Real)
    for i=1:n_steps
        if i <= 5
            θ += dθ / 5
        end
        x += cos(θ)
        y += sin(θ)
    end
    x, y, θ
end

function pixel(img, x, y, θ, dθ, i, j)
    x, y, θ = update_xyθ(j, x, y, θ, dθ)
    θ_perp = θ + π/2
    bilerp(img, x + cos(θ_perp) * i, y + sin(θ_perp) * i)
end

function score(img_a, len, x, y, θ, dθ, wt, long = true)
    img2 = [pixel(img_a, x, y, θ, dθ, i, j) for i = -4:4, j = 1:len]
    v_sum = sum(img2, dims=1)
    v_sum_med = median(v_sum)
    sum(wt .* img2 ./ max.(v_sum / v_sum_med, 1.0))
end

function _score(img, len, x, y, θ, dθ, wt)
    img2 = [pixel(img, x, y, θ, dθ, i, j) for i = -4:4, j = 1:len]
    img3 = img2 ./ sum(img2, 1)
    img4 = wt .* img3
    img2, img3, img4
end

function track_curve!(x, y, θ, img, x0::Real, y0::Real, θ0::Real,
        len_segment::Real)
    n_segments = length(x)
    rθ = -0.8:0.05:0.8
    rθ2 = -0.2:0.02:0.2
    wt = 3 .- abs.(-4:4)
    curve_length = len_segment * n_segments
    _x, _y, _θ = x0, y0, θ0
    for i = 1:n_segments
        cur_length = len_segment * i - len_segment
        long_length = min(cur_length + len_segment * 6, curve_length) - cur_length
        short_length = min(cur_length + len_segment * 3, curve_length) - cur_length
        _scores = [score(img, long_length, _x, _y, _θ, dθ, wt) for dθ in rθ]
        _best_dθ = rθ[argmax(_scores)]
        rθ3 = rθ2 .+ _best_dθ
        scores = [score(img, short_length, _x, _y, _θ, dθ, wt, false) for dθ in rθ3]
        best_dθ = rθ3[argmax(scores)]
        _x, _y, _θ = update_xyθ(len_segment, _x, _y, _θ, best_dθ)
        @inbounds θ[i] = _θ
        @inbounds x[i] = _x
        @inbounds y[i] = _y
    end
    x, y, θ
end

track_curve(img, x0, y0, θ0, n_segments = 37, len_segment=5) = track_curve!(
    zeros(n_segments), zeros(n_segments), fill(θ0, n_segments), img, x0, y0,
    θ0, len_segment)
