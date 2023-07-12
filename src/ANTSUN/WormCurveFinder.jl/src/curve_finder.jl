downsample(data) = (data[1:2:end-1,1:2:end-1] .+ data[2:2:end,2:2:end]) / 2
downsample(data,n) = foldl((x,y)->downsample(x), 1:n, init=data)

function center_of_mass(img)
    M = 0.
    R = [0.,0.]
    for i = 1:size(img,1), j = 1:size(img,2)
        M += img[i,j]
        R .+= img[i,j] .* [i, j]
    end
    R ./ M
end

function img_axis_pca(img_bin)
    points = findall(img_bin)
    points = Float64.(hcat(getindex.(points, 1), getindex.(points, 2)))

    # mean center
    μ_x = mean(points[:,1])
    μ_y = mean(points[:,2])
    points[:,1] .-= μ_x
    points[:,2] .-= μ_y

    cov_mat = cov(points)
    mat_eigvals = eigvals(cov_mat)
    mat_eigvecs = eigvecs(cov_mat)

    eigvals_order = sortperm(mat_eigvals, rev=true) # sort order descending

    (μ_x, μ_y), mat_eigvecs[eigvals_order[1],:], points
end

function calc_θ_vecs(u, v)
    acos(dot(u, v) / (norm(u) * norm(v)))
end


function calc_head_vec_sign(axis_vec, points, center_xy, img)
    θ = tanh(axis_vec[1] / axis_vec[2])
    mat_rot = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    ptx_tfm = (mat_rot * points')'
    intensity_tfm = [bilinear_interpolation(img, x_, y_) for (x_, y_) =
        zip(ptx_tfm[:,1] .+ center_xy[1], ptx_tfm[:,2] .+ center_xy[2])];
    idx_pos = findall(ptx_tfm[:,2] .> 0);
    idx_neg = findall(ptx_tfm[:,2] .< 0);

    sum(intensity_tfm[idx_pos]) > sum(intensity_tfm[idx_neg]) ? 1 : -1
end

function find_head_tip(head_vec, idx_corners, center_xy)
    N_corner = size(idx_corners,1)
    result_θ = zeros(N_corner) # angle between head vector and corner vector
    result_mag = zeros(N_corner) # magnitude

    for i = 1:N_corner
        result_θ[i] = calc_θ_vecs(head_vec,
            [idx_corners[i][1], idx_corners[i][2]] .- center_xy)
        result_mag[i] = norm([idx_corners[i][1], idx_corners[i][2]] .-
            center_xy)
    end

    idx_θ_filter = findall(abs.(result_θ) .< 45 * pi ./ 180)
    idx_max = findmax(result_mag[idx_θ_filter])[2]
    idx_ = idx_θ_filter[idx_max]
    idx_corners[idx_], result_θ[idx_], result_mag[idx_]
end

function find_curve(img, n_downsample=3, ptx_init=nothing, n_ptx=10)
    img_ds = downsample(img, n_downsample)
    img_bin = img_ds .> mean(img_ds)

    center_xy, axis_vec, points = img_axis_pca(img_bin)

    head_vec = calc_head_vec_sign(axis_vec, points, center_xy, img_ds) *
        axis_vec

    idx_corners = findall(imcorner(closing(img_bin)))
    head_tip, θ_head = find_head_tip(head_vec, idx_corners, center_xy)


    x_max, y_max = Tuple(argmax(img_ds))
    x_com, y_com = center_of_mass(img_ds .* img_bin)

    x_init, y_init = isnothing(ptx_init) ? head_tip.I : ptx_init

    # println(calc_θ_vecs([x_max, y_max], head_tip.I .- center_xy))
    # θ_init = calc_θ_vecs([x_max, y_max], head_tip.I .- center_xy)
    # θ_init = pi - θ_head

    θ_init = pi + atan(reverse(([x_init, y_init] .- [x_max, y_max]))...)

    x,y,θ = track_curve(img_ds, x_init, y_init, θ_init, n_ptx, 4)
    prepend!(x, x_init); prepend!(y, y_init);

    x .* (2 ^ n_downsample), y .* (2 ^ n_downsample), x_com, y_com
end
