function compute_λ_filt(img::Array{Float32, 3})
    θ_list = zeros(size(img, 3))
    for z_ = 1:size(img, 3)
        img_grad = diff(diff(Float32.(img[:,:,z_]), dims=1), dims=2);
        fitted = Distributions.fit_mle(Distributions.Laplace, img_grad[:])
        θ_list[z_] = fitted.θ
    end
    println("E[θ]: $(mean(θ_list)) σ[θ]: $(std(θ_list))")
    mean(θ_list)
end

function compute_λ_filt(img::Array{Float32, 2})
    img_grad = diff(diff(img, dims=1), dims=2);
    fitted = Distributions.fit_mle(Distributions.Laplace, img_grad[:])
    fitted.θ
end

function filter_rof(img::Array{Float32,3}, λ_filter)
    img_filtered = zeros(Float32, size(img))
    for i = 1:size(img,3)
        img_filtered[:,:,i] = gpu_imROF(Float32.(img[:,:,i]), λ_filter, 100)
    end
    img_filtered
end

function filter_rof(img::Array{Float32,3})
    filter_rof(img, compute_λ_filt(img))
end
