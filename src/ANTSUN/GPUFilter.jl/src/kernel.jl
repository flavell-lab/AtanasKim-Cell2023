function kernel_grad_u_mag(grad_u_mag_out, grad_u_in, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    grad_u_mag_out[i,j] = sqrt(abs2(grad_u_in[i,j,1]) + abs2(grad_u_in[i,j,2]))

    return nothing
end

function kernel_grad_u(grad_u_out, u_in, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    # forwarddiffy
    if i < size_x
        grad_u_out[i,j,1] = u_in[i+1,j] - u_in[i,j]
    else
        grad_u_out[i,j,1] = u_in[size_x,j] - u_in[i,j]
    end

    # forwarddiffx
    if j < size_y
        grad_u_out[i,j,2] = u_in[i,j+1] - u_in[i,j]
    else
        grad_u_out[i,j,2] = u_in[i,size_y] - u_in[i,j]
    end

    return nothing
end

function kernel_forwarddiffx(u_out, u_in, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    if j < size_y
        u_out[i,j] = u_in[i,j+1] - u_in[i,j]
    else
        u_out[i,j] = u_in[i,size_y] - u_in[i,j]
    end

    return nothing
end

function kernel_forwarddiffy(u_out, u_in, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    if i < size_x
        u_out[i,j] = u_in[i+1,j] - u_in[i,j]
    else
        u_out[i,j] = u_in[size_x,j] - u_in[i,j]
    end

    return nothing
end

function kernel_u(u_out, img, λ, div_p, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    u_out[i,j] = img[i,j] - λ * div_p[i,j]

    return nothing
end

function kernel_div(p_out, p_in, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end


    if i == size_x
        p_out[i,j] = - p_in[i-1,j,1]
    elseif i == 1
        p_out[i,j] = p_in[i,j,1]
    else
        p_out[i,j] = p_in[i,j,1] - p_in[i-1,j,1]
    end

    if j == size_y
        p_out[i,j] += - p_in[i,j-1,2]
    elseif j == 1
        p_out[i,j] += p_in[i,j,2]
    else
        p_out[i,j] += p_in[i,j,2] - p_in[i,j-1,2]
    end

    return nothing
end

function kernel_p_update(p_in, grad_u, grad_u_mag, λ, τ, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    p_in[i,j,1] = (p_in[i,j,1] - (τ/λ) * grad_u[i,j,1]) / (1 + (τ/λ) *
        grad_u_mag[i,j])
    p_in[i,j,2] = (p_in[i,j,2] - (τ/λ) * grad_u[i,j,2]) / (1 + (τ/λ) *
        grad_u_mag[i,j])

    return nothing
end
