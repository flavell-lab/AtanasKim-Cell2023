function derivative(x::Array{T,1}, y::Array{T,1}, nu=1) where T
    spl = Spline1D(x, y, k=3) # spline order 3
    Dierckx.derivative(spl, x, nu=nu)
end

function derivative(y::Array{T,1}, nu=1) where T
    x = 1:length(y)
    spl = Spline1D(x, y, k=3)
    Dierckx.derivative(spl, x, nu=nu)
end

function integrate(y::Array{T,1}) where T
    x = 1:length(y)
    spl = Spline1D(x, y, k=3) # spline order 3
    y_int = zero(y)
    for i = 1:length(y)
        y_int[i] = Dierckx.integrate(spl, 1, x[i])
    end

    y_int
end

function integrate(Y::Array{T,2}) where T
    Y_int = zero(Y)
    for i = 1:size(Y, 1)
        Y_int[i,:] = integrate(Y[i,:])
    end

    Y_int
end

function standardize(f::Array{T,2}) where T
    σ = std(f, dims=2)
    μ = mean(f, dims=2)
    (f .- μ) ./ σ
end

function derivative(f::Array{T,2}, nu=1) where T
    f_grad = zero(f)
    n_unit = size(f, 1)

    for i = 1:n_unit
        f_grad[i, :] = derivative(f[i, :], nu)
    end

    f_grad
end

function discrete_diff(f::Array{T,2}) where T
    diff(f, dims=2)
end

function chain_process(f::Array, g_list)
    f_proc = f

    for g = g_list
        f_proc = g(f_proc)
    end

    f_proc
end
