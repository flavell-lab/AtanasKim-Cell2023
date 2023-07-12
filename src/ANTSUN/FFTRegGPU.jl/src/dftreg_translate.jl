function dftups(inp::AbstractArray{T,N},no,usfac::Int=1,offset=zeros(N)) where {T,N}
    sz = [size(inp)...]
    permV = 1:N
    for i in permV
        inp = permutedims(inp,[i;deleteat!(collect(permV),i)])
        kern = exp.((-1im*2*pi/(sz[i]*usfac))*((0:(no-1)).-offset[i])*transpose(ifftshift(0:(sz[i]-1)).-floor(sz[i]/2)))
        d = size(inp)[2:N]
        inp = kern * reshape(inp, Val(2))
        inp = reshape(inp,(no,d...))
    end
    permutedims(inp,collect(ndims(inp):-1:1))
end

function dftreg_gpu!(img1_f_g::CuArray{Complex{Float32},2}, img2_f_g::CuArray{Complex{Float32},2},
    CC_g::CuArray{Complex{Float32},2})
    L = length(img1_f_g)
    CC_g .= CUFFT.ifft(img1_f_g .* CUDA.conj(img2_f_g))
    loc = argmax(abs.(CC_g))
    CCmax = Array(CC_g)[loc]
    rfzero = sum(abs2, img1_f_g) / L
    rgzero = sum(abs2, img2_f_g) / L
    error = abs(1 - CCmax * conj(CCmax) / (rgzero * rfzero))
    diffphase = atan(imag(CCmax),real(CCmax))

    indi = size(img1_f_g)
    ind2 = tuple([div(x,2) for x in indi]...)

    locI = [Tuple(loc)...]

    shift = zeros(size(locI))
    for i = eachindex(locI)
        if locI[i]>ind2[i]
            shift[i]=locI[i]-indi[i]-1
        else shift[i]=locI[i]-1
        end
    end
    
    error, shift, diffphase
end

function dftreg_subpix_gpu!(img1_f_g::CuArray{Complex{Float32},2}, img2_f_g::CuArray{Complex{Float32},2},
    CC2x_g::CuArray{Complex{Float32},2}, up_fac::Int=10)
    ## initial estimate by 2x upsample
    # embed 2x fft
    dim_input = collect(size(img1_f_g))
    ranges = [(x+1-div(x,2)):(x+1+div(x-1,2)) for x in dim_input]
    CC2x_g .= 0
    CC2x_g[ranges...] .= CUFFT.fftshift(img1_f_g) .* CUFFT.conj(CUFFT.fftshift(img2_f_g))

    # compute cross-correlation and locate the peak
    CC2x_g = CUFFT.ifft(CUFFT.ifftshift(CC2x_g))
    loc = argmax(abs.(CC2x_g))

    indi = size(CC2x_g)
    locI = collect(Tuple(loc))
    CC2x_max = Array(CC2x_g)[loc]

    # obtain shift in original pixel grid
    ind2 = indi ./ 2
    shift = zeros(size(locI))
    for i = eachindex(locI)
        if locI[i] > ind2[i]
            shift[i] = locI[i] - indi[i] - 1
        else
            shift[i] = locI[i] - 1
        end
    end
    shift = shift / 2

    ## refine subpixel estimation
    if up_fac > 2
        # refine the estimate with matrix multiply DFT
        shift = round.(Int, shift * up_fac) / up_fac # initial shift estimate
        dft_shift = ceil(up_fac * 1.5) / 2 # center of output at dft_shift + 1

        # mat multiplies dft around the current shift estimate
        CC_refine = dftups(Array(img2_f_g .* CUFFT.conj(img1_f_g)), ceil(Int, up_fac * 1.5),
            up_fac, dft_shift .- shift * up_fac) / (prod(ind2) * up_fac ^ 2)

        # locate max and map back to the original grid
        loc = argmax(abs.(CC_refine))
        locI = Tuple(loc)
        CC_refine_max = CC_refine[loc]
        locI = locI .- dft_shift .- 1
        shift = shift .+ locI ./ up_fac
        
        img1_00 = dftups(Array(img1_f_g .* CUFFT.conj(img1_f_g)), 1, up_fac)[1] / (prod(ind2) * up_fac ^ 2)
        img2_00 = dftups(Array(img2_f_g .* CUFFT.conj(img2_f_g)), 1, up_fac)[1] / (prod(ind2) * up_fac ^ 2)
        CC_max = CC_refine_max
    else
        img1_00 = sum(img1_f_g .* CUFFT.conj(img1_f_g)) / prod(indi)
        img2_00 = sum(img2_f_g .* CUFFT.conj(img2_f_g)) / prod(indi)
        CC_max = CC2x_max
    end
    
    error = 1 - CC_max * conj(CC_max) / (img1_00 * img2_00)
    error = sqrt(abs.(error))
    diffphase = atan(imag(CC_max), real(CC_max))
    
    error, shift, diffphase
end

function subpix_shift_gpu!(img_f_g::CuArray{Complex{Float32},2}, N_g::CuArray{Float32,2}, shift, diffphase)
    sz = [size(img_f_g)...]
    N_ = Float32(0)
    for i = eachindex(sz)
        shifti = ifftshift((-div(sz[i],2)):(ceil(Integer,sz[i]/2)-1))*shift[i]/sz[i]
        resh = (repeat([1],inner=[i-1])...,length(shifti))
        N_ = N_ .- Float32.(reshape(shifti,resh))
    end
    
    copyto!(N_g, N_)
    exp(1im * diffphase) .* (img_f_g .* exp.(Complex{Float32}(2im * pi) * N_g))
end

function dftreg_resample_gpu!(img_f_g::CuArray{Complex{Float32}, 2}, N_g::CuArray{Float32,2}, shift, diffphase)
    real(CUFFT.ifft(subpix_shift_gpu!(img_f_g, N_g, shift, diffphase)))
end

function reg_stack_translate!(img_stack_reg_g::CuArray{Float32,3}, img1_f_g::CuArray{Complex{Float32},2},
        img2_f_g::CuArray{Complex{Float32},2}, CC2x_g::CuArray{Complex{Float32},2},
        N_g::CuArray{Float32,2}; reg_param::Dict)
    size_x, size_y, size_z = size(img_stack_reg_g)
    # reset arrays
    CC2x_g .= 0
    N_g .= 0
    
    # register
    for z = 2:size_z
        z1, z2 = z - 1, z
        # copy data and fft
        img1_g = view(img_stack_reg_g, :,:,z1)
        img2_g = view(img_stack_reg_g, :,:,z2)
        img1_f_g .= CUFFT.fft(img1_g)
        img2_f_g .= CUFFT.fft(img2_g)

        # register
        if !haskey(reg_param, z)
            error, shift, diffphase = dftreg_subpix_gpu!(img1_f_g, img2_f_g, CC2x_g)
            reg_param[z] = (error, shift, diffphase)
        else
            error, shift, diffphase = reg_param[z]
        end
        
        # resample
        img_stack_reg_g[:,:,z] .= dftreg_resample_gpu!(img2_f_g, N_g, shift, diffphase)
        
        # reset arrays
        CC2x_g .= 0
        N_g .= 0
    end
    
    nothing
end