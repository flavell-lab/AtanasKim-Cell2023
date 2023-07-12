# FFTRegGPU.jl
Fast FFT GPU registrtion using [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation). This GPU version is based on [SubpixelRegistration.jl](https://github.com/romainFr/SubpixelRegistration.jl)  
Currently it only supports translation.

## Usage and example  
### Registering a set of 2D images
```julia
# allocate GPU memory
img1_g = CuArray{Float32}(undef, size_x, size_y)
img2_g = CuArray{Float32}(undef, size_x, size_y)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)

# copy to GPU
copyto!(img1_g, Float32.(img1))
copyto!(img2_g, Float32.(img2))

# perform FFT using CUFFT
img1_f_g .= CUFFT.fft(img1_g)
img2_f_g .= CUFFT.fft(img2_g)

# register (find the optimal translation)
error, shift, diffphase = dftreg_gpu!(img1_f_g, img2_f_g, CC_g)

# resample the moving image
img2_reg_g = dftreg_resample_gpu!(img1_f_g, N_g, shift, diffphase)

# copy to CPU
Array(img2_reg_g)
```
### Registering a set of 2D images (subpixel registration)
Use the function `dftreg_subpix_gpu!`. For the argument `CC2x_g`, the array size should be 2x of the image size.
```julia
# allocate GPU memory
img1_g = CuArray{Float32}(undef, size_x, size_y)
img2_g = CuArray{Float32}(undef, size_x, size_y)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)

# copy to GPU
copyto!(img1_g, Float32.(img1))
copyto!(img2_g, Float32.(img2))

# perform FFT using CUFFT
img1_f_g .= CUFFT.fft(img1_g)
img2_f_g .= CUFFT.fft(img2_g)

# register (find the optimal translation)
error, shift, diffphase = dftreg_subpix_gpu!(img1_f_g, img2_f_g, CC2x_g)

# resample the moving image
img2_reg_g = dftreg_resample_gpu!(img1_f_g, N_g, shift, diffphase)

# copy to CPU
Array(img2_reg_g)
```

### Registering z-stack
- Moving targets on the stage can cause shearing in z-stack. To correct this, the images within the stack are registered together.
- `reg_stack_translate!` is a memory-efficient and convenient function to register the frames in each z-stack. Here in the example, the script loads the z-stack at each time point, registers it, and then saves the registered z-stack.  
```julia
size_x, size_y, size_z = 256, 256, 94
img_stack_reg = zeros(Float32, size_x, size_y, size_z)
img_stack_reg_g = CuArray{Float32}(undef, size_x, size_y, size_z)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)

@showprogress for t = 1:100
    copyto!(img_stack_reg_g, get_zstack(t)) # copy data to GPU
    reg_stack_translate!(img_stack_reg_g, img1_f_g, img2_f_g, CC2x_g, N_g) # register
    copyto!(img_stack_reg, img_stack_reg_g) # copy result to GPU
    save_zstack(t, img_stack_reg)
end
```

## Performance
Measured time to call `reg_stack_translate!`  
Hardware: GTX 1080  
Julia v1.5.3 and CUDA.jl v.2.4.1

| size (x,y,z) | mean time (ms) |
| - | - |
| 32,32,32 | 43.817 ms |
| 64,64,64 | 91.215 ms |
| 128,128,64 | 103.022 ms |
| 256,256,64 | 131.046 ms |
| 256,256,256 | 525.257 ms |
