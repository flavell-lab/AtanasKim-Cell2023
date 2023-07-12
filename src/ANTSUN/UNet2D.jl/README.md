# UNet2D.jl
This is a Julia wrapper for https://github.com/flavell-lab/unet2d

## Instrunctions
### Model creation and evaluation
```julia
path_weights = ...
model = create_model(1, 1, 32, path_weights=path_weights)

img_test_single = rand(Float64, 1024, 1024)
img_test_single = Float32.(standardize(img_test_single))

eval_model(img_test_single, model)
```

### Batch evaluation
For batch evaluation, the input array dim should be (n, h, w):
```julia
img_test_batch = rand(Float64, 3, 1024, 1024)
img_test_batch = Float32.(standardize(img_test_batch))
eval_model(img_test_batch, model)
```

### CPU evaluation
The package uses "cuda:0" as default device. To evaluate using CPU or anyother device, create a torch device:
```julia
device_cpu = UNet2D.py_torch.device("cpu")
model = create_model(1, 1, 32, device=device_cpu)
eval_model(img_test_single, model, device=device_cpu)
```
