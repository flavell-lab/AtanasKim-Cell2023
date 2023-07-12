const py_torch = PyNULL()
const py_unet2d = PyNULL()
const torch_device = PyNULL()
const DEFAULT_DEVICE = "cuda:0"

function __init__()
    # py packages
    copy!(py_torch, pyimport("torch"))
    copy!(py_unet2d, pyimport("unet2d"))
    copy!(torch_device, py_torch.device(DEFAULT_DEVICE))
end