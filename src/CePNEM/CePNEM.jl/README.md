# CePNEM.jl
Model of neural encoding of behavior, together with `Gen.jl` tools to fit it.

## Usage  
Here's an example to fit the latest model (`nl10d`).

Example only: the data loading parts are examples and should be modified for actual computation.
```julia
using Gen, Statistics, StatsBase, HDF5, FlavellBase, EncoderModelGen, ANTSUNData

rg_t = 401:800 # time point index to fit
n_obs = length(rg_t)

# load trace and behavior
trace = load_trace() # load neural trace
ys = trace[rg_t]

v = velocity[rg_t]
θh = θh[rg_t]
P = pumping[rg_t]

path_h5_result = "..." # path of the HDF5 file to save the result

ret = run_mcmc_10(ys, v, θh, P; n_init=100000, n_iters=11000, lr_adjust=1.1, model=:nl10d)

h5open(path_h5_result, "w") do h5f
    write(h5f, "sampled_trace_params", nl10d_traces_to_params(ret[1]))
    write(h5f, "accept", ret[2])
    write(h5f, "δ_vals", ret[3])
end
```
