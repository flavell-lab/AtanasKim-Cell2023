# CePNEM.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/CePNEM.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/CePNEM.jl/dev/ 

The *C. elegans* Probabilistic Neural Encoding Model, together with `Gen.jl` tools to fit it.

## Citation

Brain-wide representations of behavior spanning multiple timescales and states in *C. elegans*

Adam A. Atanas*, Jungsoo Kim*, Ziyu Wang, Eric Bueno, McCoy Becker, Di Kang, Jungyeon Park, Cassi Estrem, Talya S. Kramer, Saba Baskoylu, Vikash K. Mansinghka, Steven W. Flavell

bioRxiv 2022.11.11.516186; doi: https://doi.org/10.1101/2022.11.11.516186

\* equal contribution

## Usage  
Here's an example to fit the latest model (`nl10d`).

Example only: the data loading parts are examples and should be modified for actual computation.
```julia
using Gen, Statistics, StatsBase, HDF5, FlavellBase, CePNEM, ANTSUNData
Gen.@load_generated_functions

rg_t = 1:800 # time point index to fit
n_obs = length(rg_t)

# load trace and behavior
ys = trace[rg_t] # neural trace

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
