# ANTSUNData.jl
## Usage
## Data convert/export
To convert the ANTSUN JLD2 to HDF5, use the following functions
```julia
export_jld2_h5(path_data_dict; path_h5=path_h5)
```
For the info on the arguments, check the [docstring](https://github.com/flavell-lab/ANTSUNData.jl/blob/6c941055c35c64ffb29be2dfcb86a7122830f840/src/data_h5.jl#L100) of the function.

### Data import
To import the converted HDF5 ANTSUNData afile,
```julia
import_data(path_h5)
```

For the info on the arguments, check the [docstring](https://github.com/flavell-lab/ANTSUNData.jl/blob/6c941055c35c64ffb29be2dfcb86a7122830f840/src/data_h5.jl#L3) of the function.
