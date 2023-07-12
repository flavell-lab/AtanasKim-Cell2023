# NRRDIO.jl
Note: currently this package only supports reading/writing compressed data.

## Writing
Example:
```julia
write_nrrd(path_nrrd, test_data, (0.54,0.54,0.54)) # path, array, spacing
```

## Reading
Example:
```julia
nrrd = NRRD(path_nrrd)
img_read = read_img(nrrd)
```
