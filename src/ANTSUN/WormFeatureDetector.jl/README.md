# WormFeatureDetector.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/WormFeatureDetector.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/WormFeatureDetector.jl/dev/

A collection of heuristics for locating various features of the worm.

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, and `MHDIO.jl` packages from the `flavell-lab` github repository.
The example code provided here assumes the `FlavellBase` and `ImageDataIO` packages have been loaded in the current Julia environment.

## Worm curvature similarity heuristic

### Worm head location

In order to use this heuristic, it's necessary to determine the location of the worm's head.
The algorithm that finds the worm's head location does this by effectively turning the worm into a blob and finding an extremum of that blob. Note that this algorithm only works on data where neuron centroids have already been computed (for example, by the `SegmentationTools.jl` package).
Example code:

```julia
centroids = read_centroids_roi("/path/to/centroids")
img_size = size(read_img(MHD("/path/to/mhd")))
find_head(centroids, img_size)
```

### Using the worm curvature heuristic

This heuristic (implemented by the `elastix_difficulty_wormcurve` method) posits that two frames are similar to each other if the worm's curvature is similar, as this would result in a smaller amount of bending. It computes an estimate for the worm's centerline based on the images, and outputs its centerline fits as images which can be inspected for errors.

This heuristic requires data that has nuclear-localized fluorescent proteins in enough neurons to get an estimate of the worm shape, and has already been filtered (eg by the `GPUFilter.jl` package). It also requires you to have previously determined the worm's head location.

Example code, if you are using this heuristic with `RegistrationGraph.jl`:

```julia
curves = Dict()
heur = (rootpath, frame1, frame2) -> elastix_difficulty_wormcurve(rootpath, frame1, frame2, "MHD_filtered_cropped", "head_pos.txt", "img_prefix", 2, curves; figure_save_path="worm_curves")
```

## HSN and nerve ring location heuristic

### Finding HSN and the nerve ring

As a prerequisite to use this heuristic, HSN and the nerve ring must be located in each frame. You will likely need to modify the parameters on a per-dataset basis.

```julia
# initialize HSN heuristics to values that work on this dataset
threshold_hsn_outer=2.5
threshold_hsn_inner=3.5
density_hsn_outer=0.1
density_hsn_inner=0.7
radius_hsn_outer=[20,20,1]
radius_hsn_inner=[3,3,0]
radius_hsn_threshold=[5,5,3]

find_hsn("/path/to/data", 1:100, "MHD", "img_prefix", 1, threshold_hsn_outer, threshold_hsn_inner, density_hsn_outer, density_hsn_inner, radius_hsn_outer, radius_hsn_inner, radius_hsn_threshold; outfile="hsn_locs.txt")

# similarly, initialize nerve ring heuristics
threshold_nr=2.5
region_nr=[350:450, 1:70, 5:30]
radius_nr=[5,5,3]

find_nerve_ring("/path/to/data", 1:100, "MHD", "img_prefix", 1, threshold_nr, region_nr, radius_nr; outfile="nr_locs.txt")
```

### Using the HSN and nerve ring heuristic

This heuristic (implemented by the `elastix_difficulty_HSN_NR` method) tries to identify frames with similar HSN and nerve ring locations to be registered together. It only works on data taken with a non-nuclear-localized fluorescent protein expressed only in HSN, and it also requires you to have previously identified the HSN and nerve ring locations in each frame.

Example code, if you are using this heuristic with `RegistrationGraph.jl`:

```julia
heur = (rootpath, frame1, frame2) -> elastix_difficulty_hsn_nr(rootpath, frame1, frame2, "hsn_locs.txt", "nr_locs.txt", 1:100)
```
