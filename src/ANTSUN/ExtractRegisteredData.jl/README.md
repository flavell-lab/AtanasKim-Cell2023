# ExtractRegisteredData.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/ExtractRegisteredData.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/ExtractRegisteredData.jl/dev/

This package matches neurons across time points using image registration, and then extracts activity channel (GCaMP) traces of the detected neurons.

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, `CaSegmentation.jl`, `Clustering.jl`, and `SegmentationTools.jl` packages. Note that you must use the Flavell Lab fork of `Clustering.jl` rather than the original Julia package.
It also requires you to have [installed `transformix`](https://simpleelastix.readthedocs.io/GettingStarted.html#manually-building-on-linux).


## Neuron matching

To match ROIs between frames - and thereby identify neurons - the `extract_roi_overlap` function runs `transformix` according to the registration parameters. It uses this to match the ROIs between registered frames together and finds overlaps between ROIs. Assuming you've previously registered the dataset and obtained marker channel activity, you can run:

```julia
roi_overlaps, roi_activity_diff, overlap_errors = extract_roi_overlap(best_reg, param_path, param);
```

We can then make a matrix of all ROIs across all frames, whose `(i,j)`th entry is the quality of the match between ROIs `i` and `j`. The match quality is determined by a number of heuristic parameters, including ROI overlap fraction, ROI signal difference in the marker channel, and registration quality. There is also `label_map` which is a dictionary mapping frames and original ROIs to these new matrix-index ROIs.

From here, we can cluster the matrix to get a map of ROIs to neurons, and vice versa

```julia
regmap_matrix, label_map = make_regmap_matrix(roi_overlaps, roi_activity_diff, q_dict, best_reg, param)

new_label_map, inv_map, hmer = find_neurons(regmap_matrix, label_map, param)
```

## Extracting traces

Once we have the neuron identity maps, we can extract traces, which will be stored as a dictionary mapping neurons to dictionaries mapping time points to the neural activity. Note that many of the traces will be for neurons that were only detected in a very small number of frames - this usually happens if a neuron is difficult to register, or is inconsistently detected by the UNet. It is recommended to threshold to remove neurons that are not present in many frames. See the `CaAnalysis.jl` package for tools to process the traces data.

```julia
traces, traces_errors = extract_traces(inv_map, "/path/to/gcamp/activities")
```
