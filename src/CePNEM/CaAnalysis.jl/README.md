# CaAnalysis.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/CaAnalysis.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/CaAnalysis.jl/dev/

## Usage

The `process_traces` method can process raw GCaMP and marker traces output from the `ExtractRegisteredData.jl` package and perform a series of operations on them, eg:

```julia
data_dicts[dataset]["traces_array"], data_dicts[dataset]["traces_array_F_F20"], data_dicts[dataset]["raw_zscored_traces_array"],
            data_dicts[dataset]["valid_rois"], data_dicts[dataset]["bleach_param"], data_dicts[dataset]["bleach_curve"], data_dicts[dataset]["bleach_resid"] = 
            process_traces(params[dataset], data_dicts[dataset]["activity_traces"], data_dicts[dataset]["marker_traces"],
                        params[dataset]["num_detections_threshold"], 1:params[dataset]["max_t"], min_intensity=0, normalize_fn=mean,
                        activity_bkg=data_dicts[dataset]["activity_bkg"], marker_bkg=data_dicts[dataset]["marker_bkg"],
                        denoise=false, bleach_corr=true, divide=true, interpolate=true);
```
