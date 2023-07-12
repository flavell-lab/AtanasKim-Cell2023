# RegistrationGraph.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/RegistrationGraph.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/RegistrationGraph.jl/dev/


A collection of tools for running elastix registration. Suppose there is a set of N frames to be registered, but the frames are too dissimilar to be able to ensure a quality registration for each pair of frames. In this case, it is often helpful to generate a heuristic to evaluate how similar two frames are to each other (and hence, how likely elastix will be to succeed). Once such a heuristic can be determined, registration problems can be selected to minimize difficulty and maximum path length, as a graph optimization problem. This package provides several heuristics for various registration problems, graph theory solutions for constructing the registration problem graph, and automated syncing of scripts and data to the OpenMind server.


## Prerequisites

- This package requires you to have previously installed the `FlavellBase.jl`, `MHDIO.jl`,  `ImageDataIO.jl`, `WormCurveFinder.jl`, `WormFeatureDetector.jl`, `SegmentationTools.jl`, and `SLURMManager.jl` packages from the `flavell-lab` github repository (in that order).
- The example code provided here assumes the `FlavellBase` and `ImageDataIO` packages have been loaded in the current Julia environment.
- [Set up an OpenMind account](https://github.mit.edu/MGHPCC/openmind/wiki/Cookbook:-Getting-started)
- Ensure that you are using a Unix-based shell. This comes by default on Mac and Linux systems, but on Windows, it is recommended to install Ubuntu (Windows Subsystem for Linux) through the Windows Store and run the code from the Ubuntu terminal.
- Install the `rsync` program if it isn't installed already
- [Set up `ssh` keys into OpenMind.](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2) Leave the passphrase blank, and don't do step 4.
- Install Anaconda3 on OpenMind and install the `h5py`, `SimpleITK`, `scipy`, and `skimage` packages. (I have a copy at `/om/user/aaatanas/anaconda3` with packages set up if you just want to copy it.)
- Install Julia on OpenMind and install the `SLURMManager.jl` package inside it. Ensure that Julia is configured to be recognized in your default `ssh` shell.

## Creating an elastix difficulty file from frame difference heuristics

The package `WormFeatureDetector.jl` contains a variety of heuristics that can be used to assess the difference between two frames; consult that package to find a heuristic appropriate for your situations. Once you've identified such a heuristic, you will need to condense it down to a function, giving it all the parameters in advance:

```julia
# give the heuristic all its parameters in advance, so it's a function of just three arguments
heur! = (t1, t2) -> heuristic(t1, t2, params...)
# now compute elastix difficulty
generate_elastix_difficulty("/path/to/elx_difficulty_output_file", 1:100, heur!)
```

## Generating a registration problem graph

As a prerequisite for this step, it is assumed that you have a heuristic for elastix registration difficulty between frames, which has been used to generate an elastix difficutly file of the appropriate format. If you're using one of the heuristics above, the formatting will already have been done; otherwise, the correct format is a comma-separated list of frames for which you have the difficulty, followed by a space/newline-separated matrix whose [i,j]th entry is the heuristic dissimilarity of frame i to frame j. Both the list of frames and the matrix are allowed to use the `[` and `]` characters, which will be ignored in the parsing. Furthermore, the matrix is allowed to be triangular, and not all frames need be included in the heuristic calculation, as long as the indexing of the images is consistent with the indexing of the matrix. The frame numbers must be integers. Example for a 7-frame video of which 3 frames were discarded:

```text
[1,3,5,7]
[0.0 5.0 7.0 8.0
 0.0 0.0 1.0 6.0
 0.0 0.0 0.0 10.0
 0.0 0.0 0.0 0.0]
```

Suppose that data was saved in a file `elastix_difficulty.txt`. Then you can perform the following actions to generate a registration graph from it.

```julia
# loads the file into a graph
# you may want to modify the difficulty_importance parameter
# see documentation for load_graph for more details
graph = load_graph("/path/to/data/elastix_difficulty.txt")

# after generating the difficulty file, you decided that frame 3 has bad data
# so you want to prevent it from being part of the registration problems
graph = remove_frame(graph, 3)

# generates set of registration problems from the graph
# let's say we want at least 10 edges from each node
subgraph = make_voting_subgraph(graph, 10)
```

Once you're satisfied with the graph, you can save it:

```julia
# saves subgraph to a text file containing a list of edges
output_graph(subgraph, "/path/to/data/registration_problems.txt")
```

## Running elastix on OpenMind

### Configuring the server to run elastix

This is implemented by the `write_sbatch_graph` function, which syncs all the data from the local computer to the OpenMind server and generates files that will run elastix through `sbatch`. There are many keyword arguments and parameter settings you can use to control how the registration runs. Example code:

```julia
# loads registration problems you previously computed from the graph
problems = load_registration_problems(["/path/to/data/registration_problems.txt"])

# syncs data to server and generates sbatch files for elastix
write_sbatch_graph(problems, param_path, param_path, param)

# runs elastix on OpenMind
run_elastix_openmind(param_path, param)

# waits for elastix to finish running
wait_for_elastix(param)

# syncs data back from server
sync_registered_data(param_path, param)

# elastix has its transform paramete files use absolute paths - these need to be converted to the path on your machine
fix_param_paths(problems, param_path, param)
```

## Checking elastix quality

After running elastix on a set of registration problems, it is likely that many, but not all, of them will succeed. By running a quality metric, the algorithm's perceived difficulty of registration pairs can be modified based on how well elastix performed in each instance. This allows the optimal resolution to be selected, and helps subsequent algorithms evaluate how strongly to weight each registration.

The `make_quality_dict` function takes as input a list of metrics, and evaluates them on all resolutions of the registration. Note that smaller values are better (0 is perfect registration). Example code:

```julia
# initialize dictionary of functions
# usually, metrics will require other parameters, you will need to specify them here
# because all input functions must have exactly the parameters moving, fixed, resolution
# if you're using a mask, the keyword parameter mask_dir will also be provided to the function
evaluation_functions = Dict()
evaluation_functions[param["quality_metric"]] = (moving, fixed, resolution) -> metric_tfm(calculate_ncc(read_img(NRRD(joinpath(path_fixed, param_path["get_basename"](fixed, param["ch_marker"])*".nrrd"))), read_img(NRRD(joinpath(param_path["path_dir_reg"], "$(moving)to$(fixed)", "result.$(resolution[1]).R$(resolution[2]).nrrd")))), threshold=param["reg_quality_threshold"])


# now, compute and output quality dictionary
q_dict, best_reg, q_dict_errors = make_quality_dict(param_path, param, problems, evaluation_functions)
```

## Visualizing registration

You can directly compare an image to a registration-mapped image, together with their ROIs:

```julia
# previously, load fixed and moving images and ROIs
visualize_roi_predictions(fixed_image_rois, moving_image_rois, fixed_image, moving_image)
```

If you've already run an algorithm to match ROIs between the two images, you can set the keyword variable
`roi_match` to display the match on the plot as well.
