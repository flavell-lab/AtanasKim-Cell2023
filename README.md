## Intro
This repository contains the source codes used in [the article below](https://github.com/flavell-lab/AtanasKim-Cell2023/tree/main#citation).

## Citation
To cite this work (datasets, methods, code, packages, models, results, etc.), please refer to the article: 

#### Brain-wide representations of behavior spanning multiple timescales and states in C. elegans
**Adam A. Atanas\***, **Jungsoo Kim\***, Ziyu Wang, Eric Bueno, McCoy Becker, Di Kang, Jungyeon Park, Cassi Estrem, Talya S. Kramer, Saba Baskoylu, Vikash K. Mansingkha, Steven W. Flavell  
bioRxiv 2022.11.11.516186; doi: https://doi.org/10.1101/2022.11.11.516186  

**\* Equal Contribution**

## Data files
The processed data files and the trained neural network weights are available in [the repository](https://doi.org/10.5281/zenodo.8150515).

## WormWideWeb
The datasets and modeling results (encoding detection) from this project can be browsed in [WormWideWeb](https://wormwideweb.org/). On the website, you can easily find the neural (GCaMP) traces of specific recorded/identified neuron classes, along with the information on how those neurons encode behavioral information—identified by the CePNEM model.

## Notebooks and scripts
Notebooks/scripts used in this project are stored in the directory `notebook`.  

## Package directory
List of packages in this repository (in the directory `src`). Note that the source code included in this repository is from when the article was published. Any updates (e.g. bug fix, new feature) to any of the packages are not reflected on the code included in this repository.  
**To use/install or access the latest/maintained versions, please access the individual repositories** using the links below: 

### ANTSUN
Calcium and behavior data processing and extraction

 - [ANTSUN](https://github.com/flavell-lab/ANTSUN) contains a notebook that executes the full ANTSUN pipeline, from raw microscope data to neural traces and behaviors
 - [BehaviorDataNIR.jl](https://github.com/flavell-lab/BehaviorDataNIR.jl) extracts behavioral data from NIR microscope recordings
 - [CaAnalysis.jl](https://github.com/flavell-lab/CaAnalysis.jl) processes GCaMP traces
 - [Clustering.jl](https://github.com/flavell-lab/Clustering.jl) contains a custom hierarchical clustering algorithm used in neuron identity matching
 - [ExtractRegisteredData.jl](https://github.com/flavell-lab/ExtractRegisteredData.jl) links neural identities between time points and extracting GCaMP traces
 - [FFTRegGPU.jl](https://github.com/flavell-lab/FFTRegGPU.jl) performs shear-correction in GPU
 - [FlavellBase.jl](https://github.com/flavell-lab/FlavellBase.jl) contains generic code utilities
 - [GPUFilter.jl](https://github.com/flavell-lab/GPUFilter.jl) library of GPU-accelerated filtering methods
 - [ImageDataIO.jl](https://github.com/flavell-lab/ImageDataIO.jl) contains IO code that allows different ANTSUN packages to communicate with each other
 - [MHDIO.jl](https://github.com/flavell-lab/MHDIO.jl) contains IO code for the MHD image format
 - [ND2Process.jl](https://github.com/flavell-lab/ND2Process.jl) contains IO code for the ND2 image format
 - [NRRDIO.jl](https://github.com/flavell-lab/NRRDIO.jl) contains IO code for the NRRD image format
 - [NeuroPALData.jl](https://github.com/flavell-lab/NeuroPALData.jl) contains a collection of NeuroPAL labeling utilities
 - [RegistrationGraph.jl](https://github.com/flavell-lab/RegistrationGraph.jl) generates and executes registration problems to map neural identities between time points
 - [SLURMManager.jl](https://github.com/flavell-lab/SLURMManager.jl) interacts with `SLURM` to execute code on a cluster
 - [SegmentationStats.jl](https://github.com/flavell-lab/SegmentationStats.jl) contains segmentation utlility code
 - [SegmentationTools.jl](https://github.com/flavell-lab/SegmentationTools.jl) performs semantic and instance segmentation using the 3D-UNet `pytorch-3dunet`, and can generate and format UNet input and training data
 - [TotalVariation.jl](https://github.com/flavell-lab/TotalVariation.jl) performs total-variation noise filtering
 - [UNet2D.jl](https://github.com/flavell-lab/UNet2D.jl) implements a Julia wrapper for `unet2d`
 - [WormCurveFinder.jl](https://github.com/flavell-lab/WormCurveFinder.jl) fits a spline to the confocal imaging data
 - [WormFeatureDetector.jl](https://github.com/flavell-lab/WormFeatureDetector.jl) computes metrics of how similar worm postures are in different time points
 - [pytorch-3dunet](https://github.com/flavell-lab/pytorch-3dunet) implements the 3D-UNet for neuron segmentation
 - [unet2d](https://github.com/flavell-lab/unet2d) implements 2D-UNets, used for worm segmentation and head detection
### CePNEM
Modeling and analysis tools
 - [ANTSUNData.jl](https://github.com/flavell-lab/ANTSUNData.jl) implements IO operations on HDF5 ANTSUN output
 - [ANTSUNDataJLD2.jl](https://github.com/flavell-lab/ANTSUNDataJLD2.jl) converts ANTSUN output into the HDF5 format
 - [AnalysisBase.jl](https://github.com/flavell-lab/AnalysisBase.jl) implements various GCaMP analysis utilities
 - [CePNEM.jl](https://github.com/flavell-lab/CePNEM.jl) contains the mathematical formulation of the CePNEM model
 - [CePNEMAnalysis.jl](https://github.com/flavell-lab/CePNEMAnalysis.jl) and its associated notebooks contain most of the code for interpreting CePNEM model results
 - [EncoderModel.jl](https://github.com/flavell-lab/EncoderModel.jl) implements non-probabilistic (i.e.: MSE-minimization) versions of CePNEM
 - [FlavellConstants.jl](https://github.com/flavell-lab/FlavellConstants.jl) stores parameter settings and values that are held fixed
 - [HierarchicalPosteriorModel.jl](https://github.com/flavell-lab/HierarchicalPosteriorModel.jl) implements a hierarchical posterior model for interpreting CePNEM fits on the same neuron class across animals
### Instrumentation
Microscope control and data acquisition software (GUI)
 - [ConfocalTrackerControl.jl](https://github.com/flavell-lab/ConfocalTrackerControl.jl) worm tracking and data acquisition software (GUI)
 - [StageControl.jl](https://github.com/flavell-lab/StageControl.jl) interfaces the MAC6000 stage controller
