module WormFeatureDetector

using WormCurveFinder, NRRDIO, Statistics, ProgressMeter, Images,
    ImageTransformations, HDF5, ImageSegmentation, LinearAlgebra,
    CoordinateTransformations, StaticArrays, ImageDataIO, FlavellBase,
    Interpolations, Rotations, Plots

include("worm_feature_detector.jl")
include("worm_curve_finder.jl")
include("heuristics.jl")

export
    find_gut_granules,
    find_hsn,
    find_nerve_ring,
    in_conv_hull,
    curve_distance,
    elastix_difficulty_hsn_nr,
    elastix_difficulty_wormcurve!
end # module
