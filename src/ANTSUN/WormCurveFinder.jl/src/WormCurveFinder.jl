module WormCurveFinder

using Statistics, LinearAlgebra, Images

include("interpolation.jl")
include("track.jl")
include("curve_finder.jl")

export track_curve,
    find_curve

end # module
