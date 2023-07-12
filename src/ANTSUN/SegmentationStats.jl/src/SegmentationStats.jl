module SegmentationStats
include("label_component.jl")
include("segmented_instance.jl")
include("stats.jl")

export mark_search_pix!,
    label_2d!,
    # stats
    moment,
    centroid,
    get_centroids,
    get_centroids_round,
    weighted_moment,
    weighted_centroid,
    # segmented_instance
    get_segmented_instance

end # module
