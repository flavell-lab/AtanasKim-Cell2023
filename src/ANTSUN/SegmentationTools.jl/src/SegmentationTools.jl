module SegmentationTools

using FlavellBase, ImageDataIO, UNet2D, HDF5, Interact, NRRDIO, Distributions,
    StatsBase, LinearAlgebra, PyCall, ProgressMeter, DataStructures, Images, Plots,
    ImageSegmentation, WormFeatureDetector, ImageTransformations,
    CoordinateTransformations, StaticArrays, Interpolations, Rotations, SegmentationStats

include("init.jl")
include("find_head.jl")
include("unet_visualization.jl")
include("make_unet_input.jl")
include("semantic_segmentation.jl")
include("instance_segmentation.jl")
include("centroid_visualization.jl")
include("crop_worm.jl")

export
    find_head,
    find_head_unet,
    volume,
    instance_segmentation,
    consolidate_labeled_img,
    get_activity,
    create_weights,
    make_unet_input_h5,
    view_label_overlay,
    visualize_prediction_accuracy_2D,
    visualize_prediction_accuracy_3D,
    make_plot_grid,
    display_predictions_2D,
    display_predictions_3D,
    centroids_to_img,
    view_roi_2D,
    view_roi_3D,
    view_centroids_2D,
    view_centroids_3D,
    instance_segment_concave,
    get_points,
    distance,
    compute_mean_iou,
    detect_incorrect_merges,
    watershed_threshold,
    instance_segmentation_threshold,
    instance_segmentation_watershed,
    get_crop_rotate_param,
    crop_rotate,
    crop_rotate!,
    uncrop_img_roi,
    uncrop_img_rois,
    call_unet,
    get_neighbors,
    get_neighbors_diagonal
end # module
