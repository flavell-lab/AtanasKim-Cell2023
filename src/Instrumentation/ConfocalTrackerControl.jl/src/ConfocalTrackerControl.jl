module ConfocalTrackerControl

using QML, Spinnaker, PyCall, LinearAlgebra, Statistics, CxxWrap,
    StageControl, LibSerialPort, Dates, DataStructures, HDF5,
    NIDAQ, ProgressMeter, Qt5QuickControls2_jll
    
include("constant.jl")
include("unit.jl")
include("data.jl")
include("gui.jl")
include("gui_loop.jl")
include("track.jl")
include("nidaq.jl")
include("cam.jl")
include("init.jl")

export loop_main,
    stop_loop_main,
    save_h5,
    cam_alignment,
    cam_default,
    cam_adjust_exposure,
    cam_adjust_gain

end # module
