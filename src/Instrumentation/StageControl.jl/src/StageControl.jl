module StageControl

using LibSerialPort

include("position.jl")
include("velocity.jl")
include("common.jl")

export set_zero,
    query_position,
    read_position,
    move_relative,
    # velocity.jl
    set_velocity,
    set_velocity_x,
    set_velocity_y,
    halt_stage,
    # common.jl
    flush_buffer,
    find_stage_port,
    check_baud_rate
end # module
