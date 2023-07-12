"""
    set_zero(s::SerialPort)
Sets the current x, y position to 0, 0
"""
function set_zero(s::SerialPort, stage_num::UInt8)
    cmd = zeros(UInt8, 13)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = stage_num # stage #
    cmd[3] = 0x53 # command 83
    cmd[4] = 0x00
    cmd[5] = 0x05 # index 5 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00 
    cmd[9] = 0x00
    cmd[10] = 0x00
    cmd[11] = 0x00
    cmd[12] = 0x00
    cmd[13] = 0x0D # end

    check_bytes_written(write(s, cmd), 13)
end

function set_zero(s::SerialPort)
    cmd = zeros(UInt8, 26)
    # stage x
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage #
    cmd[3] = 0x53 # command 83
    cmd[4] = 0x00
    cmd[5] = 0x05 # index 5 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00 
    cmd[9] = 0x00
    cmd[10] = 0x00
    cmd[11] = 0x00
    cmd[12] = 0x00
    cmd[13] = 0x0D # end

    # stage y
    cmd[1+13] = 0x23 # CAN comamnd marker
    cmd[2+13] = 0x02 # stage #
    cmd[3+13] = 0x53 # command 83
    cmd[4+13] = 0x00
    cmd[5+13] = 0x05 # index 5 for the command
    cmd[6+13] = 0x00
    cmd[7+13] = 0x04 # number of bytes for data (8-11)
    cmd[8+13] = 0x00 
    cmd[9+13] = 0x00
    cmd[10+13] = 0x00
    cmd[11+13] = 0x00
    cmd[12+13] = 0x00
    cmd[13+13] = 0x0D # end
    
    check_bytes_written(write(s, cmd), 26)
end

"""
    move_relative(s::SerialPort, stage_num::UInt8, d::Int32)
Moves a stage by stage unit (10000 / 1 mm)
"""
function c(s::SerialPort, stage_num::UInt8, d::Int32)

    cmd = zeros(UInt8, 13)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = stage_num # stage
    cmd[3] = 0x41 # command 65
    cmd[4] = 0x00
    cmd[5] = 0x09 # index 9 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00 # number of bytes (MSB)
    cmd[9] = d & 0x000000ff
    cmd[10] = (d & 0x0000ff00) >> 8
    cmd[11] = (d & 0x00ff0000) >> 16
    cmd[12] = (d & 0xff000000) >> 24
    cmd[13] = 0x0D # end

    check_bytes_written(write(s, cmd), 13)
end

"""
    move_relative(s::SerialPort, dx::Int32, dy::Int32)
Moves x,y stages by stage unit (10000 / 1 mm)
"""
function move_relative(s::SerialPort, dx::Int32, dy::Int32)

    cmd = zeros(UInt8, 26)
    # stage x
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage 1
    cmd[3] = 0x41 # command 65
    cmd[4] = 0x00
    cmd[5] = 0x09 # index 9 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00 # number of bytes (MSB)
    cmd[9] = dx & 0x000000ff
    cmd[10] = (dx & 0x0000ff00) >> 8
    cmd[11] = (dx & 0x00ff0000) >> 16
    cmd[12] = (dx & 0xff000000) >> 24
    cmd[13] = 0x0D # end

    # stage y
    cmd[1+13] = 0x23 # CAN comamnd marker
    cmd[2+13] = 0x02 # stage 2
    cmd[3+13] = 0x41 # command 65
    cmd[4+13] = 0x00
    cmd[5+13] = 0x09 # index 9 for the command
    cmd[6+13] = 0x00
    cmd[7+13] = 0x04 # number of bytes for data (8-11)
    cmd[8+13] = 0x00 # number of bytes (MSB)
    cmd[9+13] = dy & 0x000000ff
    cmd[10+13] = (dy & 0x0000ff00) >> 8
    cmd[11+13] = (dy & 0x00ff0000) >> 16
    cmd[12+13] = (dy & 0xff000000) >> 24
    cmd[13+13] = 0x0D # end

    check_bytes_written(write(s, cmd), 26)
end

"""
    query_position(s::SerialPort, stage_num::UInt8)
Writes a command to get the position of a stage
Position is 20000 / 1 mm
"""
function query_position(s::SerialPort, stage_num::UInt8)
    cmd = zeros(UInt8, 9)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = stage_num # stage #
    cmd[3] = 0x54 # command 84
    cmd[4] = 0x00
    cmd[5] = 0x05 # index 5 for the command
    cmd[6] = 0x00
    cmd[7] = 0x00 # number of bytes for data (8-11)
    cmd[8] = 0x00
    cmd[9] = 0x0D # end

    check_bytes_written(write(s, cmd), 9)
end

"""
    query_position(s::SerialPort)
Writes a command to get positions from both stages
Position is 20000 / 1 mm
"""
function query_position(s::SerialPort)
    cmd = zeros(UInt8, 18)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage #
    cmd[3] = 0x54 # command 84
    cmd[4] = 0x00
    cmd[5] = 0x05 # index 5 for the command
    cmd[6] = 0x00
    cmd[7] = 0x00 # number of bytes for data (8-11)
    cmd[8] = 0x00
    cmd[9] = 0x0D # end
    cmd[9+1] = 0x23 # CAN comamnd marker
    cmd[9+2] = 0x02 # stage #
    cmd[9+3] = 0x54 # command 84
    cmd[9+4] = 0x00
    cmd[9+5] = 0x05 # index 5 for the command
    cmd[9+6] = 0x00
    cmd[9+7] = 0x00 # number of bytes for data (8-11)
    cmd[9+8] = 0x00
    cmd[9+9] = 0x0D # end

    check_bytes_written(write(s, cmd), 18)
end

function read_position(s::SerialPort, stage_num::UInt8)
    read_bytes = bytesavailable(s)
    read_data = read(s, read_bytes)

    @assert read_data[1] == 0x23
    @assert read_data[2] == stage_num
    @assert read_data[3] == 128+84

    int_pos = Int32(read_data[12]) << 24
    int_pos += Int32(read_data[11]) << 16
    int_pos += Int32(read_data[10]) << 8
    int_pos += Int32(read_data[9])

    int_pos
end

function read_position(s::SerialPort)
    read_bytes = bytesavailable(s)
    read_data = read(s, read_bytes)

    @assert read_data[1] == 0x23
    @assert read_data[2] == 0x01 # stage 1, x
    @assert read_data[3] == 128+84
    @assert read_data[13+1] == 0x23
    @assert read_data[13+2] == 0x02 # stage 2, y
    @assert read_data[13+3] == 128+84

    int_pos_x = Int32(read_data[12]) << 24
    int_pos_x += Int32(read_data[11]) << 16
    int_pos_x += Int32(read_data[10]) << 8
    int_pos_x += Int32(read_data[9])

    int_pos_y = Int32(read_data[13+12]) << 24
    int_pos_y += Int32(read_data[13+11]) << 16
    int_pos_y += Int32(read_data[13+10]) << 8
    int_pos_y += Int32(read_data[13+9])

    int_pos_x, int_pos_y
end

# """
#     move_absolute(s::SerialPort, x::Int, y::Int)
# Move (absolute) the stage position
# """
# function move_absolute(s::SerialPort, x::Int, y::Int)
#     sp_return = write(s, "MOVE X=$x Y=$y\r")
#     check_sp_return(sp_return)

#     nothing
# end

# """
#     get_position(s::SerialPort)
# Returns the current x, y position in Int64
# """
# function get_position(s::SerialPort)
#     sp_return = write(s, "WHERE X Y\r")
#     if sp_return != SP_OK
#         error("Serial port error: $sp_return")
#     end

#     sleep(0.017) # wait until available
#     read_bytes = bytesavailable(s)
#     if read_bytes == 0
#         error("0 bytes to read")
#     end

#     return_str = String(read(s, read_bytes))
#     if !startswith(return_str, ":A")
#         error("WHERE X Y command returned error")
#     end

#     return parse.(Int, split(strip(return_str[3:end-1])))
# end
