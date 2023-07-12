function set_velocity_x(s::SerialPort, speed_x::Int32)
    cmd = zeros(UInt8, 13)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage x
    cmd[3] = 0x41 # command 65
    cmd[4] = 0x00
    cmd[5] = 0x0a # index 10 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00
    cmd[9] = speed_x & 0x000000ff
    cmd[10] = (speed_x & 0x0000ff00) >> 8
    cmd[11] = (speed_x & 0x00ff0000) >> 16
    cmd[12] = (speed_x & 0xff000000) >> 24
    cmd[13] = 0x0D # end

    check_bytes_written(write(s, cmd), 13)
end

function set_velocity_y(s::SerialPort, speed_x::Int32)
    cmd = zeros(UInt8, 13)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x02 # stage y
    cmd[3] = 0x41 # command 65
    cmd[4] = 0x00
    cmd[5] = 0x0a # index 10 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00
    cmd[9] = speed_x & 0x000000ff
    cmd[10] = (speed_x & 0x0000ff00) >> 8
    cmd[11] = (speed_x & 0x00ff0000) >> 16
    cmd[12] = (speed_x & 0xff000000) >> 24
    cmd[13] = 0x0D # end

    check_bytes_written(write(s, cmd), 13)
end

function set_velocity(s::SerialPort, speed_x::Int32, speed_y::Int32)
    cmd = zeros(UInt8, 26)
    # stage x
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage x
    cmd[3] = 0x41 # command 65
    cmd[4] = 0x00
    cmd[5] = 0x0a # index 10 for the command
    cmd[6] = 0x00
    cmd[7] = 0x04 # number of bytes for data (8-11)
    cmd[8] = 0x00
    cmd[9] = speed_x & 0x000000ff
    cmd[10] = (speed_x & 0x0000ff00) >> 8
    cmd[11] = (speed_x & 0x00ff0000) >> 16
    cmd[12] = (speed_x & 0xff000000) >> 24
    cmd[13] = 0x0D # end

    # stage y
    cmd[13+1] = 0x23 # CAN comamnd marker
    cmd[13+2] = 0x02 # stage y
    cmd[13+3] = 0x41 # command 65
    cmd[13+4] = 0x00
    cmd[13+5] = 0x0a # index 10 for the command
    cmd[13+6] = 0x00
    cmd[13+7] = 0x04 # number of bytes for data (8-11)
    cmd[13+8] = 0x00
    cmd[13+9] = speed_y & 0x000000ff
    cmd[13+10] = (speed_y & 0x0000ff00) >> 8
    cmd[13+11] = (speed_y & 0x00ff0000) >> 16
    cmd[13+12] = (speed_y & 0xff000000) >> 24
    cmd[13+13] = 0x0D # end

    check_bytes_written(write(s, cmd), 26)
end

function halt_stage(s::SerialPort)
    cmd = zeros(UInt8, 9)
    cmd[1] = 0x23 # CAN comamnd marker
    cmd[2] = 0x01 # stage 1
    cmd[3] = 0x42 # command 66
    cmd[4] = 0x00
    cmd[5] = 0x01 # index 1 for the command
    cmd[6] = 0x00
    cmd[9] = 0x0D # end
    check_bytes_written(write(s, cmd), 9)
    cmd[2] = 0x02 # stage 2
    check_bytes_written(write(s, cmd), 9)
end
