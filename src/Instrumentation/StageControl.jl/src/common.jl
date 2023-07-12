function check_bytes_written(b_actual, b_target)
    b_actual != b_target &&
        error("$b_actual bytes were written (should be $b_target bytes)")

    nothing
end

function flush_buffer(s::SerialPort)
    sp_flush(s.ref, SP_BUF_BOTH)

    read_bytes = bytesavailable(s)
    if read_bytes != 0
        error("Buffer not cleard")
    end

    nothing
end

function get_port_list_info(;nports_guess::Integer=64)
    ports = LibSerialPort.sp_list_ports()
    list_port = String[]
    list_desc = String[]

    for port in unsafe_wrap(Array, ports, nports_guess, own=false)
        port == C_NULL && return list_port, list_desc

        push!(list_port, LibSerialPort.sp_get_port_name(port))
        push!(list_desc, LibSerialPort.sp_get_port_description(port))
    end

    sp_free_port_list(ports)

    list_port, list_desc
end

function find_stage_port(;nports_guess::Integer=64)
    list_port, list_desc = get_port_list_info(nports_guess=nports_guess)
    idx_port = findfirst(occursin.("LEP MAC6000 Controller",
        list_desc))

    isnothing(idx_port) && error("Could not find the device LED MAC6000 Controller.")
    list_port[idx_port]
end

function check_baud_rate(s::SerialPort, baud_rate=115200)
    sp_return = write(s, "CAN 32 84 60 0\r")

    sleep(0.001)
    read_bytes = bytesavailable(s)
    return_str = String(read(s, read_bytes))
    
    if !startswith(return_str, ":A")
        error("The command returned error")
    end

    # e.g. ":A  CAN 32 212 60 115200\n"
    current_baud_rate = parse(Int, split(return_str)[end])

    if current_baud_rate != baud_rate
        error("Baud rate is $current_baud_rate, not $baud_rate")
    end

    nothing
end
