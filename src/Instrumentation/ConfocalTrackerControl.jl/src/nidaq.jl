empty_str = ""
ptr_empty_str = pointer(empty_str)

function init_nidaq()
    # tasks creation
    # analog: laser (488 nm analog), stim for opto (AO))
    global task_ai = analog_input("$NIDAQ_DEV_NAME/ai0, $NIDAQ_DEV_NAME/ai1, $NIDAQ_DEV_NAME/_ao0_vs_aognd",
        terminal_config=NIDAQ.Differential, range=[0,10])
    # digital: confocal camera AUX 1 OUT, behavior camera
    global task_di = digital_input("$NIDAQ_DEV_NAME/port0/line0:1")

    # configure sample clocks
    # AI
    NIDAQ.catch_error(NIDAQ.DAQmxCfgSampClkTiming(task_ai.th, ptr_empty_str, NIDAQ_SAMPLE_RATE_AI,
        NIDAQ.DAQmx_Val_Rising, NIDAQ.DAQmx_Val_ContSamps, 100 * NIDAQ_SAMPLE_RATE_AI))
    # DI
    NIDAQ.catch_error(NIDAQ.DAQmxCfgSampClkTiming(task_di.th, "ai/SampleClock", NIDAQ_SAMPLE_RATE_AI,
        NIDAQ.DAQmx_Val_Rising, NIDAQ.DAQmx_Val_ContSamps, 100 * NIDAQ_SAMPLE_RATE_AI))
end

function read_all_available!(t::NIDAQ.AITask, buffer::Array{Float64,1})
    num_samples_per_chan::Int= -1
    outdata_ref = Ref{Cuint}()
    NIDAQ.DAQmxGetTaskNumChans(t.th, outdata_ref)
    num_channels = outdata_ref.x
    num_samples_per_chan_read = Int32[0]
    buffer_size = length(buffer)
    
    NIDAQ.catch_error(NIDAQ.ReadAnalogF64(t.th,
        convert(Int32,num_samples_per_chan), #numSampsPerChan	
        1.0, # timeout
        reinterpret(Bool32, NIDAQ.Val_GroupByChannel), # fillMode
        Ref(buffer,1),
        convert(UInt32, buffer_size),
        Ref(num_samples_per_chan_read,1),
        reinterpret(Ptr{Bool32},C_NULL)) )

    num_samples_per_chan_read[1], num_channels
end

function read_all_available!(t::NIDAQ.DITask, buffer::Array{UInt32,1})
    num_samples_per_chan::Int=-1
    outdata_ref = Ref{Cuint}()
    NIDAQ.DAQmxGetTaskNumChans(t.th, outdata_ref)
    num_channels = outdata_ref.x
    num_samples_per_chan_read = Int32[0]
    buffer_size = length(buffer)
    
    NIDAQ.catch_error(NIDAQ.ReadDigitalU32(t.th,
            convert(Int32, num_samples_per_chan),
            1.0,
            reinterpret(Bool32, NIDAQ.Val_GroupByChannel),
            Ref(buffer, 1),
            convert(UInt32, buffer_size),
            Ref(num_samples_per_chan_read, 1),
            reinterpret(Ptr{Bool32}, C_NULL)))
    
    num_samples_per_chan_read[1], num_channels
end

function reshape_daqmx_data(buffer, num_samples_per_chan_read, num_channels)
    data = buffer[1:num_samples_per_chan_read * num_channels]
    
    num_channels == 1 ? data : reshape(data, (div(length(data), num_channels), convert(Int64, num_channels)))
end

function nidaq_read_data()
    ai_n_read, ai_n_ch = read_all_available!(task_ai, session.buffer_ai)
    di_n_read, di_n_ch = read_all_available!(task_di, session.buffer_di)
    push!(session.list_daqmx_read, (ai_n_read, di_n_read))
    push!(session.list_ai_read, reshape_daqmx_data(session.buffer_ai, ai_n_read, ai_n_ch))
    push!(session.list_di_read, reshape_daqmx_data(session.buffer_di, di_n_read, di_n_ch))
    
    nothing
end
