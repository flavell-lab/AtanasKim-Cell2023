function loop_control(ch_control)
    img_array_uint32 = zeros(UInt32, IMG_SIZE_X, IMG_SIZE_Y)
    
    net_out = zeros(Float32, (3,3))
    x_nose, y_nose = 0, 0
    x_mid, y_mid = 0, 0
    x_pharynx, y_pharynx = 0, 0
    p_nose, p_pharynx, p_mid = 0., 0., 0.
    x_offset, y_offset = 0, 0
    
    q_move_stage = false
    q_move_stage_count = 0
    
    for (q_iter_save, q_recording) in ch_control
#         push!(list_t_control, time_ns())
        session.n_loop += 1
                
        # get image
        imid, imtimestamp = getimage!(cam, session.img_array,
            normalize=false, release=true)
        q_recording && push!(session.list_cam_info, (q_iter_save, q_recording, imid, imtimestamp))
        
        # detect features
        net_out .= dlc.get_pose(session.img_array[IMG_CROP_RG_X, IMG_CROP_RG_Y])

        # offsetting since input image to net is cropped
        y_nose, y_mid, y_pharynx = round.(Int, net_out[:,1]) .+ IMG_CROP_RG_Y[1] .+ 1
        x_nose, x_mid, x_pharynx = round.(Int, net_out[:,2]) .+ IMG_CROP_RG_X[1] .+ 1
        p_nose, p_mid, p_pharynx = net_out[:,3]
        
        @sync begin
            # stage control
            q_move_stage = false
            @async if session.q_tracking
                detected_pts = [x_nose y_nose;
                    x_mid y_mid; x_pharynx y_pharynx]

                # determine if enough info to move stage
                if all([p_nose, p_mid, p_pharynx] .> param.θ_net) && check_pts_order_pca(detected_pts)
                    # determine offset
                    x_offset, y_offset = round.(Int, get_offset_loc(detected_pts, param.Δ_offset))
                    q_move_stage = true
                end
                
                if q_move_stage
                    # determine displacement
                    Δx, Δy = (x_offset, y_offset) .- (IMG_CENTER_X, IMG_CENTER_Y)
                    Δ_displacement = sqrt(Δx ^ 2 + Δy ^ 2)

                    if param.Δ_move_min <= Δ_displacement <= param.Δ_move_max
                        v_x_target = pid_x(Δx)
                        v_y_target = pid_y(Δy)
                        
                        set_velocity(sp_stage, round(Int32, v_x_target), round(Int32, v_y_target))
                    else
                        q_move_stage_count += 1
                        if q_move_stage_count > 3
                            q_move_stage_count = 0

                            pid_x.reset()
                            pid_y.reset()
                            set_velocity(sp_stage, Int32(0), Int32(0))
                        end
                    end
                else
                    q_move_stage_count += 1
                    if q_move_stage_count > 3
                        q_move_stage_count = 0
                        pid_x.reset()
                        pid_y.reset()
                        set_velocity(sp_stage, Int32(0), Int32(0))
                    end
                end # if q_move_stage
            end # async

            # display update
            @async if !q_iter_save
                # 1s avg speed text
                str_speed_avg = "N/A"
                if length(session.speed_cb) > 0
                    cbnan = map(x->all(.!(isnan.(x.val))), session.speed_cb)
                    cb_first = session.speed_cb[findfirst(cbnan)]
                    cb_last = session.speed_cb[findlast(cbnan)]
                    
                    Δt_cb = (cb_last.t - cb_first.t) / 1e9 # seconds
                    Δstage =  norm(cb_last.val .- cb_first.val, 2)
                    speed_worm_stage = Δstage / Δt_cb 
                    speed_worm_mm = covert_stage_unit_to_mm(speed_worm_stage) # mm/s
                    str_speed_avg = rpad(string(round(speed_worm_mm, digits=2)), 4, "0")
                    @emit updateTextSpeedAvg(str_speed_avg)
                end
                
                if q_recording
                    Δt_recording = (time_ns() - session.t_recording_start) / 1e9
                    (Δt_recording_m, Δt_recording_s) = fldmod(Δt_recording, 60)
                    str_recording_duration = lpad(round(Int, Δt_recording_m), 2, "0") *
                        ":" * lpad(round(Int, Δt_recording_s), 2, "0")
                    @emit updateTextRecordingDuration(str_recording_duration)
                end
                
                # camera img
                img_array_uint32 .= UInt32.(session.img_array)
                img_array_uint32 .= (0xff000000 .+ img_array_uint32 .+
                    (img_array_uint32 .<< 8) .+ (img_array_uint32 .<< 16))

                # mark detected location
                if session.q_show_net
                    p_nose > param.θ_net && mark_feature!(img_array_uint32, x_nose, y_nose,
                        session.q_tracking ? param.c_nose_on : param.c_nose_off)
                    p_mid > param.θ_net && mark_feature!(img_array_uint32, x_mid, y_mid,
                        session.q_tracking ? param.c_mid_on : param.c_mid_off)
                    p_pharynx > param.θ_net && mark_feature!(img_array_uint32, x_pharynx, y_pharynx,
                        session.q_tracking ? param.c_pharynx_on : param.c_pharnyx_off)

                    q_move_stage && mark_feature!(img_array_uint32, x_offset, y_offset, param.c_offset)
                end

                # ruler
                session.q_show_ruler && mark_ruler!(img_array_uint32, RULER_RG_X, RULER_RG_Y)

                # crosshair
                session.q_show_crosshair && mark_crosshair!(img_array_uint32, IMG_CENTER_X, IMG_CENTER_Y,
                    IMG_CROP_CENTER_X, IMG_CROP_CENTER_Y, param.Δ_move_max)

                # updatedisplay buffer
                canvas_buffer .= img_array_uint32[:]
                @emit updateCanvas()
            end # @async display update
            
            @async if q_iter_save && q_recording
                push!(session.list_img, deepcopy(session.img_array))
                push!(session.list_pos_net, 
                    [x_nose y_nose p_nose; x_mid y_mid p_mid; x_pharynx y_pharynx p_pharynx])

            end
        end
#         yield()
    end # for
end

function loop_stage(ch_stage)
    x_stage, y_stage = Float64(0), Float64(0)
    
    for (q_iter_save, q_recording) in ch_stage
#         push!(list_t_stage, time_ns())
        try
            sleep(0.001)
            query_position(sp_stage)
            sleep(0.025)
            x_stage, y_stage = Float64.(read_position(sp_stage) ./ 2)
            push!(session.speed_cb, ValWithTime((x_stage, y_stage)))
            session.x_stage = x_stage
            session.y_stage = y_stage
        catch
            x_stage, y_stage = NaN, NaN
        end
        session.x_stage, session.y_stage = x_stage, y_stage
        q_recording && push!(session.list_pos_stage, Float64[x_stage, y_stage])
    end
#     yield()
end

function loop_recording(ch_recording)
    for (q_iter_save, q_recording) in ch_recording
        q_recording && nidaq_read_data()
    end 
end

function loop_main()
    ch_stage = Channel{Tuple{Bool,Bool}}(16)
    ch_control = Channel{Tuple{Bool,Bool}}(16)
    ch_recording = Channel{Tuple{Bool,Bool}}(16)
    session.q_loop = true

    @sync begin
        @async loop_stage(ch_stage)
        @async loop_control(ch_control)
        @async loop_recording(ch_recording) # Threads.@spawn
        
        local loop_count = 1
        local q_recording = false
        
        start!(cam)
        Timer(0, interval=1/LOOP_INTERVAL_CONTROL) do timer
            if !q_recording && session.q_recording # start rec
                start(task_di)
                start(task_ai)
                stop!(cam)
                sleep(0.001)
                start!(cam)
            elseif q_recording && !(session.q_recording) # stop rec
                stop!(cam)
                sleep(0.001)
                start!(cam)
                nidaq_read_data()
                stop(task_ai)
                stop(task_di)
            end
            q_recording = session.q_recording
                        
            if session.q_loop == false && loop_count == 1
                close(ch_control)
                close(ch_stage)
                close(timer)
                stop!(cam)
            elseif isodd(loop_count)
                put!(ch_control, (true, q_recording))
                put!(ch_stage, (true, q_recording))
                loop_count += 1
            elseif loop_count == 20
                put!(ch_control, (false, q_recording))
                put!(ch_recording, (false, q_recording))
                loop_count = 1
            else
                put!(ch_control, (false, q_recording))
                loop_count += 1
            end
        end # timer
    end         
end

function stop_loop_main()
    session.q_loop = false
end
