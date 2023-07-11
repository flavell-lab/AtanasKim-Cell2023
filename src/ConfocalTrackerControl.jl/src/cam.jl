function cam_adjust_gain(gain=25.)
    Spinnaker.gain!(cam, gain)
    cam_gain = Spinnaker.gain(cam)
    @assert(isapprox(gain, cam_gain[1], rtol=0.05))
    @assert(cam_gain[2] == "Off")
    
    nothing
end

function cam_adjust_exposure(exposure=4000)
    Spinnaker.exposure!(cam, exposure)
    cam_exposure = Spinnaker.exposure(cam)
    @assert(isapprox(exposure, cam_exposure[1], rtol=0.05))
    @assert(cam_exposure[2] == "Off")
    
    nothing
end

function cam_adjust_framerate(framerate=LOOP_INTERVAL_CONTROL)
    Spinnaker.set!(Spinnaker.SpinFloatNode(cam, "AcquisitionFrameRate"),
        Float64(framerate))
    @assert(isapprox(Float64(framerate), Spinnaker.framerate(cam), rtol=0.05))
    
    nothing
end

function cam_default()
    cam_adjust_gain(25.)
    cam_adjust_exposure(4000)
    
    nothing
end

function cam_alignment()
    cam_adjust_gain(0.)
    cam_adjust_exposure(500)
    
    nothing
end

function init_cam()
    camlist = CameraList()
    global cam = camlist[0]
    
    cam_adjust_framerate()
    cam_default()
    
    triggermode!(cam, "Off")
    start!(cam)
    imid, imtimestamp = getimage!(cam, session.img_array, normalize=false)
    stop!(cam)
    
    buffermode!(cam, "NewestOnly")
    buffercount!(cam, 3)

    nothing
end