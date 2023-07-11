function init_deepnet()
    tf_config = config = py_tf.ConfigProto()
    tf_config.gpu_options.allow_growth = true

    global dlc = py_dlc.DLCLive(PATH_MODEL, pcutoff=0.75, 
        tf_config=tf_config)

    start!(cam)
    imid, imtimestamp = getimage!(cam, session.img_array, normalize=false)
    dlc.init_inference(session.img_array[IMG_CROP_RG_X, IMG_CROP_RG_Y])
    stop!(cam)
    
    nothing
end

function init_pid()
    # PID
    global pid_x = py_pid.PID(PID_X_P, PID_X_I, PID_X_D, setpoint=0.)
    global pid_y = py_pid.PID(PID_Y_P, PID_Y_I, PID_Y_D, setpoint=0.)
    pid_x.output_limit = (-15000, 15000)
    pid_y.output_limit = (-15000, 15000)
    pid_x.setpoint = Float64.(0)
    pid_y.setpoint = Float64.(0)
    
    nothing
end

function check_pts_order_pca(points::Array{<:Integer})
    check_pts_order_pca(Float64.(points))
end

function check_pts_order_pca(points::Array{<:AbstractFloat})
    # input pts shape: # [points] by [dim]
    pts = copy(points)
    
    # mean center
    pts .-= mean(pts, dims=1)
    
    # perform pca
    mat_cov = cov(pts)
    mat_eigval = eigvals(mat_cov)
    mat_eigvec = eigvecs(mat_cov)
    eigval_order = sortperm(mat_eigval, rev=true)
    pc_n = eigval_order[1] # 1st component
    prj = (pts * mat_eigvec)[:, pc_n]
    
    # check the order
    issorted(prj) || issorted(prj, rev=true)
end

"""
    get_offset_loc(pts, Δd=15)

Finds loc (x,y) that is Δd away from the pharynx (x_p, y_p)
towards the mid point (x_m, y_m)

Arguments
---------
* `pts`: keypoints in this order: `(x_n, x_m, x_p, y_n, y_m, y_p)`
* `Δd`: offset distance
"""
function get_offset_loc(pts, Δd=15)
    x_n, x_m, x_p, y_n, y_m, y_p = pts
    
    if x_p == x_m
        return x_m, y_m - Δd
    else
        m = (y_m - y_p) / (x_m - x_p)
        b = y_m - x_m * (y_m - y_p) / (x_m - x_p)

        uhat = [1,0]
        v = [x_p - x_m, y_p - y_m]

        Δx = Δd * (dot(uhat, v) / (norm(uhat) * norm(v)))

        return x_m + Δx, (x_m + Δx) * m + b
    end
end