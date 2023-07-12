"""
    write_sbatch_graph(
        edges, param_path_fixed::Dict, param_path_moving::Dict, param::Dict;
        clear_cmd_dir::Bool=true,
        cpu_per_task_key::String="cpu_per_task", 
        memory_key::String="memory", 
        duration_key::String="duration",
        duration_julia_key::String="duration_julia",
        job_name_key::String="job_name",
        fixed_channel_key::String="ch_marker",
        moving_channel_key::String="ch_marker",
        head_dir_key::String="path_head_pos",
        om_data_key::String="path_om_data",
        om_scripts_key::String="path_om_scripts",
        nrrd_dir_key::String="path_dir_nrrd_filt",
        nrrd_om_dir_key::String="path_om_nrrd_filt",
        mask_dir_key::String="path_dir_mask",
        mask_om_dir_key::String="path_om_mask",
        reg_dir_key::String="path_dir_reg",
        reg_om_dir_key::String="path_om_reg",
        path_head_rotate_key::String="path_head_rotate",
        parameter_files_key::String="parameter_files",
        cmd_dir_key::String="path_dir_cmd",
        cmd_om_key::String="path_om_cmd",
        cmd_array_dir_key::String="path_dir_cmd_array",
        cmd_array_om_key::String="path_om_cmd_array"
    )


Syncs data from local computer to a remote server and creates command files for elastix on that server.

**WARNING:** This program can permanently delete data if run with incorrect arguments.

# Arguments
- `edges`: List of registration problems to perform
- `param_path_fixed::Dict`: Dictionary containing paths for the fixed images including:
    - `get_basename`: Function that maps channel and time point to NRRD filename
    - `path_dir_cmd`: Path to elastix command directory
    - `path_om_cmd`: Path to elastix command directory on the server
    - `path_om_scripts`: Path to directory to store scripts on the server
    - `path_dir_cmd_array`: Path to elastix array command directory
    - `path_om_cmd_array`: Path to elastix array command directory on the server
    - `path_om_log`: Path to log file on server
    - `path_om_tmp`: Path to temporary directory on server
    - `path_om_env`: Path to script to set environment variables
    - `path_run_elastix`: Path to script that runs elastix given command on the server
    - `path_elastix`: Path to elastix executable on the server
    - `name_head_rotate_logfile`: Name of head rotate log files
- `param_path_moving::Dict`: Dictionary containing paths for the moving images including the same keys as with the fixed dictionary.
- `param::Dict`: Dictionary containing parameters including:
    - `email`: Email to inform user of task termination. If `nothing`, no emails will be sent
    - `use_sbatch`: Use `sbatch`, rather than directly running code on the server. This should always be set to `true` on OpenMind
    - `server_dtn`: Address of server to transfer data to
    - `user`: Username on server
    - `array_size`: Size of `sbatch` array to use
    - `partition`: Priority of jobs on `sbatch`

## Optional keyword arguments

 - `clear_cmd_dir::Bool`: Whether to clear the elastix command directory, useful if you are re-running registrations
 - `cpu_per_task_key::String`: Key in `param` to CPU cores per elastix task. Default `cpu_per_task`
 - `memory_key::String`: Key in `param` to memory per elastix task. Default `memory`
 - `duration_key::String`: Key in `param` to the duration of each elastix task. Default `duration`
 - `duration_julia_key::String`: Key in `param` to the duration of the Julia script that runs all elastix task. Default `duration_julia`
 - `job_name_key::String`: Key in `param` to the name of the elastix tasks. Default `job_name`
 - `fixed_channel_key::String`: Key in `param` to the fixed channel. Default `ch_marker`
 - `moving_channel_key::String`: Key in `param` to the moving channel. Default `ch_marker`
 - `head_dir_key::String`: Key in `param_path_*` to the head position of the worm. Default `path_head_pos`
 - `om_data_key::String`: Key in `param_path_*` to the path to sync the data on the server. Default `path_om_data`
 - `om_scripts_key::String`: Key in `param_path_*` to the path to sync the scripts on the server. Default `path_om_scripts
 - `nrrd_dir_key::String`: Key in `param_path_*` to the path to the NRRD files. Default `path_dir_nrrd_filt`
 - `nrrd_om_dir_key::String`: Key in `param_path_*` to the path to the NRRD files on the server. Default `path_om_nrrd_filt`
 - `mask_dir_key::String`: Key in `param_path_*` to the mask path. Default `path_dir_mask`
 - `mask_om_dir_key::String`: Key in `param_path_*` to the mask path on the server. Default `path_om_mask`
 - `reg_dir_key::String`: Key in `param_path_*` to the registration output directory. Default `path_dir_reg`
 - `reg_om_dir_key::String`: Key in `param_path_*` to the registration output directory on the server. `path_om_reg`
 - `path_head_rotate_key::String`: Key in `param_path_fixed` to the path on the server to the head rotation python file. Default `path_head_rotate`
 - `parameter_files_key::String`: Key in `param_path_fixed` to the path on the server to the elastix parameter files. Default `parameter_files`
 - `cmd_dir_key::String`: Key in `param_path_fixed` to the local path to store the `elastix` scripts. Default `path_dir_cmd`
 - `cmd_om_key::String`: Key in `param_path_fixed` to the path on the server to store the `elastix` scripts. Default `path_om_cmd`
 - `cmd_array_dir_key::String`: Key in `param_path_fixed` to the local path to store the `elastix` array scripts. Default `path_dir_cmd_array`
 - `cmd_array_om_key::String`: Key in `param_path_fixed` to the path on the server to store the `elastix` array scripts. Default `path_om_cmd_array`
"""
function write_sbatch_graph(edges, param_path_fixed::Dict, param_path_moving::Dict, param::Dict;
        clear_cmd_dir::Bool=true,
        cpu_per_task_key::String="cpu_per_task", 
        memory_key::String="memory", 
        duration_key::String="duration",
        duration_julia_key::String="duration_julia",
        job_name_key::String="job_name",
        fixed_channel_key::String="ch_marker",
        moving_channel_key::String="ch_marker",
        head_dir_key::String="path_head_pos",
        om_data_key::String="path_om_data",
        om_scripts_key::String="path_om_scripts",
        nrrd_dir_key::String="path_dir_nrrd_filt",
        nrrd_om_dir_key::String="path_om_nrrd_filt",
        mask_dir_key::String="path_dir_mask",
        mask_om_dir_key::String="path_om_mask",
        reg_dir_key::String="path_dir_reg",
        reg_om_dir_key::String="path_om_reg",
        path_head_rotate_key::String="path_head_rotate",
        parameter_files_key::String="parameter_files",
        cmd_dir_key::String="path_dir_cmd",
        cmd_om_key::String="path_om_cmd",
        cmd_array_dir_key::String="path_dir_cmd_array",
        cmd_array_om_key::String="path_om_cmd_array")

    script_dir_remote = param_path_fixed[om_scripts_key]
    data_dir_remote = param_path_fixed[om_data_key]
    data_dir_remote_moving = param_path_moving[om_data_key]
    nrrd_dir_local = param_path_fixed[nrrd_dir_key]
    nrrd_dir_local_moving = param_path_moving[nrrd_dir_key]
    nrrd_dir_remote = param_path_fixed[nrrd_om_dir_key]
    nrrd_dir_remote_moving = param_path_moving[nrrd_om_dir_key]
    mask_dir_local = param_path_fixed[mask_dir_key]
    mask_dir_local_moving = param_path_moving[mask_dir_key]
    mask_dir_remote = param_path_fixed[mask_om_dir_key]
    mask_dir_remote_moving = param_path_moving[mask_om_dir_key]
    reg_dir_local = param_path_fixed[reg_dir_key]
    reg_dir_remote = param_path_fixed[reg_om_dir_key]
    head_dir = param_path_fixed[head_dir_key]
    head_dir_moving = param_path_moving[head_dir_key]

    get_basename = param_path_fixed["get_basename"]
    get_basename_moving = param_path_moving["get_basename"]

    cmd_dir_local = param_path_fixed[cmd_dir_key]
    cmd_dir_remote = param_path_fixed[cmd_om_key]
    cmd_dir_array_local = param_path_fixed[cmd_array_dir_key]
    cmd_dir_array_remote = param_path_fixed[cmd_array_om_key]
    head_rotate_path = param_path_fixed[path_head_rotate_key]
    log_dir = param_path_fixed["path_om_log"]
    run_elx_command = param_path_fixed["path_run_elastix"]
    elastix_path = param_path_fixed["path_elastix"]
    parameter_files = param_path_fixed[parameter_files_key]
    euler_logfile = param_path_fixed["name_head_rotate_logfile"]
    env_cmd = param_path_fixed["path_om_env"]
    
    cpu_per_task = param[cpu_per_task_key]
    mem = param[memory_key]
    duration = param[duration_key]
    duration_julia = param[duration_julia_key]

    fixed_channel = param[fixed_channel_key]
    moving_channel = param[moving_channel_key]
    job_name = param[job_name_key]
    email = param["email"]
    use_sbatch = param["use_sbatch"]
    server = param["server_dtn"]
    user = param["user"]
    array_size = param["array_size"]

    temp_dir = param_path_fixed["path_om_tmp"]
    temp_file = joinpath(temp_dir, "elx_commands.txt")
    partition = param["partition"]


    # erase previous scripts and replace them with new ones
    if clear_cmd_dir
        println("Resetting $(cmd_dir_local)...")
        rm(cmd_dir_local, recursive=true, force=true)
        if !isnothing(cmd_dir_array_local)
            rm(cmd_dir_array_local, recursive=true, force=true)
        end
    end
    create_dir(cmd_dir_local)
    create_dir(cmd_dir_array_local)
    duration_str = Dates.format(duration, "HH:MM:SS")
    duration_julia_str = duration_julia


    # Euler registration requires knowing worm head location
    use_euler = (head_rotate_path !== nothing)
    if use_euler
        if head_dir === nothing
            raise(error("Head path cannot be empty if head rotation is being used"))
        end
        println("Getting head position...")
        head_pos = read_head_pos(head_dir)
        head_pos_moving = read_head_pos(head_dir_moving)
    end

    println("Writing elastix script files...")
    count = 1
    edges_in_arr = []
    for i in 1:length(edges)
        edge = edges[i]
        dir=string(edge[1])*"to"*string(edge[2])
        push!(edges_in_arr, joinpath(cmd_dir_remote, dir*".sh"))
        script_str=""
        fixed_final=lpad(edge[2],4,"0")
        moving_final=lpad(edge[1],4,"0")
        # set sbatch parameters in script
        if use_sbatch
            # if using array, only set them in array
            if cmd_dir_array_local === nothing
                script_str *= replace("#!/bin/bash
                #SBATCH --job-name=$(job_name)
                #SBATCH --output=$(log_dir)/elx_$(dir).txt
                #SBATCH --nodes=1
                #SBATCH --ntasks=1
                #SBATCH --cpus-per-task=$(cpu_per_task)
                #SBATCH --time=$(duration_str)
                #SBATCH --mem=$(mem)G\n", "    " => "")
                if email !== nothing
                    script_str *= "#SBATCH --mail-user=$(email)
                    #SBATCH --mail-type=END\n"
                end
            elseif length(edges_in_arr) == array_size || i == length(edges)
                # modify in case we finished all the scripts
                modified_array_size = length(edges_in_arr)
                array_str = ""
                array_str *= replace("#!/bin/bash
                #SBATCH --job-name=$(job_name)
                #SBATCH --output=$(log_dir)/$(job_name)_%J.out
                #SBATCH --error=$(log_dir)/$(job_name)_%J.err
                #SBATCH --nodes=1
                #SBATCH --cpus-per-task=$(cpu_per_task)
                #SBATCH --time=$(duration_str)
                #SBATCH --mem=$(mem)G
                #SBATCH --array=1-$(modified_array_size)\n", "    " => "")
                if email !== nothing
                    array_str *= "#SBATCH --mail-user=$(email)
                    #SBATCH --mail-type=END\n"
                end
                script_list_file = joinpath(cmd_dir_remote, "$(job_name)_$(count).txt")
                array_str *= "$(run_elx_command) $(script_list_file) \$SLURM_ARRAY_TASK_ID\n"
                write_txt(joinpath(cmd_dir_local, "$(job_name)_$(count).txt"), reduce((x,y)->x*"\n"*y, edges_in_arr)*"\n")
                write_txt(joinpath(cmd_dir_array_local, "$(job_name)_$(count).sh"), array_str)
                count += 1
                edges_in_arr = []
            end
        end


        # make directory
        reg = joinpath(reg_dir_remote, dir)
        script_str *= "[ ! -d $(reg) ] && mkdir $(reg)\n"
        script_str *= "source $(env_cmd)\n"

        # Euler registration
        if use_euler
            script_str *= "python $(head_rotate_path)"*
                " "*joinpath(nrrd_dir_remote, get_basename(edge[2], fixed_channel)*".nrrd")*
                " "*joinpath(nrrd_dir_remote_moving, get_basename_moving(edge[1], moving_channel)*".nrrd")*
                " "*joinpath(reg_dir_remote, dir, "$(dir)_euler.txt")*
                " $(head_pos[edge[2]][1]),$(head_pos[edge[2]][2])"*
                " $(head_pos_moving[edge[1]][1]),$(head_pos_moving[edge[1]][2]) > $(joinpath(reg_dir_remote, dir, euler_logfile))\n"
        end
        
        # elastix image and output parameters
        script_str *= elastix_path*
            " -f "*joinpath(nrrd_dir_remote, get_basename(edge[2], fixed_channel)*".nrrd")*
            " -m "*joinpath(nrrd_dir_remote_moving, get_basename_moving(edge[1], moving_channel)*".nrrd")*
            " -out "*joinpath(reg_dir_remote, dir)
        # mask parameters
        if mask_dir_local !== nothing
            script_str *= " -fMask "*joinpath(mask_dir_remote, get_basename(edge[2], fixed_channel)*".nrrd")*
            " -mMask "*joinpath(mask_dir_remote_moving, get_basename_moving(edge[1], moving_channel)*".nrrd")
        end
        # initial condition parameters
        if use_euler
            script_str *= " -t0 "*joinpath(reg_dir_remote, dir, "$(dir)_euler.txt")
        end
        # add parameter files
        for pfile in parameter_files
            script_str *= " -p $(pfile)"
        end
        # write elastix script
        script_str = mapfoldl(x->lstrip(x) * "\n", *, split(script_str, "\n"))
        filename = joinpath(cmd_dir_local, "$(dir).sh")
        open(filename, "w") do f
            write(f, script_str)
        end
    end

    # write sbatch script to run Julia script
    julia_sbatch_str = replace("#!/bin/bash
                #SBATCH --job-name=julia_run_elx
                #SBATCH --output=$(log_dir)/julia_run_elx.txt
                #SBATCH --nodes=1
                #SBATCH --ntasks=1
                #SBATCH --cpus-per-task=1
                #SBATCH --time=$(duration_julia_str)
                #SBATCH --mem=$(mem)G
                
                julia -e \"using SLURMManager; submit_scripts(\\\"$(temp_file)\\\", partition=\\\"$(partition)\\\")\"\n", "    " => "")

    julia_sbatch_file = joinpath(param_path_fixed["path_root_process"], "run_elastix_julia.sh")

    open(julia_sbatch_file, "w") do f
        write(f, julia_sbatch_str)
    end

    println("Syncing data to server...")
    # make necessary directories on server
    run(`ssh $(user)@$(server) "mkdir -p $(data_dir_remote)"`)
    run(`ssh $(user)@$(server) "mkdir -p $(script_dir_remote)"`)
    run(`ssh $(user)@$(server) "mkdir -p $(data_dir_remote_moving)"`)
    run(`ssh $(user)@$(server) "mkdir -p $(log_dir)"`)
    reg = reg_dir_remote
    run(`ssh $(user)@$(server) "mkdir -p $(reg)"`)
    # sync all data to the server
    run(Cmd(["rsync", julia_sbatch_file, "$(user)@$(server):"*data_dir_remote]))
    if clear_cmd_dir
        run(Cmd(["rsync", "-r", "--delete", cmd_dir_local*"/", "$(user)@$(server):"*cmd_dir_remote]))
        if cmd_dir_array_local !== nothing
            run(Cmd(["rsync", "-r", "--delete", cmd_dir_array_local*"/", "$(user)@$(server):"*cmd_dir_array_remote]))
        end
    else
        run(Cmd(["rsync", "-r", cmd_dir_local*"/", "$(user)@$(server):"*cmd_dir_remote]))
        if cmd_dir_array_local !== nothing
            run(Cmd(["rsync", "-r", cmd_dir_array_local*"/", "$(user)@$(server):"*cmd_dir_array_remote]))
        end
    end
    run(Cmd(["rsync", "-r", nrrd_dir_local*"/", "$(user)@$(server):"*nrrd_dir_remote]))
    if param_path_fixed != param_path_moving
        run(Cmd(["rsync", "-r", "--delete", nrrd_dir_local_moving*"/", "$(user)@$(server):"*nrrd_dir_remote_moving]))
    end
    if mask_dir_local !== nothing
        run(Cmd(["rsync", "-r", "--delete", mask_dir_local*"/", "$(user)@$(server):"*mask_dir_remote]))
        if param_path_fixed != param_path_moving
            run(Cmd(["rsync", "-r", "--delete", mask_dir_local_moving*"/", "$(user)@$(server):"*mask_dir_remote_moving]))
        end
    end
end


"""
    run_registration(
        param_path, path_img_fixed, path_img_moving; reg_dir_key="path_dir_reg_neuropal",
        elastix_key="path_elastix_local", parameter_files_key="parameter_files_local",
        init_param_file_key=nothing
    )

Runs elastix registration locally, for NeuroPAL.

# Arguments:
- `param_path`: Dictionary containing the keys below.
- `path_img_fixed`: Path to fixed image.
- `path_img_moving`: Path to moving image.
- `reg_dir_key` (optional, default `path_dir_reg_neuropal`): Path to registration output directory.
- `elastix_key` (optional, default `path_elastix_local`): Path to `elastix` executable.
- `parameter_files_key` (optional, default `parameter_files_local`): Array of paths to parameter files.
- `init_param_file_key` (optional, default `nothing`): Initial parameter file.
"""
function run_registration(param_path, path_img_fixed, path_img_moving; reg_dir_key="path_dir_reg_neuropal",
        elastix_key="path_elastix_local", parameter_files_key="parameter_files_local", init_param_file_key=nothing)
    reg_dir = param_path[reg_dir_key]
    path_elastix = param_path[elastix_key]
    
    cmd_arr = [path_elastix]
    push!(cmd_arr, "-f")
    push!(cmd_arr, path_img_fixed)
    push!(cmd_arr, "-m")
    push!(cmd_arr, path_img_moving)
    push!(cmd_arr, "-out")
    push!(cmd_arr, reg_dir)
    for p in param_path[parameter_files_key]
        push!(cmd_arr, "-p")
        push!(cmd_arr, p)
    end

    if !isnothing(init_param_file_key)
        push!(cmd_arr, "-t0")
        push!(cmd_arr, param_path[init_param_file_key])
    end

    run(Cmd(cmd_arr));
end

"""
    average_registered_images(reg_dir, central_nrrd_path, registration_problems, res_name)

Averages all registered images, assuming they are all being registered to the same time point.
# Arguments:
- `reg_dir`: Registration directory that contains all the moving images registered to the fixed image.
- `central_nrrd_path`: Path to fixed image.
- `registration_problems`: List of registration problems of the moving images to the fixed image.
- `res_name`: Name of registration file (ie: resolution) to use.
"""
function average_registered_images(reg_dir, central_nrrd_path, registration_problems, res_name)
    imgs = [Int64.(read_img(NRRD(central_nrrd_path)))]
    @showprogress for (p1, p2) in registration_problems
        try
            push!(imgs, Int64.(read_img(NRRD(joinpath(reg_dir, "$(p1)to$(p2)", res_name)))))
        catch e
            @warn("Registration $(p1), $(p2) failed")
        end
    end
    return sum(imgs) / length(imgs)
end
        

"""
    run_elastix_openmind(param_path::Dict, param::Dict; extra_cmd_paths=[])

Runs elastix on OpenMind. Requires `julia` to be installed under the relevant username and activated in the default ssh shell.
Note that you cannot have multiple instances of this command running simultaneously with the same `temp_dir`.
Returns the jobid of the Julia script-submission job.

# Arguments
- `param_path::Dict`: Dictionary of parameter paths including:
    - `path_om_tmp`: Path to temporary directory on OpenMind.
    - `path_om_cmd`: Path to elastix command directory on OpenMind.
    - `path_om_cmd_array`: Path to elastix array command directory on OpenMind.
- `param::Dict`: Dictionary of parameter settings including:
    - `user`: OpenMind username
    - `server`: Login node address on OpenMind
    - `partition`: Partition to run elastix using (eg `use-everything`)
- `extra_cmd_paths` (optional, default `[]`): Extra paths to script files to run 
"""
function run_elastix_openmind(param_path::Dict, param::Dict; extra_cmd_paths=[])
    temp_dir = param_path["path_om_tmp"]
    temp_file = joinpath(temp_dir, "elx_commands.txt")
    all_temp_files = joinpath(temp_dir, "*")
    cmd_path = [param_path["path_om_cmd"]]
    if param_path["path_om_cmd_array"] !== nothing
        cmd_path = [param_path["path_om_cmd_array"]]
    end
    append!(cmd_path, extra_cmd_paths)
    all_script_files = joinpath.(cmd_path, "*")
    user = param["user"]
    server = param["server"]
    partition = param["partition"]
    run_elastix_julia_path = joinpath(param_path["path_om_data"], "run_elastix_julia.sh")
    run(`ssh $(user)@$(server) "mkdir -p $(temp_dir)"`)
    run(`ssh $(user)@$(server) "rm -f $(all_temp_files)"`)
    run(`ssh $(user)@$(server) "ls -d $(all_script_files[1]) > $(temp_file)"`)
    for i=2:length(all_script_files)
        run(`ssh $(user)@$(server) "ls -d $(all_script_files[i]) >> $(temp_file)"`)
    end
    return squeue_submit_sbatch_remote(param, run_elastix_julia_path, partition="flavell")
end


function squeue_submit_sbatch_remote(param, path_sh; partition="normal")
    user = param["user"]
    server = param["server"]
    jobid = run_parse_int(pipeline(`ssh $(user)@$(server) "sbatch --partition=$(partition) $path_sh"`, `awk '{ print $4 }'`))
    return jobid
end

"""
    get_squeue_status(param::Dict)

Gets the number of running and pending `squeue` commands from the given user.

# Arguments
- `param::Dict`: Parameter dictionary including:
    - `user`: Username on OpenMind
    - `server`: Login node address on OpenMind
"""
function get_squeue_status(param::Dict)
    user = param["user"]
    server = param["server"]
    running = run_parse_int(pipeline(`ssh $(user)@$(server) "julia -e \"using SLURMManager; println(squeue_n_running(\\\"$(user)\\\"))\""`))
    pending = run_parse_int(pipeline(`ssh $(user)@$(server) "julia -e \"using SLURMManager; println(squeue_n_pending(\\\"$(user)\\\"))\""`))
    return running + pending
end


"""
    get_squeue_status(param::Dict, jobid::Int)

Gets the number of running and pending `squeue` commands from the given user.

# Arguments
- `param::Dict`: Parameter dictionary including:
    - `user`: Username on OpenMind
    - `server`: Login node address on OpenMind
- `jobid::Int`: Sbatch job ID of the Julia script-submission script
"""
function get_squeue_status(param::Dict, jobid::Int)
    user = param["user"]
    server = param["server"]
    running = run_parse_int(pipeline(`ssh $(user)@$(server) "julia -e \"using SLURMManager; println(squeue_n_running($(jobid)))\""`))
    pending = run_parse_int(pipeline(`ssh $(user)@$(server) "julia -e \"using SLURMManager; println(squeue_n_pending($(jobid)))\""`))
    return running + pending
end

"""
    get_lock(param_path::Dict, param::Dict; lock_key::String="path_dir_lock")

Get a lock in `param_path[lock_key]`. Wait a minimum of `param["lock_wait"]` between successive lock checks.
"""
function get_lock(param_path::Dict, param::Dict; lock_key::String="path_dir_lock")
    while length(readdir(param_path[lock_key])) > 0
        sleep(rand() + param["lock_wait"])
    end
    touch(joinpath(param_path[lock_key], split(param_path["path_root_process"], "/")[end]*".lock"))

    # check for failed race condition
    sleep(rand() + param["lock_wait"])
    if length(readdir(param_path[lock_key])) > 1
        release_lock(param_path, param; lock_key=lock_key)
        get_lock(param_path, param; lock_key=lock_key)
    end
end

"""
    release_lock(param_path::Dict, param::Dict; lock_key::String="path_dir_lock")

Release a lock in `param_path[lock_key]`.
"""
function release_lock(param_path::Dict, param::Dict; lock_key::String="path_dir_lock")
    rm(joinpath(param_path[lock_key], split(param_path["path_root_process"], "/")[end]*".lock"))
end

"""
    wait_for_elastix(param::Dict)

This function stalls until all the user's jobs on OpenMind are completed.

# Arguments
- `param::Dict`: Parameter dictionary including:
    - `user`: Username on OpenMind
    - `server`: Login node address on OpenMind
    - `elx_wait_delay`: Time to wait between checking whether elastix is done, in seconds
"""
function wait_for_elastix(param::Dict)
    while get_squeue_status(param) > 0
        sleep(param["elx_wait_delay"])
    end
end

"""
    wait_for_elastix(param::Dict, jobid::Int)

This function stalls until the specified job is completed.

# Arguments
- `param::Dict`: Parameter dictionary including:
    - `user`: Username on OpenMind
    - `server`: Login node address on OpenMind
    - `elx_wait_delay`: Time to wait between checking whether elastix is done, in seconds
- `jobid::Int`: Sbatch job ID of the Julia script-submission script
"""
function wait_for_elastix(param::Dict, jobid::Int)
    while get_squeue_status(param, jobid) > 0
        sleep(param["elx_wait_delay"])
    end
end

"""
    sync_registered_data(param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg", reg_om_dir_key::String="path_om_reg")

Deletes unnecessary files, then syncs registration data from a remote compute server back to the local computer.

# Arguments

- `param_path::Dict`: Dictionary of paths
- `param::Dict`: Dictionary of parameters including:
    - `user`: Username on OpenMind
    - `server_dtn`: Data transfer node address on OpenMind
- `reg_dir_key::String` (optional, default `path_dir_reg`): Key in `param_path` to the path to the registration output directory
- `reg_om_dir_key::String` (optional, default `path_om_reg`): Key in `param_path` to the path to the registration output directory on the server
"""
function sync_registered_data(param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg", reg_om_dir_key::String="path_om_reg")
    reg_dir_local = param_path[reg_dir_key] 
    create_dir(reg_dir_local)
    user = param["user"]
    server = param["server_dtn"]
    reg_dir_remote = param_path[reg_om_dir_key]

    # delete unnecessary files
    run(`ssh $(user)@$(server) "find $(reg_dir_remote) -name IterationInfo* -delete"`)
    run(`ssh $(user)@$(server) "find $(reg_dir_remote) -name elastix.log -delete"`)
    
    # sync remaining files
    run(Cmd(["rsync", "-r", "$(user)@$(server):"*reg_dir_remote*"/", reg_dir_local]))
end

"""
    fix_param_paths(problems, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg", n_resolution_key::String="reg_n_resolution")

Updates parameter paths in transform parameter files, to allow `transformix` to be run on them.
Returns a dictionary of errors per problem and resolution.

# Arguments
- `problems`: Registration problems to update
- `param_path::Dict`: Dictionary of paths including:
    - `path_root_process`: Path to data
    - `path_om_data`: Path to data on server
- `param::Dict`: Dictionary of parameters
- `reg_dir_key::String` (optional, default `path_dir_reg`): Key in `param_path` to the path to the registration output directory
- `n_resolution_key::String` (optional, default `reg_n_resolution`): Key in `param` to array of number of registrations with each parameter file
"""
function fix_param_paths(problems, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg", n_resolution_key::String="reg_n_resolution")
    errors = Dict()
    reg_dir_local = param_path[reg_dir_key]
    resolutions = param[n_resolution_key]
    rootpath = param_path["path_root_process"]
    @showprogress for problem in problems
        errors[problem] = Dict()
        dir = joinpath(reg_dir_local, "$(problem[1])to$(problem[2])")
        for i=1:length(resolutions)
            filename = joinpath(dir, "TransformParameters.$(i-1).txt")
            try
            modify_parameter_file(filename, filename, Dict(param_path["path_om_data"] => rootpath); is_universal=true)
            catch e
                errors[problem][i] = e
            end
            for j=1:resolutions[i]
                filename = joinpath(dir, "TransformParameters.$(i-1).R$(j-1).txt")
                try
                    modify_parameter_file(filename, filename, Dict(param_path["path_om_data"] => rootpath); is_universal=true)
                catch e
                    errors[problem][(i,j)] = e
                end
            end
        end
    end
    return errors
end

"""
    average_am_registrations(
        t_range, param_path::Dict; reg_dir_key::String="path_dir_reg_activity_marker",
        transform_key::String="name_transform_activity_marker", transform_avg_key::String="name_transform_activity_marker_avg",
        key_param_key::String="key_transform_parameters", avg_fn::Function=median
    )

Averages together registrations. All regstration parameters (including image size) must be the same except for the `TransformParameters`.

# Arguments

 - `t_range`: Time points to register
 - `param_path::Dict`: Dictionary containing paths to parameter files
 - `reg_dir_key::String` (optional, default `path_dir_reg_activity_marker`): Key in `param_path` containing registration directory
 - `transform_key::String` (optional, default `name_transform_activity_marker`): Key in `param_path` contaning transform file names
 - `transform_avg_key::String` (optional, default `name_transform_activity_marker_avg`): Key in `param_path` contaning averaged transform file names (to be created)
 - `key_param_key::String` (optional, default `key_transform_parameters`): Key in `param_path` containing the `TransformParameters` key.
 - `avg_fn::Function` (optional, default `median`): Function used to average together registrations. Default `median`.
"""
function average_am_registrations(t_range, param_path::Dict;
        reg_dir_key::String="path_dir_reg_activity_marker", transform_key::String="name_transform_activity_marker",
        transform_avg_key::String="name_transform_activity_marker_avg", key_param_key::String="key_transform_parameters", avg_fn::Function=median)
    euler_params = Dict()
    errors = Dict()
    @showprogress for t in t_range
        try
            euler_params[t] = read_parameter_file(joinpath(param_path[reg_dir_key], "$(t)to$(t)", param_path[transform_key]), param_path[key_param_key], Float64)
        catch e
            errors[t] = e
        end
    end

    min_t = minimum(keys(euler_params))
    params_avg = [avg_fn([euler_params[t][i] for t in keys(euler_params)]) for i in 1:length(euler_params[min_t])]
    params_avg_str = replace(string(params_avg), r"\[|\,|\]|"=>"")

    path_min_t_transform_avg = joinpath(param_path[reg_dir_key], "$(min_t)to$(min_t)", param_path[transform_avg_key])

    modify_parameter_file(joinpath(param_path[reg_dir_key], "$(min_t)to$(min_t)", param_path[transform_key]),
        path_min_t_transform_avg, Dict(param_path[key_param_key] => params_avg_str); is_universal=false)
    @showprogress for t in t_range
        if t == min_t
            continue
        end
        create_dir(joinpath(param_path[reg_dir_key], "$(t)to$(t)"))
        cp(path_min_t_transform_avg, joinpath(param_path[reg_dir_key], "$(t)to$(t)", param_path[transform_avg_key]), force=true)
    end
    return euler_params, params_avg, errors
end
