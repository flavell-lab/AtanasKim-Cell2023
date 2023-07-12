"""
    run_parse_int(pipe)

Parse str result from pipeline to Int
"""
function run_parse_int(pipe)
    str_ = read(pipe, String)
    parse(Int, chomp(str_))
end

"""
    squeue_n_pending(user::String)

Get num of pending jobs
"""
function squeue_n_pending(user::String)
    run_parse_int(pipeline(`squeue -u $user -h -r --start`, `wc -l`))
end

"""
    squeue_n_pending(user)

Get num of pending jobs
"""
function squeue_n_pending(jobid::Int)
    run_parse_int(pipeline(`squeue --job $jobid -h -r --start`, `wc -l`))
end


"""
    squeue_n_running(user::String)

Get num of running jobs
"""
function squeue_n_running(user::String)
    run_parse_int(pipeline(`squeue -u $user --states RUNNING -h -r`, `wc -l`))
end

"""
    squeue_n_running(jobid::Int)

Get num of running jobs
"""
function squeue_n_running(jobid::Int)
    run_parse_int(pipeline(`squeue --job $jobid --states RUNNING -h -r`, `wc -l`))
end

"""
    squeue_submit_sbatch(path_sh; partition="normal")

Submits `path_sh` to the queue with given partition (default normal) and returns the jobid
"""
function squeue_submit_sbatch(path_sh; partition="normal")
    jobid = run_parse_int(pipeline(`sbatch --partition=$(partition) $path_sh`, `awk '{ print $4 }'`))
    return jobid
end
