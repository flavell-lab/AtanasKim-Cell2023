module SLURMManager

using Dates, JLD2, DataStructures

USER = ENV["USER"]
TIME_CHECK_EVERY = 600 # seconds
TIME_RESUBMIT = 300 # seconds
MAX_ATTEMPT = 10
MAX_JOBS_IN_QUEUE = 1000

include("job.jl")
include("notf.jl")
include("sbatch.jl")
include("squeue.jl")
include("submit_loop.jl")

export SLURMJob,
    # submit_loop.jl
    submit_scripts,
    submit_scripts!,
    squeue_n_pending,
    squeue_n_running,
    run_parse_int

end # module
