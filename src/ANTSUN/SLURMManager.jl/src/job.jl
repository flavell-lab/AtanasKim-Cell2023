mutable struct SLURMJob
    path_sh::String
    jobid::Union{Int,Nothing}
    submitted::Bool
    completed::Bool
    n_array::Int

    function SLURMJob(path_sh)
        n_array = read_sbatch_array_len(path_sh)
        new(path_sh, -1, false, false, n_array)
    end
end
