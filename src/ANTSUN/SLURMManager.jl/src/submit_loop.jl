function submit_scripts(path_txt; verbose::Bool=true, partition="normal")
    path_root = splitdir(path_txt)[1]
    name_jobdb = splitext(basename(path_txt))[1] * ".jld2"
    path_jobdb = joinpath(path_root, name_jobdb)

    if isfile(path_jobdb)
        @load path_jobdb dict_jobs
    else
        script_path_list = read_script_paths(path_txt)
        dict_jobs = OrderedDict{String, SLURMJob}()
        for path_script = script_path_list
            dict_jobs[path_script] = SLURMJob(path_script)
        end
    end

    submit_scripts!(dict_jobs, path_jobdb=path_jobdb, verbose=verbose, partition=partition)
end

function submit_scripts!(dict_jobs::OrderedDict{String, SLURMJob};
    path_jobdb::String, verbose::Bool=true, partition="normal")
    list_key_submit = []
    for (k_, v_) = dict_jobs
        if !v_.submitted
            push!(list_key_submit, k_)
        end
    end

    for k_ = list_key_submit
        job = dict_jobs[k_]
        n_array = job.n_array

        n_attempt = 0
        q_submit_ok = false

        # try sbatch until success, trying maximum MAX_ATTEMPT
        while !(q_submit_ok) && (n_attempt < MAX_ATTEMPT)
            # wait until queue is not full
            while !(squeue_n_pending(USER) + squeue_n_running(USER) +
                n_array <= MAX_JOBS_IN_QUEUE)
                # queue is too full. checking again after waiting
                sleep(TIME_CHECK_EVERY)
            end

            # try submitting the job
            try
                jobid = squeue_submit_sbatch(job.path_sh, partition=partition)
                job.jobid = jobid
                job.submitted = true
                q_submit_ok = true
                verbose &&
                    println_dt("$(basename(job.path_sh)) submitted id=$jobid")
                flush(stdout)
            catch
                # sbatch failed. trying again after waiting
                n_attempt += 1
                verbose &&
                    println_dt("job submit failed. waiting $(TIME_RESUBMIT)s") 
                flush(stdout)
                sleep(TIME_RESUBMIT)
            end

            # write job db
            @save path_jobdb dict_jobs
        end

        if n_attempt == MAX_ATTEMPT
            error("Max attempt (n=$(MAX_ATTEMPT)) exceeded.")
        end

    end
    verbose && println_dt("all jobs have been submitted.") 
    flush(stdout)

    nothing
end
