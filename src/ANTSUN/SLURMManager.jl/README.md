# SLURMManager.jl
If you have not installed Julia on OpenMind, please check [OpenMind Julia](https://github.com/flavell-lab/FlavellLabWiki/wiki/OpenMind-Julia#installing-julia-on-openmind)

## Usage
### 1. Make a file for sbatch list  
This is a text file that lists the paths of .sh files (sbatch files) you'd like to run.

Example `2020-10-12 sbatch manifest.txt`
```bash
/path_to_script/sbatch1.sh
/path_to_script/sbatch2.sh
/path_to_script/sbatch3.sh
/path_to_script/sbatch4.sh
```
### 2. Define an sbatch script to run the manager
Example `2020-10-12 sbatch submit.sh` 
 - make sure to change `--job-name`, `--output` and the path to your script list file)  
 - adjust `--time` based on the number/duration of your jobs
```bash
#!/bin/bash
#SBATCH --job-name=yourjobname
#SBATCH --output=/path_to_your_log_dir/jobsubmit_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00:00
#SBATCH --mem=500

julia -e "using SLURMManager; submit_scripts("path_to_your_script_list/2020-10-12 sbatch manifest.txt")
```
This requires minimal computing power, so there's no need to increase the num CPU, RAM, etc.

### 3. Submit the manager sbatch script
```bash
sbatch 2020-10-12 sbatch submit.sh
```

### Note
- max array size per sbatch script is 999 (the max pending + running is 1000)
- jobs will be submitted in the order defined in your script list file  
- each time a job is succesfully submitted, it saves a log so that the same job won't be resubmitted (e.g. if the manager sbatch task is intruppted and requeued, it continues with previously unsubmitted jobs)

## Notification feature
Doc TBD
