#!/usr/bin/env bash

#SBATCH --job-name={jname}
{singleton}

#SBATCH --partition={partition}
#SBATCH -c {num_devices}
#SBATCH -C {constraint}

#SBATCH --output={log_fname}
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

cd {task_dirname}

source /share/data/pals/jjahn/mc3/bin/activate {conda_env}

{job_cmd}
