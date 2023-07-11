#!/usr/bin/env bash

#SBATCH --job-name={jname}
{singleton}

#SBATCH --partition={partition}
#SBATCH -c {num_devices}

#SBATCH --output={log_fname}
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

cd {task_dirname}

eval "$(/share/data/pals/jjahn/mc3/bin/conda 'shell.bash' 'hook')"
conda activate {conda_env}

{job_cmd}
