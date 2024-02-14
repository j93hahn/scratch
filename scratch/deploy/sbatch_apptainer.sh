#!/usr/bin/env bash

#SBATCH --job-name={jname}
{singleton}

#SBATCH --partition={partition}
#SBATCH {cores}

#SBATCH --output={log_fname}
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

cd {tdir}

apptainer exec --nv --no-home --bind /share/data/2pals:/share/data/2pals {container} /bin/bash -c "{job_cmd}"
