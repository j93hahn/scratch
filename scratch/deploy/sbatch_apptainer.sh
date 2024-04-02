#!/usr/bin/env bash

#SBATCH --job-name={jname}
{singleton}

#SBATCH --partition={partition}
#SBATCH {cores}

#SBATCH --output={log_fname}
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

cd {tdir}

apptainer exec --nv --no-home --bind /share:/share {container} \
    /bin/bash -c "source /conda_tmp/mc3/bin/activate; \
                  export TORCH_EXTENSIONS_DIR=/share/data/2pals/jjahn/torch_cuda; \
                  {job_cmd}"
