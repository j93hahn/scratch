## Building an Apptainer

First, access a node with a GPU and set the following environment variables. Using `/scratch` is recommended for fast I/Os and SSD access.

```
export MYDIR=/scratch/$USER && mkdir -p $MYDIR && cd $MYDIR
export APPTAINER_CACHEDIR=$MYDIR/apptainer
export TMPDIR=$MYDIR/tmp
```

Now, construct a definition file. Please see the documentation [here](https://apptainer.org/docs/user/latest/) for more examples and in-depth explanations of the various components of the definition file. This file will be used to build the image, and it will contain all of the necessary software and dependencies for your environment. For example, to build an image with PyTorch 2.2.0 and CUDA 12.3, along with some system dependencies, copy the following into a file named `image.def`:
```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.01-py3

%environment
    DEBIAN_FRONTEND=noninteractive
    TZ=America/Chicago
    export DEBIAN_FRONTEND TZ

%post
    # Install system dependencies
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        git \
        imagemagick \
        libegl1 \
        libegl1-mesa-dev \
        libgl1 \
        libgl1-mesa-dev \
        libgles2 \
        libgles2-mesa-dev \
        libglvnd-dev \
        libglvnd0 \
        libglx0 \
        libnss3-dev \
        libopenexr-dev \
        libx264-dev \
        tmux \
        unzip \
        vim \
        wget \
        htop

    # Update pip
    pip install --upgrade pip
```

Now, build the image. We recommend using the `--sandbox` option to build a sandbox, which will allow you to both modify and test the image before deploying it to the cluster.
```
apptainer build --sandbox YOUR_IMAGE image.def
```

Modify the image as needed by installing software. Using `--writable` will allow you to modify the image, and `--no-home` will prevent the image from mounting your home directory. **Only install necessary packages for the environment here; the apptainer will still have access to the host filesystem and all of your external data.** For example, to install Python packages:
```
apptainer shell --writable --no-home YOUR_IMAGE
pip install numpy matplotlib pandas
```

Press `Ctrl+D` to exit the shell and return to the host. If you want to access the GPU to install packages with CUDA dependencies, pass `--nv`:
```
apptainer shell --writable --no-home --nv YOUR_IMAGE
pip install tensorflow-gpu
```

You can also run commands in the image without entering the shell by storing them in a bash script and passing it to `apptainer exec`:
```
echo "pip install numpy matplotlib pandas" > install.sh
apptainer exec --writable --no-home YOUR_IMAGE /bin/bash install.sh
```

If you want to install packages in the container as the root user, you can use the `--fakeroot` option. For example, to install `apache2` as root, use the following command:
```
apptainer exec --writable --no-home --fakeroot YOUR_IMAGE /bin/bash -c "sudo apt-get install -y --no-install-recommends apache2"
```

Once you are satisfied with the image, build it as an immutable SIF file. This will prevent you from modifying the image in the future, but it will also make it both easier and safer to deploy to the cluster as this file will not be able to edit or delete any of your data.
```
apptainer build FINAL_IMAGE.sif YOUR_IMAGE
```

Move the file to the cluster and remove the temporary directory:
```
export SAVEDIR=/path/to/your/save/directory
mv FINAL_IMAGE.sif $SAVEDIR
cd $HOME && rm -rf $MYDIR
```

## Using the Apptainer

The apptainer acts as a virtual environment like `conda` and has access to the host filesystem, so you can access your data and external code/software from within the container. For example, if you have a Python script that consumes data and saves some information to disk, you can do something like this:
```
export DATADIR=/path/to/your/data
export DISCLOC=/path/to/your/disc/save/location && mkdir -p $DISCLOC && cd $DISCLOC
apptainer exec $SAVEDIR/FINAL_IMAGE.sif /bin/bash -c "python /path/to/your/script.py --data $DATADIR"
```

Once the apptainer has executed the script, the results will be saved to `$DISCLOC` on the host filesystem. You can then access the results directly from the host without needing to copy them out of the container (as the container cannot modify the host filesystem).

## Submitting Jobs

To submit a job to the cluster, you can use the `apptainer exec` command in a job script. For example, to run a Python script that consumes data and saves some information to disk, you can create a job script like this:
```
#!/usr/bin/env bash

#SBATCH --job-name=$JOB_NAME
#SBATCH -d singleton

#SBATCH --partition=$PARTITION
#SBATCH -c $NUM_DEVICES
#SBATCH -C $CONSTRAINTS

#SBATCH --output=slurm.out
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

cd $DISCLOC

apptainer exec $SAVEDIR/FINAL_IMAGE.sif /bin/bash -c "python /path/to/your/script.py --data $DATADIR"
```

Assuming the above file is named `job.sh`, you can submit the job to the cluster with the following command: `sbatch job.sh`. Extending this to multiple jobs is as straightforward as creating multiple job scripts and submitting them all at once. The beauty of the apptainer is that all of these jobs can run simultaneously and access the same apptainer without interfering with each other or the host filesystem because the apptainer itself is immutable.
