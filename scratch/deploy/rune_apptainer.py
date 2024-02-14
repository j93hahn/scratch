import argparse
import subprocess
import json
import os.path as osp
from pathlib import Path
from tempfile import NamedTemporaryFile
from termcolor import cprint


_VALID_ACTIONS = ('run', 'cancel')
_DEFAULT_PARTITION = 'gpu'
_USER = 'jjahn'
_APPTAINER_DIR = '/share/data/2pals/jjahn/apptainers'
_CPU_PARTITIONS = ('cpu', 'cpu-long')


def load_json_log(fname=None) -> dict:
    assert fname is not None
    with open(fname, "r") as f:
        cfg = json.load(f)
    return cfg


def load_template():
    template_fname = Path(__file__).resolve().parent / "sbatch_apptainer.sh"
    with template_fname.open("r") as f:
        template = f.read()
    return template


def formatted_job_cmd(tdir: Path, user_job_cmd: str, ignore_config: bool):
    if ignore_config:   # use the user supplied job command as is
        assert user_job_cmd, "user_job_cmd must be specified if ignore_config is True"
        return user_job_cmd
    cfg = load_json_log(tdir / "config.json")
    job_cmd = ' '.join(cfg['script'])

    if user_job_cmd:    # append user supplied job command
        job_cmd += ' ' + user_job_cmd

    # get values from the config file to format missing components in the job command
    vals = []
    for i in range(list(cfg).index('script') + 1, len(list(cfg))):
        vals.append(list(cfg.values())[i])
    return job_cmd.format(*vals), cfg['job_id']


def generate_script(tdir: Path, args: argparse.Namespace):
    """Generate the sbatch script as a string and return it."""
    if args.no_singleton:
        singleton = ""
    else:
        singleton = "#SBATCH -d singleton"

    job_cmd, jname = formatted_job_cmd(tdir, args.job, args.ignore_config)
    if args.partition in _CPU_PARTITIONS:
        cores = f'-c{args.num_cores}'
    else:
        cores = f'-G{args.num_cores}'

    script = load_template()
    return script.format(
        jname=jname,
        singleton=singleton,
        partition=args.partition,
        cores=cores,
        # constraint=args.constraint,
        log_fname=Path(tdir) / args.log,
        tdir=tdir,
        container=(Path(_APPTAINER_DIR) / args.apptainer_image).as_posix(),
        job_cmd=job_cmd
    )


def main():
    parser = argparse.ArgumentParser(
        description="submit job(s) to slurm",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', type=str, required=True,
        help='a json file containing a list of absolute paths to the job folders OR a single path to a job folder'
    )
    parser.add_argument(
        '-a', '--action', default='run', choices=_VALID_ACTIONS,
        help='the action to perform'
    )
    parser.add_argument(
        '-p', '--partition', default=_DEFAULT_PARTITION, type=str,
        help='the job partition to submit to'
    )
    parser.add_argument(
        '-n', '--num-cores', type=int, default=1,
        help='number of cores to run the job'
    )
    parser.add_argument(
        '--no-singleton', action='store_true',
        help='if set, do not run the job as a singleton'
    )
    parser.add_argument(
        '-j', '--job', type=str, default="",
        help='supplements to the job command supplied in the config file'
    )
    parser.add_argument(
        '-i', '--ignore-config', action='store_true',
        help='ignore the config file and only use the user supplied job command'
    )
    parser.add_argument(
        '-l', '--log', type=str, default="beehive.out",
        help='the log file to write the output to'
    )
    parser.add_argument(
        '-im', '--apptainer-image', type=str, default="nerfstudio.sif",
        help='the apptainer to use'
    )
    # TODO: figure out how to build apptainer images that are agnostic to compute architecture
    # parser.add_argument(
    #     '-c', '--constraint', type=str, default="a4000|a6000",
    #     help='WARNING: do not use! the constraint to use on the hardware'
    # )
    parser.add_argument(
        '-P', '--print', action='store_true',
        help='print mode: print the slurm script to stdout instead of submitting it'
    )
    args = parser.parse_args()
    print('args={' + ', '.join(f'{k}={v}' for k, v in vars(args).items()) + '}')
    assert (Path(_APPTAINER_DIR) / args.apptainer_image).exists(), \
        f"{args.apptainer_image} is not a valid apptainer image"

    # load the experiment directories and corresponding job ids
    if args.file.endswith('.json'):
        edirs = load_json_log(args.file)
        print(f"Found {len(edirs)} experiments to {args.action} from {args.file}")
    else:   # single experiment sbatch submit
        _path = osp.abspath(args.file)
        assert osp.exists(_path) and osp.isdir(_path), \
            f"{_path} does not exist or is not a directory"
        _jname = load_json_log(osp.join(_path, "config.json"))['job_id']
        edirs = {_path: _jname}
        print(f"Running single slurm job in {_path}")

    # for each experiment, generate the slurm script and submit/print it
    _printed = False
    for tdir, jname in edirs.items():
        tdir = Path(tdir)
        assert tdir.exists() and tdir.is_dir(), \
            f"{tdir} does not exist or is not a directory"

        if args.action == 'run':
            script = generate_script(tdir, args)
            if args.print and not _printed:
                print("--- printing sbatch script ---")
                print(script)
                _printed = True

            if not args.print:
                if sbatch_is_running(jname):
                    cprint(f"Job {jname} has already been submitted", 'yellow')
                    continue

                sbatch_run(script)
                cprint(f"Submitted batch job {jname}", 'cyan')
        else:
            if args.print and not _printed:
                print("--- printing scancel command ---")
                print(f"scancel -n {jname}\n")
                _printed = True

            if not args.print:
                if not sbatch_is_running(jname):
                    cprint(f"Job {jname} is already not running", 'yellow')
                    continue

                sbatch_cancel(jname)
                cprint(f"Cancelled batch job {jname}", 'red')


def sbatch_run(script: str):
    with NamedTemporaryFile(suffix='.sh') as sbatch_file:
        script = str.encode(script)
        sbatch_file.file.write(script)
        sbatch_file.file.seek(0)
        sbatch_script = sbatch_file.name
        # capture_output=True captures stdout/stderr in the log file (default: slurm.out)
        subprocess.run(f"sbatch {sbatch_script}", shell=True, capture_output=True)


def sbatch_cancel(jname):
    subprocess.run(f"scancel -n {jname}", shell=True, check=True)


def sbatch_is_running(jname: str):
    """Check if the job ID has already been submitted to slurm."""
    res = subprocess.run(f"squeue -u {_USER} | grep {jname}", shell=True, capture_output=True)
    return res.returncode == 0
