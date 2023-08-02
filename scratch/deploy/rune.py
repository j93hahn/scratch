import argparse
import subprocess
import json
import os.path as osp
from pathlib import Path
from tempfile import NamedTemporaryFile
from termcolor import cprint
from wandb.sdk.lib.runid import generate_id


_VALID_ACTIONS = ('run', 'cancel')
_DEFAULT_PARTITION = 'greg-gpu'


def load_cfg(fname=None):
    assert fname is not None
    with open(fname, "r") as f:
        cfg = json.load(f)
    return cfg


def load_template():
    template_fname = Path(__file__).resolve().parent / "sbatch_template.sh"
    with template_fname.open("r") as f:
        template = f.read()
    return template


def formatted_job_cmd(tdir: Path, user_job_cmd: str, ignore_config: bool):
    if ignore_config:   # use the user supplied job command as is
        assert user_job_cmd, "user_job_cmd must be specified if ignore_config is True"
        return user_job_cmd
    cfg = load_cfg(tdir / "config.json")
    job_cmd = ' '.join(cfg['script']) + ' '

    if user_job_cmd:    # append user supplied job command
        job_cmd += user_job_cmd

    # get values from the config file to format missing components in the job command
    vals = []
    for i in range(list(cfg).index('script') + 1, len(list(cfg))):
        vals.append(list(cfg.values())[i])
    return job_cmd.format(*vals)


def generate_script(tdir: Path, args: argparse.Namespace):
    """Generate the sbatch script as a string and return it."""
    if args.no_singleton:
        singleton = ""
    else:
        singleton = "#SBATCH -d singleton"

    job_cmd = formatted_job_cmd(tdir, args.job, args.ignore_config)

    script = load_template()
    return script.format(
        jname=generate_id(length=8),    # generate a random name for the job
        singleton=singleton,
        partition=args.partition,
        num_devices=args.num_cores,
        log_fname=Path(tdir) / args.log,
        task_dirname=tdir,
        conda_env=args.conda,
        job_cmd=job_cmd
    )


def main():
    parser = argparse.ArgumentParser(
        description="submit job(s) to slurm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', type=str, required=True,
        help='a json file containing a list of absolute paths to the job folders or a single path to a job folder'
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
        '-l', '--log', type=str, default="slurm.out",
        help='the log file to write the slurm output to'
    )
    parser.add_argument(
        '-c', '--conda', type=str, default="base",
        help='the conda environment to use'
    )
    parser.add_argument(
        '-P', '--print', action='store_true',
        help='print mode: print the slurm script to stdout instead of submitting it'
    )
    args = parser.parse_args()
    print('args={' + ', '.join(f'{k}={v}' for k, v in vars(args).items()) + '}')

    if args.action not in _VALID_ACTIONS:
        raise ValueError(
            f"action must be one of {_VALID_ACTIONS}, but given: {args.action}"
        )

    if args.file.endswith('.json'):
        edirs = load_cfg(args.file)
        print(f"Found {len(edirs)} experiments to {args.action} from {args.file}\n")
    else:   # single experiment sbatch submit
        _path = osp.abspath(args.file)
        assert osp.exists(_path) and osp.isdir(_path), \
            f"{_path} does not exist or is not a directory"
        edirs = [_path]
        print(f"Running single sbatch job in {edirs[0]}\n")

    _printed = False
    for tdir in edirs:
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
                sbatch_run(script)
                cprint(f"Submitted batch job named {tdir.name}", 'cyan')
        else:
            if args.print and not _printed:
                print("--- printing scancel command ---")
                print(f"scancel -n {tdir.name}\n")
                _printed = True

            if not args.print:
                sbatch_cancel(tdir.name)
                cprint(f"Cancelled batch job named {tdir.name}", 'red')


def sbatch_run(script: str):
    with NamedTemporaryFile(suffix='.sh') as sbatch_file:
        script = str.encode(script)
        sbatch_file.file.write(script)
        sbatch_file.file.seek(0)
        sbatch_script = sbatch_file.name
        # capture_output=True captures stdout/stderr in the log file (slurm.out)
        subprocess.run(f"sbatch {sbatch_script}", shell=True, capture_output=True)


def sbatch_cancel(jname):
    subprocess.run(f"scancel -n {jname}", shell=True, check=True)
