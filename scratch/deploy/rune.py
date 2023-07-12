"""
Inspiration taken from Haochen Wang.

Designed for usage on the TTIC cluster. This script will run the experiments
using the parameters specified in the config file and command line.
"""


import argparse
import subprocess
import json
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from termcolor import cprint
from scratch.utils.mailer import send_email


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


def generate_script(tdir: Path, args: argparse.Namespace):
    """Generate the sbatch script as a string and return it."""
    if args.nost:
        singleton = ""
    else:
        singleton = "#SBATCH -d singleton"

    with open(tdir / "config.json", "r") as f:
        cfg = json.load(f)
        job_cmd = ' '.join(cfg['script']) + ' '

    if args.job:
        # get values from the config file to format missing components in the job command
        vals = []
        for i in range(list(cfg).index('script') + 1, len(list(cfg))):
            vals.append(list(cfg.values())[i])
        job_cmd += args.job.format(*vals)

    script = load_template()
    return script.format(
        jname=tdir.name, singleton=singleton,
        partition=args.partition, num_devices=args.num_cores,
        log_fname=Path(tdir) / args.log,
        task_dirname=tdir, conda_env=args.conda,
        job_cmd=job_cmd
    )


def main():
    parser = argparse.ArgumentParser(description="run the experiments")
    parser.add_argument(
        '-f', '--file', type=str, required=True,
        help='a json file containing a list of absolute paths to the job folders'
    )
    parser.add_argument(
        '-a', '--action', default='run',
        help='one of {}, default {}'.format(_VALID_ACTIONS, _VALID_ACTIONS[0])
    )
    parser.add_argument(
        '-p', '--partition', default=_DEFAULT_PARTITION, type=str,
        help='the job partition. default {}'.format(_DEFAULT_PARTITION)
    )
    parser.add_argument(
        '-n', '--num-cores', type=int, default=1,
        help='Number of cores to run the job.'
    )
    parser.add_argument(
        '-j', '--job', type=str, required=False, default="",
        help='supplements to the job command supplied in the config file'
    )
    parser.add_argument(
        '-l', '--log', type=str, required=False, default="slurm.out",
        help='store the slurm output in this file'
    )
    parser.add_argument(
        '-c', '--conda', type=str, required=False, default="base",
        help='the conda environment to use'
    )
    parser.add_argument(
        '-m', '--mock', default=False, action='store_true',
        help='in mock mode the slurm command is printed but not executed'
    )
    parser.add_argument(
        '--nost', default=False, action='store_true',
        help='do use the singleton option'
    )
    args = parser.parse_args()
    print(args)

    if args.mock:
        print("WARN: using mock mode")

    if args.action not in _VALID_ACTIONS:
        raise ValueError(
            f"action must be one of {_VALID_ACTIONS}, but given: {args.action}"
        )

    edirs = load_cfg(args.file)
    print(f"Found {len(edirs)} experiments to {args.action}.")

    for tdir in edirs:
        tdir = Path(tdir)
        assert tdir.exists() and tdir.is_dir(), \
            f"{tdir} does not exist or is not a directory"

        script = generate_script(tdir, args)

        if args.mock:
            print(script)
            return

        if args.action == 'cancel':
            sbatch_cancel(tdir.name)
            cprint(f"Cancelled job in folder {tdir.name}", 'red')
        else:
            sbatch_run(tdir, script)
            cprint(f"Submitted job in folder {tdir.name}", 'cyan')


def sbatch_run(tdir: Path, script: str):
    with NamedTemporaryFile(suffix='.sh') as sbatch_file:
        script = str.encode(script)
        sbatch_file.file.write(script)
        sbatch_file.file.seek(0)
        sbatch_script = sbatch_file.name
        ret = subprocess.run(f"sbatch {sbatch_script}", shell=True, capture_output=True)
        if ret.returncode != 0:
            send_email("Exp failed", f"Experiment {tdir.name} failed.")
        else:
            send_email("Exp success", f"Experiment {tdir.name} succeeded.")


def sbatch_cancel(jname):
    subprocess.run(f"scancel -n {jname}", shell=True, check=True)
