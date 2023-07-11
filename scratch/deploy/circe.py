"""
Inspiration taken from Haochen Wang.

Designed for usage on the TTIC cluster. Given a template, this script will plant
various configuration files for different experiments into a single directory.
"""


import argparse
import json
import os
import sys
import itertools
import time
from datetime import datetime
from termcolor import cprint

_DEFAULT_COLOR = 'cyan'
_ALLOC_EXPERIMENTS_SPEC = {
    'required': ['uniform', 'singular', {'uniform': 'script'}],
    'ignore': ['_desc']
}


def main():
    parser = argparse.ArgumentParser(description="allocate experiment directories and plant configs")
    parser.add_argument(
        '-f', '--file', type=str, required=True,
        help='the json file containing the experiment specifications'
    )
    parser.add_argument(
        '-d', '--dir', type=str, default='./',
        help='the directory in which to plant experiment folders and configs'
    )
    parser.add_argument(
        '-l', '--log', type=str, default='allocated.json',
        help='a json containing a list of abspaths to allocated exps'
    )
    parser.add_argument(
        '-m', '--mock', action='store_true',
        help='mock mode: print the configs to stdout but do not plant them'
    )
    args = parser.parse_args()

    LAUNCH_FNAME = args.file
    LAUNCH_DIR_ABSPATH = os.path.abspath(args.dir)
    ALLOC_LOG_FNAME = os.path.join(LAUNCH_DIR_ABSPATH, args.log)

    with open(LAUNCH_FNAME, 'r') as f:
        launch_config = json.load(f)

    cfgs = extract_launch_config(launch_config)

    if args.mock:
        print('\nrunning in mock mode, will not plant experiment folders and configs\n')
        for cfg in cfgs:
            json.dump(cfg, sys.stdout, indent=4)
            sys.stdout.write('\n\n')
        return

    # create the launch directory if it doesn't exist
    if not os.path.isdir(LAUNCH_DIR_ABSPATH):
        os.mkdir(LAUNCH_DIR_ABSPATH)
        print(f'Created directory {LAUNCH_DIR_ABSPATH}')
    os.chdir(LAUNCH_DIR_ABSPATH)

    # plant the experiment folders and configs
    alloc_acc = []
    for cfg in cfgs:
        dir_name = datetime.now().strftime("%y_%m%d_%H%M_%S")
        cfg_abspath = os.path.join(LAUNCH_DIR_ABSPATH, dir_name)
        alloc_acc.append(cfg_abspath)
        plant_config(cfg, cfg_abspath)
        time.sleep(1)

    # write the allocation log
    with open(ALLOC_LOG_FNAME, 'w') as f:
        json.dump(alloc_acc, f, indent=4)
        f.write('\n')
        cprint(f'logged allocations to {ALLOC_LOG_FNAME}', color=_DEFAULT_COLOR)


def extract_launch_config(launch_config: dict) -> dict:
    verify_launch_config(launch_config, _ALLOC_EXPERIMENTS_SPEC)

    cfgs = cartesian_expand(launch_config['singular'])
    for i in range(len(cfgs)):
        # update the singular config with the uniform config if it exists
        cfgs[i] = {**launch_config['uniform'], **cfgs[i]}

    return cfgs


def verify_launch_config(launch_config: dict, requirements: dict):
    """Verifies the launch config dictionary against the requirements.

    Args:
        launch_config (dict): the launch config file
        requirements (dict): the requirements for the launch config

    Returns:
        Edits the launch config in place. Raises an error if the
            launch config does not meet the requirements.
    """
    def dfs_delete_key(key, cfg):
        if key in cfg:
            del cfg[key]
        for param in cfg:
            if isinstance(cfg[param], dict):
                dfs_delete_key(key, cfg[param])

    for key in requirements['ignore']:
        dfs_delete_key(key, launch_config)

    for key in requirements['required']:
        if isinstance(key, str):
            assert key in launch_config and launch_config[key], \
                f"field '{key}' is required in the launch config. Given {launch_config.keys()}"
        elif isinstance(key, dict):
            for subkey, subval in key.items():
                assert subval in launch_config[subkey] and launch_config[subkey][subval], \
                    f"field '{subval}' must be one of {subkey}. Given {launch_config[subkey]}"


def cartesian_expand(cfg: dict) -> list:
    """Expands a config dictionary into a list of configs. The length of the output
        is the product of the lengths of the values in the config dictionary.

    Args:
        cfg (dict): the config dictionary

    Returns:
        A list of configs.
    """
    output = []
    for x in itertools.product(*cfg.values()):
        output.append(dict(zip(cfg, x)))
    return output


def plant_config(cfg: dict, cfg_abspath: str):
    """Plants a config dictionary into a directory."""
    if not os.path.isdir(cfg_abspath):
        os.mkdir(cfg_abspath)
    with open(os.path.join(cfg_abspath, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
        f.write('\n')
        cprint(f"allocating: {cfg_abspath.split('/')[-1]}", color=_DEFAULT_COLOR)
