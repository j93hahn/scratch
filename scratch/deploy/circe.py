"""
Inspiration taken from Haochen Wang.

Designed for usage on the TTIC cluster. Given a template, this script will plant
various configuration files for different experiments into a single directory.
"""


import argparse
import json
import os
import os.path as osp
import sys
import itertools
from termcolor import cprint
from .rune import load_cfg


_DEFAULT_COLOR = 'cyan'
_ALLOC_EXPERIMENTS_SPEC = {
    'required': ['uniform', 'singular', {'uniform': 'script'}],
    'ignore': ['_desc']
}


def main():
    parser = argparse.ArgumentParser(
        description="allocate experiment directories and plant configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', type=str, required=True,
        help='the json file containing the experiment specifications'
    )
    parser.add_argument(
        '-d', '--dir', type=str, default='runs',
        help='the directory in which to plant experiment folders and configs'
    )
    parser.add_argument(
        '-l', '--log', type=str, default='allocated.json',
        help='a json containing a list of abspaths to allocated exps'
    )
    parser.add_argument(
        '-m', '--mode', type=str, default='cartesian', choices=['cartesian', 'monopole'],
        help='the mode in which to expand the experiment specifications'
    )
    parser.add_argument(
        '-P', '--print', action='store_true',
        help='print mode: print the configs to stdout but do not plant them'
    )
    args = parser.parse_args()

    LAUNCH_FNAME = args.file
    LAUNCH_DIR_ABSPATH = osp.abspath(args.dir)
    ALLOC_LOG_FNAME = osp.abspath(args.log)

    launch_config = load_cfg(LAUNCH_FNAME)
    cfgs = extract_from_launch_config(launch_config, args.mode)

    if args.print:
        print('\nrunning in print mode, will not plant experiment folders and configs\n')
        for cfg in cfgs:
            json.dump(cfg, sys.stdout, indent=4)
            sys.stdout.write('\n\n')
        return

    # create the launch directory if it doesn't exist
    if not osp.isdir(LAUNCH_DIR_ABSPATH):
        os.mkdir(LAUNCH_DIR_ABSPATH)
        print(f'Created directory {LAUNCH_DIR_ABSPATH}')
    os.chdir(LAUNCH_DIR_ABSPATH)

    # plant the experiment folders and configs
    alloc_acc = []
    for cfg in cfgs:
        dir_name = cfg['name']
        del cfg['name']
        cfg_abspath = osp.join(LAUNCH_DIR_ABSPATH, dir_name)
        alloc_acc.append(cfg_abspath)
        plant_config(cfg, cfg_abspath)

    # write the allocation log
    with open(ALLOC_LOG_FNAME, 'w') as f:
        json.dump(alloc_acc, f, indent=4)
        f.write('\n')
        cprint(f'logged allocations to {ALLOC_LOG_FNAME}', color=_DEFAULT_COLOR)


def extract_from_launch_config(launch_config: dict, mode: str) -> dict:
    verify_launch_config(launch_config, _ALLOC_EXPERIMENTS_SPEC)

    if mode == 'cartesian':
        cfgs = cartesian_expansion(launch_config['singular'])
    elif mode == 'monopole':
        cfgs = monopole_expansion(launch_config['singular'])
    for i in range(len(cfgs)):
        # update the singular config with job/experiment name and the uniform config
        name = {'name': '_'.join(map(str, list(cfgs[i].values())))}
        cfgs[i] = {**name, **launch_config['uniform'], **cfgs[i]}

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

    # verify that the last field in the uniform config is 'script'; a requirement for
    # the generate_script() command in rune.py
    assert 'script' == list(launch_config['uniform'].keys())[-1], \
        f"the last field in the uniform config must be 'script'. Given {launch_config['uniform'].keys()[-1]}"


def cartesian_expansion(cfg: dict) -> list:
    """Expands a config dictionary into a list of configs via cartesian expansion. The length of
        the output is the product of the lengths of each key's list in the singular config.

    Args:
        cfg (dict): the config dictionary

    Returns:
        A list of configs.
    """
    output = []
    for x in itertools.product(*cfg.values()):
        output.append(dict(zip(cfg, x)))
    return output


def monopole_expansion(cfg: dict) -> list:
    """Expands a config dictionary into a list of configs via monopole expansion. The length of
        the output is the length of the longest list in the singular config.

    Args:
        cfg (dict): the config dictionary

    Returns:
        A list of configs. Note that the length of the output is the length of the longest list.
    """
    output = []
    for x in list(zip(*cfg.values())):
        output.append(dict(zip(cfg, x)))
    return output


def plant_config(cfg: dict, cfg_abspath: str):
    """Plants a config dictionary into a directory."""
    if not osp.isdir(cfg_abspath):
        os.mkdir(cfg_abspath)
    with open(osp.join(cfg_abspath, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
        f.write('\n')
        cprint(f"allocating: {cfg_abspath.split('/')[-1]}", color=_DEFAULT_COLOR)
