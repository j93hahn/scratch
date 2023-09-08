import argparse
import json
import os
import os.path as osp
import sys
import subprocess
import itertools
from functools import reduce
from .rune import load_json_log
from termcolor import cprint
from wandb.sdk.lib.runid import generate_id


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
        '-m', '--mode', type=str, default='cartesian',
        help='the mode in which to expand the experiment specifications, should be one of cartesian, monopole, c* or m* which support hybrid expansion'
    )
    parser.add_argument(
        '-P', '--print', action='store_true',
        help='print mode: print the configs to stdout but do not plant them'
    )
    args = parser.parse_args()

    LAUNCH_FNAME = args.file
    LAUNCH_DIR_ABSPATH = osp.abspath(args.dir)
    ALLOC_LOG_FNAME = osp.abspath(args.log)

    # load the launch config and extract each experiment's specifications
    launch_config = load_json_log(LAUNCH_FNAME)
    cfgs = extract_from_launch_config(launch_config, args.mode)

    if args.print:
        print("--- printing example directory and planted config file ---\n")
        print(f"exp directory: {osp.join(LAUNCH_DIR_ABSPATH, cfgs[0]['name'])}")
        json.dump(cfgs[0], sys.stdout, indent=4)
        sys.stdout.write('\n\n')
        return

    # create the launch directory if it doesn't exist
    if not osp.isdir(LAUNCH_DIR_ABSPATH):
        os.mkdir(LAUNCH_DIR_ABSPATH)
        print(f'Created directory {LAUNCH_DIR_ABSPATH}')
    os.chdir(LAUNCH_DIR_ABSPATH)

    # plant the experiment folders and configs
    alloc_acc = {}
    overwrite = "n"
    for cfg in cfgs:
        dir_name = cfg['name']
        del cfg['name']
        cfg_abspath = osp.join(LAUNCH_DIR_ABSPATH, dir_name)

        jname = generate_id(length=8)       # generate a unique id for each experiment
        alloc_acc[cfg_abspath] = jname
        cfg = {**{'job_id': jname}, **cfg}  # add the job id to the beginning of the config

        skip = False    # handle overwriting logic of existing directories
        if osp.exists(cfg_abspath) and osp.isdir(cfg_abspath):
            if overwrite != "a":
                ans = input(f"directory {cfg_abspath} already exists --> overwrite? [y/n/a] ")
                if ans == "a":
                    overwrite = "a"
                    cprint("overwriting all remaining existing directories...", color='yellow')
                    subprocess.run(f'rm -rf {cfg_abspath}', shell=True)
                elif ans == "y":
                    subprocess.run(f'rm -rf {cfg_abspath}', shell=True)
                else:
                    cprint(f"skipping {cfg_abspath}", color=_DEFAULT_COLOR)
                    skip = True
            else:
                subprocess.run(f'rm -rf {cfg_abspath}', shell=True)

        if not skip:
            plant_config(cfg, cfg_abspath)

    # write the allocation log containing all experiments and their job ids
    with open(ALLOC_LOG_FNAME, 'w') as f:
        json.dump(alloc_acc, f, indent=4)
        f.write('\n')
        cprint(f'logged {len(alloc_acc)} allocations to {ALLOC_LOG_FNAME}', color=_DEFAULT_COLOR)


def extract_from_launch_config(launch_config: dict, mode: str) -> dict:
    verify_launch_config(launch_config, _ALLOC_EXPERIMENTS_SPEC)

    if mode == 'cartesian':
        cfgs = cartesian_expansion(launch_config['singular'])
    elif mode == 'monopole':
        cfgs = monopole_expansion(launch_config['singular'])
    else:
        assert mode[0] in ['c', 'm'] and mode[1:].isnumeric(), \
            f"mode must be one of 'c*' or 'm*'. Given {mode}"
        cfgs = hybrid_expansion(launch_config['singular'], mode)

    # update each config with the job/experiment folder name
    for i in range(len(cfgs)):
        _exp_folder = {'name': '_'.join(map(str, list(cfgs[i].values()))).replace('/', '_')}
        cfgs[i] = {**_exp_folder, **launch_config['uniform'], **cfgs[i]}

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
        the output is the length of the shortest list among the singular config's keys.

    Args:
        cfg (dict): the config dictionary

    Returns:
        A list of configs. Note that the length of the output is the length of the longest list.
    """
    output = []
    for x in list(zip(*cfg.values())):
        output.append(dict(zip(cfg, x)))
    return output


def hybrid_expansion(cfg: dict, mode: str) -> list:
    """Expands a config dictionary into a list of configs via hybrid expansion. The user specifies
        which keys to apply cartesian/monopole expansion to, and the opposite method of expansion
        is applied to the output of the first operation and the remaining keys.

    Args:
        cfg (dict): the config dictionary

    Returns:
        A list of configs.
    """
    # remove duplicate indices and verify that they are in bounds
    indices = list(set(reduce(lambda acc, x: acc + [int(x)], mode[1:], [])))
    for i in range(len(indices)):
        assert indices[i] >= 0 and indices[i] < len(cfg), \
            f"index {indices[i]} is out of bounds for config with length {len(cfg)}"

    # extract the initial keys and the final keys
    initial_keys = [list(cfg.keys())[index] for index in indices]
    initial_dict = {key: cfg[key] for key in initial_keys}
    final_dict = {key: cfg[key] for key in cfg if key not in initial_keys}

    output = []
    if mode[0] == 'c':  # not a recommended recipe
        cprint("WARNING: hybrid expansion mode {c*} is not recommended", color='red')
        _I, _F = cartesian_expansion(initial_dict), monopole_expansion(final_dict)
        while _I and _F:
            _dict = {**_I.pop(0), **_F.pop(0)}
            output.append({key: _dict[key] for key in list(cfg.keys())})
    else:
        _output = monopole_expansion(initial_dict)
        _output = [{key: [_dict[key]] for key in _dict} for _dict in _output]
        _output = [dict(**x, **final_dict) for x in _output]
        for _dict in _output:
            _dict = {key: _dict[key] for key in list(cfg.keys())}   # restore the original key order
            output.append(cartesian_expansion(_dict))               # cartesian expansion of the remaining keys
        output = [item for sublist in output for item in sublist]   # flatten the list
    return output


def plant_config(cfg: dict, cfg_abspath: str):
    """Plants a config dictionary into a directory."""
    os.mkdir(cfg_abspath)
    with open(osp.join(cfg_abspath, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
        f.write('\n')
        cprint(f"allocating: {cfg_abspath.split('/')[-1]}", color=_DEFAULT_COLOR)
