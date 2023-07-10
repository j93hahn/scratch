"""
Inspiration taken from Haochen Wang.

Designed for usage on the TTIC cluster. This script will run the experiments
using the parameters specified in the config file.
"""


from pathlib import Path
import subprocess

from termcolor import cprint
import yaml

from scratch.utils.mailer import exp_fail, send_email
import subprocess
import yaml


def run(file):
    with open(file) as f:
        file_ = yaml.safe_load(f)

    dataset, scale = file_['arch']['dataset'], file_['arch']['scale']

    ret = subprocess.run(f'python /share/data/pals/jjahn/nerf-pytorch/run_nerf.py --config /share/data/pals/jjahn/nerf-pytorch/configs/{dataset}.txt --distance_scale {scale}', shell=True, capture_output=True)
    if ret.returncode != 0:
        exp_fail()
    else:
        send_email(subject=f"Experiment {dataset} with scale {scale} succeeded", body=ret.stdout.decode('utf-8'), to="jjahn@uchicago.edu")


# if __name__ == '__main__':
#     run('config.yml')

def exec_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


def load_cfg(fname=None):
    assert fname is not None
    with open(fname, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    return


if __name__ == '__main__':
    exec_cmd("python deploy.py")


commands = {
    'tensorf':  """
                python /share/data/pals/jjahn/TensoRF/train.py \
                    --config /share/data/pals/jjahn/tensoRF/configs/{}.txt \
                    --distance_scale {} \
                    --fea2denseAct {}
                """,
    'nerfacto': """
                python /share/data/pals/jjahn/nerfstudio/nerfstudio/scripts/train.py nerfacto \
                    --vis wandb --data /share/data/pals/jjahn/data/blender/lego --experiment-name blender_lego \
                    --relative-model-dir nerfstudio_models --steps-per-save 0 --pipeline.model.background-color white \
                    --pipeline.model.proposal-initial-sampler uniform \
                    --pipeline.model.near-plane 2. --pipeline.model.far-plane 6. \
                    --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False \
                    --pipeline.model.distortion-loss-mult 0 --pipeline.model.disable-scene-contraction True \
                    --steps_per_eval_all_images 0 blender-data
                """,
    'mipnerf':  """
                python
                """,
    'nerf':     """
                python
                """,
    'dvgo':     """
                python
                """,
    'plenoxel': """
                python
                """,
}


# def main():
#     code_root = dir_of_this_file(__file__)
#     cfg = load_my_cfg()

#     cmd = template.format(
#         code_root=str(code_root), **cfg
#     )
#     print(cmd)
#     exec_cmd(cmd)


# if __name__ == "__main__":
#     main()


# from my_utils import exec_cmd, dir_of_this_file, load_my_cfg, gpu_list_str


# def main():
#     code_root = dir_of_this_file(__file__)
#     cfg = load_my_cfg()

#     prompt = cfg['prompt']
#     model = cfg['model']
#     per_rank_bs = cfg['per_rank_bs']

#     gpus = gpu_list_str()

#     cfg_name = {
#         "sd": "dreamfusion-sd.yaml",
#         "if": "dreamfusion-if.yaml",
#         "pd": "prolificdreamer.yaml",
#     }[model]

#     cmd = f"""
#     python {code_root}/launch.py --config configs/{cfg_name} \
#         --train --gpu {gpus} \
#         system.prompt_processor.prompt="{prompt}" \
#         data.batch_size={per_rank_bs}
#     """
#     print(cmd)
#     exec_cmd(cmd)


# if __name__ == "__main__":
#     main()