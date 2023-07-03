from pathlib import Path
import subprocess

from termcolor import cprint
import yaml


def exec_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


def load_cfg(fname=None):
    assert fname is not None
    with open(fname, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == '__main__':
    exec_cmd("python deploy.py")


# template = """
# python {code_root}/launch.py --config configs/zero123.yaml \
#     --train --gpu 0 system.loggers.wandb.enable=false \
#     data.image_path=/whc/load_3s/images/{img_name}_rgba.png \
#     system.freq.guidance_eval=0 \
#     system.guidance.pretrained_model_name_or_path="/whc/load_3s/zero123/105000.ckpt" \
#     system.guidance.cond_elevation_deg={elev_deg}
# """


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