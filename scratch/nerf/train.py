import torch
from internal import fields
from internal import configs
from internal import datasets
from internal import train_utils
from lightning import seed_everything
import gin
import tyro
from absl import flags, app
from scratch.utils.wizard import WandbWizard
import jax


def main():
    config = configs.load_config()
    seed_everything(config.seed)
    wizard = WandbWizard(config.name, config.save_dir)
    dataset = datasets.dataset_dict[config.dataset](config.data_dir)
    scheduler =


if __name__ == '__main__':
    configs.define_flags()
    with gin.config_scope('train'):
        app.run(main)
