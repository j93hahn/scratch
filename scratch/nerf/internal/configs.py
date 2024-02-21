import os
import gin
import dataclasses
from absl import flags


@gin.configurable
@dataclasses.dataclass
class Config:   # configuration class for the entire project
    seed:                   int = 42
    name:                   str = "test"
    save_dir:               str = "./"
    checkpoint_dir:         str = "ckpts"
    data_dir:               str = "./nerf_synthetic/lego"
    dataset:                str = "blender"

    max_steps:              int = 25000
    lr_init:                float = 5e-2
    lr_final:               float = 5e-4
    lr_delay_steps:         int = 1000
    lr_delay_mult:          int = 1
    beta1:                  float = 0.9
    beta2:                  float = 0.999
    eps:                    float = 1e-8


def define_flags():
    flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
    flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def load_config():
    gin.parse_config_files_and_bindings(
        flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True
    )
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(f"{config.checkpoint_dir}/config.gin", "w") as f:
        f.write(gin.config_str())
    return config
