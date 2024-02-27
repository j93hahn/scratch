import os
import gin
import dataclasses
from absl import flags


@gin.configurable
@dataclasses.dataclass
class Config:
    # configs for experiment logistics
    seed:                   int = 42
    name:                   str = "test"
    save_dir:               str = "./"
    checkpoint_dir:         str = "ckpts"
    checkpoint_every:       int = 10000
    data_dir:               str = "./nerf_synthetic/lego"
    dataset:                str = "blender"

    # logging parameters
    update_pbar:            int = 10
    render_train_image:     int = 100
    render_test_image:      int = 1000
    render_full_test_set:   int = 10000

    # adam optimization/scheduler params on the fields
    max_steps:              int = 30000
    lr_init:                float = 5e-2
    lr_final:               float = 5e-4
    lr_delay_steps:         int = 1000
    lr_delay_mult:          int = 1
    beta1:                  float = 0.9
    beta2:                  float = 0.999
    eps:                    float = 1e-8

    # density field parameters - assume default instant-NGP backbone
    use_tcnn_density:       bool = True
    hash_n_levels:          int = 16
    hash_feats_per_level:   int = 2
    hash_map_log_size:      int = 19
    hash_base_res:          int = 16
    hash_output_dim:        int = 16
    max_grid_res:           int = 4096
    mlp_n_neurons:          int = 64
    mlp_n_layers:           int = 1

    # color field parameters
    use_tcnn_color:         bool = True
    color_n_layers:         int = 3
    color_n_neurons:        int = 64
    ff_sigma:               float = 1.0     # std of the fourier features for viewdirs
    ff_feat_dim_viewdirs:   int = 16        # dimension of the fourier features for viewdirs

    # rendering parameters
    use_scene_contraction:  bool = True
    ord_scene_contraction:  str = 'inf'
    aabb:                   list[list[float]] = [[-1., -1., -1.], [1., 1., 1.]]
    near:                   float = 0.05
    far:                    float = 10000.0
    init_trans:             float = 0.99
    tau:                    float = 0.05    # empirically doesn't impact rendering

    # sampling parameters
    use_prop_samp:          bool = True


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
