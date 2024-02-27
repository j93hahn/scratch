import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn
from internal import utils
from internal import configs
from internal import mlp


def per_level_scale(max_res, base_res, n_levels):
    scales = np.exp(
        (np.log(max_res) - np.log(base_res)) / (n_levels - 1)
    ).tolist()
    return scales


class DensityField(nn.Module):
    def __init__(self, config: configs.Config):
        super().__init__()
        self.use_scene_contraction = config.use_scene_contraction
        self.raw_feat_dim = config.hash_output_dim - 1
        if not self.use_scene_contraction:
            self.scene_box = utils.SceneBox(config.aabb)
            self.offset = torch.tensor(
                np.log(np.log(1/config.init_trans)) - np.log(config.far-config.near) - (config.tau**2)/2
            )
        else:
            self.scene_contraction = utils.SceneContraction(config.ord_scene_contraction)
            self.offset_inner = torch.tensor(
                np.log(np.log(1/config.init_trans)) - np.log(2) - (config.tau**2)/2
            )
            self.offset_outer = torch.tensor(
                np.log(np.log(1/config.init_trans)) - np.log(4) - (config.tau**2)/2
            )
        if config.use_tcnn_density:
            per_level_scale_configs = {
                'max_res': config.max_grid_res,
                'base_res': config.hash_base_res,
                'n_levels': config.hash_n_levels,
            }
            self.encoder = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=config.hash_output_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": config.hash_n_levels,
                    "n_features_per_level": config.hash_feats_per_level,
                    "log2_hashmap_size": config.hash_map_log_size,
                    "base_resolution": config.hash_base_res,
                    "per_level_scale": per_level_scale(per_level_scale_configs),
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": config.mlp_n_neurons,
                    "n_hidden_layers": config.mlp_n_layers,
                },
            )
        else:   # TODO: implement Pythonic HashEncoding
            raise ValueError("Only TCNN is supported at the moment")

    def forward(self, positions, delta):
        if not self.scene_contraction:
            positions = self.scene_box(positions)
        else:
            positions = self.scene_contraction(positions)
            positions = (positions + 2.0) / 4.0
        selector = ((positions > 0) & (positions < 1)).all(dim=-1)
        positions = positions * selector[..., None]
        if not positions.requires_grad:
            positions.requires_grad = True
        raw = self.encoder(positions)
        log_densities, raw_features = torch.split(raw, [1, self.raw_feat_dim], dim=-1)
        weights = self.compute_volrend_ws(log_densities, delta)
        return weights, raw_features

    def compute_volrend_ws(self, log_densities, delta):
        # log_densities, delta have shape [N_rays, N_samples]
        density_delta = torch.exp(
            log_densities +     # invariant to scale
            self.offset +       # high transmittance
            torch.log(delta)    # gumbelCDF
        )
        trans = torch.cumsum(density_delta[..., :-1], dim=-1)
        trans = torch.cat(
            [torch.zeros(*trans.shape[:1], 1), trans], dim=-1
        )
        trans = torch.exp(-trans)
        alpha = 1. - torch.exp(-density_delta)
        weights = alpha * trans
        return weights


class ColorField(nn.Module):
    def __init__(self, config: configs.Config):
        super().__init__()
        self.B = torch.randn(config.ff_feat_dim_viewdirs, 3) * config.ff_sigma
        self.B.requires_grad_(False)
        self.raw_feat_dim = config.hash_output_dim - 1
        if config.use_tcnn_color:
            self.encoder = tcnn.Network(
                n_input_dims=self.raw_feat_dim + config.ff_feat_dim_viewdirs * 2,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": config.color_n_neurons,
                    "n_hidden_layers": config.color_n_layers,
                }
            )
        else:
            self.encoder = nn.Sequential(
                mlp.Linear(self.raw_feat_dim + config.ff_feat_dim_viewdirs * 2, config.color_n_neurons, init_bias=1.0),
                mlp.ReLU(),
                mlp.Linear(config.color_n_neurons, config.color_n_neurons, init_bias=1.0),
                mlp.ReLU(),
                mlp.Linear(config.color_n_neurons, config.color_n_neurons, init_bias=1.0),
                mlp.ReLU(),
                mlp.Linear(config.color_n_neurons, 3, init_bias=1.0),
            )

    def forward(self, raw_features, viewdirs):
        # encode viewdirs with positional encoding
        viewdirs = utils.fourier_features(viewdirs, self.B)
        rgb = self.encoder(raw_features)
        rgb = torch.sigmoid(rgb)
        return rgb
