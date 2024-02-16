import torch
import numpy as np
import tinycudann as tcnn


def per_level_scale(max_res=4096, base_res=16, n_levels=16):
    scale = np.exp(
        (np.log(max_res) - np.log(base_res)) / (n_levels - 1)
    ).tolist()
    return scale


# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/nerfacto_field.py#L43


class DensityField(torch.nn.Module):
    def __init__(self, near, far, tau=1.0, T=0.99):
        super().__init__()
        self.offset = torch.tensor(
            np.log(np.log(1/T)) - np.log(far-near) - (tau**2)/2
        )
        self.encoder = tcnn.NetworkWithInputEncoding(   # instant-NGP backbone
            n_input_dims=3,
            n_output_dims=16,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale(),
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.B = torch.randn(
            3, 16, requires_grad=False
        )   # * scale
        self.color_field = RGBField()

    def forward(self, xyz, delta, viewdir):
        # check if xyz > 0 and xyz < 1
        selector = (xyz > 0) & (xyz < 1)
        raw = self.encoder(xyz[selector]) # [N_rays, N_samples, 16]
        raw = raw * selector

        raw = self.encoder(xyz) # [N_rays, N_samples, 16]
        log_densities, raw_features = raw[..., 0], raw[..., 1:]
        weights = self.compute_volrend_ws(log_densities, delta)
        rgb = self.color_field(raw_features)

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


class RGBField(torch.nn.Module):
    def __init__(self, type='mlp'):
        super().__init__()
        if type == 'mlp':
            self.encoder = tcnn.Network(
                n_input_dims=15 + 16,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        elif type == 'sh':
            self.encoder = ...  # TODO: figure out how this works
        else:
            raise ValueError(f"Unknown color field type: {type}")

    def forward(self, features):
        rgb = self.encoder(features)
        rgb = torch.sigmoid(rgb)
        return rgb


def get_final_image(weights, quantity):
    """
    a general purpose function to compute the final image from the weights and the quantity

    supports rgb, density, and normals
    """
    image = torch.sum(weights * quantity, dim=-1)
    return image
