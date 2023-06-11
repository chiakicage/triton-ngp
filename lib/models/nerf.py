import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.encoder import make_encoder
from lib.models.mlp import TorchMLP
from lib.config.task.nerf import NeRFConfig
from lib.models.activation import trunc_exp
from typing import Tuple
import nerfacc


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.xyz_encoder = make_encoder(cfg.network.xyz_encoder)
        self.dir_encoder = make_encoder(cfg.network.dir_encoder)
        self.base_mlp = TorchMLP(
            input_dim=self.xyz_encoder.output_dim,
            output_dim=cfg.network.base_mlp_W,
            hidden_dim=cfg.network.base_mlp_W,
            num_layers=cfg.network.base_mlp_D,
            activation=nn.ReLU(),
            out_activation=None,
        )
        self.head_mlp = TorchMLP(
            input_dim=cfg.network.base_mlp_W - 1 + self.dir_encoder.output_dim,
            output_dim=3,
            hidden_dim=cfg.network.head_mlp_W,
            num_layers=cfg.network.head_mlp_D,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
        )

    def get_density(self, position: torch.Tensor):
        base_mlp_output = self.base_mlp(self.xyz_encoder(position))
        density, base_mlp_output = torch.split(
            base_mlp_output, [1, base_mlp_output.shape[-1] - 1], dim=-1)
        density = trunc_exp(density)
        return density, base_mlp_output

    def get_rgb(self, direction: torch.Tensor, mlp_out: torch.Tensor):
        rgb = self.head_mlp(
            torch.cat([self.dir_encoder(direction), mlp_out], dim=-1))
        return rgb

    def forward(self, rays_o, rays_d, estimator):
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_directions = rays_d[ray_indices]
            positions = t_origins + t_directions * (t_starts + t_ends)[:, None] / 2.0
            density, _ = self.get_density(positions)
            return density
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_directions = rays_d[ray_indices]
            positions = t_origins + t_directions * (t_starts + t_ends)[:, None] / 2.0
            density, mlp_out = self.get_density(positions)
            rgb = self.get_rgb(t_directions, mlp_out)
            return rgb, density
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o, rays_d, sigma_fn=sigma_fn, early_stop_eps=1e-4, alpha_thre=1e-2
        )

        rgb, density, depth, extras = nerfacc.rendering(
            t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
        )

        return rgb, density, depth, extras

