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
import logging

logger = logging.getLogger("nerf")


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
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_density(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_mlp_output = self.base_mlp(self.xyz_encoder(positions))
        density, base_mlp_output = torch.split(
            base_mlp_output, [1, base_mlp_output.shape[-1] - 1], dim=-1
        )
        density = trunc_exp(density).view(-1)
        return density, base_mlp_output

    def get_rgb(self, directions: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        rgb = self.head_mlp(torch.cat([self.dir_encoder(directions), mlp_out], dim=-1))
        return rgb

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        density, mlp_out = self.get_density(positions)
        rgb = self.get_rgb(directions, mlp_out)
        return rgb, density


    
