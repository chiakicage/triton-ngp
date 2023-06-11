import torch
from torch import nn
from typing import Optional, Tuple

class TorchMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        activation: nn.Module = nn.ReLU(),
        skip_connection: Optional[Tuple[int]] = None,
        out_activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.out_activation = out_activation
        self.skip_connection = set(skip_connection) if skip_connection is not None else set()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
            return
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(1, num_layers - 1):
            if i in self.skip_connection:
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, in_tensor: torch.Tensor):
        x = in_tensor
        for i, layer in enumerate(self.layers):
            if i in self.skip_connection and i != 0:
                x = torch.cat([x, in_tensor], dim=-1)
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
