"""
resnet.py
Residual Network module.
"""

import torch
from torch import nn
from torch import Tensor
from dbgym.db import Tabular
from yacs.config import CfgNode
from dbgym.register import register


class ResNet(nn.Module):
    """
    Residual Network module.
    """

    def __init__(self, cfg: CfgNode, data: Tabular):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.model.output_dim
        self.layer = cfg.model.layer

        max_values = torch.max(data.x_d, dim=0).values
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_values[i]+1, hidden_dim)
            for i in range(data.x_d.size(1))
        ])
        self.linear = nn.Linear(data.x_c.shape[1], hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(0, self.layer - 1):
            norm = nn.LayerNorm(hidden_dim)
            layer1 = nn.Linear(hidden_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(nn.ModuleList([norm, layer1, layer2]))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, data: Tabular) -> Tensor:
        """
        Forward propagation function

        Args:
        - data: tabular data

        Returns:
        - out: output data
        """

        x_d = [self.embeddings[j](data.x_d[:, j]) for j in range(data.x_d.size(1))]
        x_d = torch.sum(torch.stack(x_d, dim=2), dim=2)
        x = x_d + self.linear(data.x_c)

        for i in range(0, self.layer - 1):
            z = self.layers[i][0](x)
            z = self.layers[i][1](z)
            z = nn.functional.relu(z)
            z = self.layers[i][2](z)
            z = nn.functional.relu(z)
            x = x + z

        x = self.layers[-1](x)
        return x

register('tabular_model', 'ResNet', ResNet)
