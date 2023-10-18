"""
gnn.py
Graph neural network module.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv


class GNN(torch.nn.Module):
    """
    Graph neural network module
    """

    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.model.input_dim
        output_dim = cfg.model.output_dim
        hidden_dim = cfg.model.hidden_dim
        layer = cfg.model.layer
        self.head = cfg.model.head
        self.model = cfg.model.subtype

        self.convs = nn.ModuleList()
        self.linear = nn.Linear(input_dim, hidden_dim)
        for _ in range(layer - 1):
            self.convs.append(self.create_conv(hidden_dim, hidden_dim))
        self.convs.append(self.create_conv(hidden_dim, output_dim))

    def create_conv(self, input_dim, output_dim) -> nn.Module:
        """
        Create convolution function

        Args:
        - input_dim: input dimension
        - output_dim: output dimension

        Returns:
        - conv: output convolution module
        """
        if self.model == "GCN":
            return GCNConv(input_dim, output_dim)
        if self.model == "GIN":
            return GINConv(nn = nn.Sequential(nn.Linear(input_dim, output_dim),
                                              nn.ReLU(), nn.Linear(output_dim, output_dim)))
        if self.model == "GAT":
            return GATConv(input_dim, output_dim, self.head, concat=False)
        if self.model == "Sage":
            return SAGEConv(input_dim, output_dim)
        raise ValueError(f"GNN model not supported: {self.model}")

    def forward(self, data: Data):
        """
        Forward propagation function

        Args:
        - hetero: input heterogeneous graph

        Returns:
        - output: output tensor
        """
        x = self.linear(data.x)
        for i in range(self.layer - 1):
            x = self.convs[i](x, data.edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, data.edge_index)
        return x
