"""
heterognn.py
Heterogeneous graph neural network module.
"""

import torch
from yacs.config import CfgNode
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear, HeteroConv, SAGEConv

class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network module
    """

    def __init__(self, cfg: CfgNode, hetero: HeteroData):
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.model.output_dim
        model = cfg.model.subtype
        layer = cfg.model.layer
        head = cfg.model.head
        self.file = cfg.dataset.file

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hetero.node_types:
            self.lin_dict[node_type] = Linear(hetero[node_type].x.shape[1], hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(layer):
            if model == "HGT":
                conv = HGTConv(hidden_dim, hidden_dim, hetero.metadata(), head)
            elif model == "HGCN":
                conv = HeteroConv({e: SAGEConv(hidden_dim, hidden_dim) for e in hetero.edge_types})
            else:
                raise ValueError(f"HeteroGNN model not supported: {model}")
            self.convs.append(conv)

        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, hetero: HeteroData) -> torch.Tensor:
        """
        Forward propagation function

        Args:
        - hetero: input heterogeneous graph

        Returns:
        - output: output tensor
        """
        y = {}

        for node_type in hetero.node_types:
            x = hetero[node_type].x
            y[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            y = conv(y, hetero.edge_index_dict)

        return self.lin(y[self.file])
