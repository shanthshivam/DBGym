"""
gnn.py
Graph neural network module.
"""

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, GINConv, Linear, SAGEConv
from yacs.config import CfgNode


class GNNEncoder(nn.Module):
    """
    Graph neural network encoder
    """
    def __init__(self, graph, dimension):
        super().__init__()

        self.emb_dict = nn.ModuleDict()
        self.lin_dict = nn.ModuleDict()
        for node_type in graph.node_types:
            max_values = torch.max(graph[node_type].x_d, dim=0).values
            embedding_list = nn.ModuleList([
                nn.Embedding(max_values[i] + 1, dimension)
                for i in range(graph[node_type].x_d.size(1))
            ])
            self.emb_dict[node_type] = embedding_list
            self.lin_dict[node_type] = Linear(graph[node_type].x_c.shape[1],
                                              dimension)

    def forward(self, graph):
        """
        Forward propagation function

        Args:
        - graph: input heterogeneous graph

        Returns:
        - x: output node embedding
        """
        x = []

        for node_type in graph.node_types:
            embedding_list = self.emb_dict[node_type]
            x_d = graph[node_type].x_d
            y_d = [embedding_list[j](x_d[:, j]) for j in range(x_d.size(1))]
            y_d = torch.sum(torch.stack(y_d, dim=2), dim=2)
            x_c = graph[node_type].x_c
            y_c = self.lin_dict[node_type](x_c)
            x.append((y_d + y_c).relu_())

        return torch.cat(x, dim=-2)


class GNN(nn.Module):
    """
    Graph neural network module
    """
    def __init__(self, cfg: CfgNode, graph: HeteroData):
        super().__init__()
        output_dim = cfg.model.output_dim
        hidden_dim = cfg.model.hidden_dim
        self.layer = cfg.model.layer
        self.head = cfg.model.head
        self.model = cfg.model.name

        self.encoder = GNNEncoder(graph, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(self.layer - 1):
            self.convs.append(self.create_conv(hidden_dim, hidden_dim))
        self.convs.append(self.create_conv(hidden_dim, output_dim))

    def create_conv(self, input_dim: int, output_dim: int) -> nn.Module:
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
            return GINConv(
                nn=nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(),
                                 nn.Linear(output_dim, output_dim)))
        if self.model == "GAT":
            return GATConv(input_dim, output_dim, self.head, concat=False)
        if self.model == "Sage":
            return SAGEConv(input_dim, output_dim)
        raise ValueError(f"GNN model not supported: {self.model}")

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward propagation function

        Args:
        - graph: input heterogeneous graph

        Returns:
        - x: output prediction results
        """
        x = self.encoder(graph)
        edge_index = graph.to_homogeneous().edge_index
        for i in range(self.layer - 1):
            x = self.convs[i](x, edge_index)
            x = nn.functional.relu(x)
        x = self.convs[-1](x, edge_index)
        return x
