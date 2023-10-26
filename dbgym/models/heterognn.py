"""
heterognn.py
Heterogeneous graph neural network module.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, HGTConv, Linear, SAGEConv
from yacs.config import CfgNode


class HeteroEncoder(torch.nn.Module):
    """
    Heterogeneous graph neural network encoder
    """
    def __init__(self, graph, dimension):
        super().__init__()

        self.emb_dict = torch.nn.ModuleDict()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            max_values = torch.max(graph[node_type].x_d, dim=0).values
            embedding_list = torch.nn.ModuleList([
                torch.nn.Embedding(max_values[i] + 1, dimension)
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
        - y: output data dict
        """
        y = {}

        for node_type in graph.node_types:
            embedding_list = self.emb_dict[node_type]
            x_d = graph[node_type].x_d
            y_d = [embedding_list[j](x_d[:, j]) for j in range(x_d.size(1))]
            y_d = torch.sum(torch.stack(y_d, dim=2), dim=2)
            x_c = graph[node_type].x_c
            y_c = self.lin_dict[node_type](x_c)
            y[node_type] = (y_d + y_c).relu_()

        return y


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network module
    """
    def __init__(self, cfg: CfgNode, graph: HeteroData):
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.model.output_dim
        model = cfg.model.name
        layer = cfg.model.layer
        head = cfg.model.head
        self.file = cfg.dataset.query.split('.')[0]

        self.encoder = HeteroEncoder(graph, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(layer):
            if model == "HGT":
                conv = HGTConv(hidden_dim, hidden_dim, graph.metadata(), head)
            elif model == "HGCN":
                conv = HeteroConv({
                    e: SAGEConv(hidden_dim, hidden_dim)
                    for e in graph.edge_types
                })
            else:
                raise ValueError(f"HeteroGNN model not supported: {model}")
            self.convs.append(conv)

        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward propagation function

        Args:
        - graph: input heterogeneous graph

        Returns:
        - output: output tensor
        """
        y = self.encoder(graph)
        for conv in self.convs:
            y = conv(y, graph.edge_index_dict)
        return self.lin(y[self.file])


class HGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network module
    """
    def __init__(self, cfg: CfgNode, graph: HeteroData):
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.model.output_dim
        model = cfg.model.name
        layer = cfg.model.layer
        head = cfg.model.head
        self.file = cfg.dataset.query.split('.')[0]

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            input_dim = graph[node_type].x_d.shape[1] + graph[
                node_type].x_c.shape[1]
            self.lin_dict[node_type] = Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(layer):
            if model == "HGT":
                conv = HGTConv(hidden_dim, hidden_dim, graph.metadata(), head)
            elif model == "HGCN":
                conv = HeteroConv({
                    e: SAGEConv(hidden_dim, hidden_dim)
                    for e in graph.edge_types
                })
            else:
                raise ValueError(f"Model not supported: {model}")
            self.convs.append(conv)

        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward propagation function

        Args:
        - graph: input heterogeneous graph

        Returns:
        - output: output tensor
        """
        y = {}

        for node_type in graph.node_types:
            x = torch.cat((graph[node_type].x_d, graph[node_type].x_c), dim=1)
            y[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            y = conv(y, graph.edge_index_dict)

        return self.lin(y[self.file])
