import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv


class GNN(torch.nn.Module):

    def __init__(self, cfg, *args, **kwargs):
        super(GNN, self).__init__()
        # These should be determined by the datasets
        self.input_dim = cfg.model.input_dim
        self.output_dim = cfg.model.output_dim
        # These should be read from the configuration 
        self.hidden_dim = cfg.model.hidden_dim
        self.layer = cfg.model.layer
        self.head = cfg.model.head
        self.model = cfg.model.subtype
        self.convs = nn.ModuleList()
        self.linear = nn.Linear(cfg.model.input_dim, cfg.model.hidden_dim)

        for _ in range(self.layer - 1):
            self.convs.append(self.create_conv(self.hidden_dim, self.hidden_dim))
        self.convs.append(self.create_conv(self.hidden_dim, self.output_dim))

    def create_conv(self, input_dim, output_dim) -> nn.Module:
        if self.model == "GCN":
            return GCNConv(input_dim, output_dim)
        if self.model == "GIN":
            return GINConv(nn = nn.Sequential(nn.Linear(input_dim, output_dim),
                                              nn.ReLU(), nn.Linear(output_dim, output_dim)))
        if self.model == "GAT":
            return GATConv(input_dim, output_dim, self.head, concat = False)
        if self.model == "Sage":
            return SAGEConv(input_dim, output_dim)
        raise ValueError("The type of GNN is not supported.")

    def forward(self, x, edge_index):
        x = self.linear(x)
        for i in range(self.layer - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x
