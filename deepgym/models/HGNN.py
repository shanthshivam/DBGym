import torch
from torch_geometric.nn import HGTConv, Linear, HeteroConv, SAGEConv

class HGT(torch.nn.Module):

    def __init__(self, cfg, data):
        super().__init__()

        self.hidden_dim = cfg.model.hidden_dim
        self.output_dim = cfg.model.output_dim
        self.layer = cfg.model.layer
        self.head = cfg.model.head

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(data[node_type].x.shape[1] , self.hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(self.layer):
            conv = HGTConv(self.hidden_dim, self.hidden_dim, data.metadata(),
                           self.head, group='sum')
            self.convs.append(conv)

        self.lin = Linear(self.hidden_dim, self.output_dim)

    def forward(self, data, cfg):
        y = {}
        for node_type in data.node_types:
            x = data[node_type].x
            y[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            y = conv(y, data.edge_index_dict)

        return self.lin(y[cfg.dataset.file])


class HGCN(torch.nn.Module):
    def __init__(self, args, data, embed):
        super().__init__()


        self.hidden_dim = args.hidden
        self.output_dim = args.output
        self.layer = args.layer
        # self.head = args.head
        self.file = args.file
        self.convs = torch.nn.ModuleList()

#  {edgetype: GCNConv(-1, self.hidden_dim)   for edgetype in  data.edge_types }
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(embed[node_type].shape[1] , self.hidden_dim)
        for _ in range(self.layer):
            # conv = HeteroConv(
            #     {edgetype: GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops= False)   for edgetype in  data.edge_types }
            #     , aggr='sum')
            conv = HeteroConv(
                {edgetype: SAGEConv(self.hidden_dim, self.hidden_dim, aggr= "mean" )   for edgetype in  data.edge_types }
                , aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(self.hidden_dim, self.output_dim)

        with torch.no_grad():  # Initialize lazy modules.
            _ = self.forward(data.x_dict, data.edge_index_dict)

    def forward(self, x_dict, edge_index_dict):
        y = {}
        for node_type, x in x_dict.items():
            y[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            y = conv(y, edge_index_dict)
        
        return self.lin(y[self.file + '_id'])

def HGNN(cfg, loader, *args, **kwargs) -> torch.nn.Module:
    '''
    Create the model according to configuration.
    Input: None
    Output: args, cfg
    '''
    loader.Embedding_hetero() # Maybe this operation can be done in previous step.
    if cfg.model.subtype == "HGCN":
        return HGCN(cfg, loader.hetero, loader.embedding_hetero)
    elif cfg.model.subtype == "HGT":
        return HGT(cfg, loader.hetero, loader.embedding_hetero)
    else:
        raise ValueError("Model not supported: {}".format(cfg.model.subtype))