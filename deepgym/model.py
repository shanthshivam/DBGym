'''
model.py
Create the model according to configuration.
'''

from yacs.config import CfgNode
from deepgym.models.gnn import GNN
from deepgym.models.heterognn import HeteroGNN
from deepgym.models.mlp import MLP
from deepgym.db2pyg import DB2PyG

model_dict = {
    "GNN": GNN,
    "HGNN": HeteroGNN, 
    "MLP": MLP,
}

def create_model(cfg: CfgNode, loader : DB2PyG|None = None):
    '''
    Create the model according to configuration.
    Input: None
    Output: args, cfg
    '''
    # Lugar is not very familiar with XGBoost, so this part is not implemented.
    if cfg.model.type in model_dict:
        return model_dict[cfg.model.type](cfg, loader)
    else:
        raise ValueError("Model not supported: {}".format(cfg.model))
