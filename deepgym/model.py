'''
Create the model according to configuration.
'''
from yacs.config import CfgNode
from .models.GNN import GNN
from .models.HGNN import HGCN, HGT
from .models.MLP import MLP

def create_model(cfg: CfgNode):
    '''
    Create the model according to configuration.
    Input: None
    Output: args, cfg
    '''
    if cfg.model.type == "GNN":
        return GNN(cfg)
    elif cfg.model.type == "HGNN":
        return HGCN(cfg) if cfg.model.subtype == "HGCN" else HGT(cfg)
    elif cfg.model.type == "MLP":
        return MLP(cfg)
    elif cfg.model.type == "XGBoost":
        # Lugar is not very familiar with XGBoost, so this part is not implemented.
        pass
    else:
        raise ValueError("Model not supported: {}".format(cfg.model))
