'''
Create the model according to configuration.
'''
from yacs.config import CfgNode
from deepgym.models.GNN import GNN
from deepgym.models.HGNN import HGCN, HGT
from deepgym.models.MLP import MLP
from .db2pyg import DB2PyG

def create_model(cfg: CfgNode, loader : DB2PyG|None = None):
    '''
    Create the model according to configuration.
    Input: None
    Output: args, cfg
    '''
    if cfg.model.type == "GNN":
        return GNN(cfg)
    elif cfg.model.type == "HGNN":
        if loader is None:
            raise ValueError("loader is None, which is not allowed when using HGNN.")
        loader.Embedding_hetero() # Maybe this operation can be done in previous step.
        return HGCN(cfg, loader.hetero, loader.embedding_hetero) if cfg.model.subtype == "HGCN" else HGT(cfg, loader.hetero, loader.embedding_hetero)
    elif cfg.model.type == "MLP":
        return MLP(cfg)
    elif cfg.model.type == "XGBoost":
        # Lugar is not very familiar with XGBoost, so this part is not implemented.
        pass
    else:
        raise ValueError("Model not supported: {}".format(cfg.model))
