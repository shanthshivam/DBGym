'''
model.py
Create the model according to configuration.
'''

from yacs.config import CfgNode
from deepgym.models.gnn import GNN
from deepgym.models.heterognn import HeteroGNN
from deepgym.models.mlp import MLP
from deepgym.db2pyg import DB2PyG


def create_model(cfg: CfgNode, dataset: DB2PyG = None):
    '''
    Create the model according to configuration

    Args:
    - cfg: The seed to set
    - cfg: The seed to set
    '''

    if cfg.model.type == "GNN":
        return GNN(cfg, dataset.hetero)
    if cfg.model.type == "HGNN":
        return HeteroGNN(cfg, dataset.hetero)
    if cfg.model.type == "MLP":
        return MLP(cfg)
    if cfg.model.type == "XGBoost":
        # Lugar is not very familiar with XGBoost, so this part is not implemented.
        pass
    else:
        raise ValueError(f"Model not supported: {cfg.model}")
