'''
model.py
Create the model according to configuration.
'''

from yacs.config import CfgNode
from dbgym.models.gnn import GNN
from dbgym.models.heterognn import HeteroGNN
from dbgym.models.mlp import MLP
from dbgym.models.xgb import xgb


def create_model(cfg: CfgNode, dataset):
    '''
    Create the model according to configuration

    Args:
    - cfg (CfgNode): The configuration
    - dataset: The dataset for model initialization

    Output:
    - model: The model
    '''

    if cfg.model.type == "GNN":
        return GNN(cfg, dataset.hetero)
    if cfg.model.type == "HGNN":
        return HeteroGNN(cfg, dataset.hetero)
    if cfg.model.type == "MLP":
        return MLP(cfg, dataset)
    if cfg.model.type == "XGBoost":
        return xgb(cfg, dataset)
    raise ValueError(f"Model not supported: {cfg.model}")
