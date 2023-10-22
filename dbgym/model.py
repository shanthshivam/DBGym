'''
model.py
Create the model according to configuration.
'''

import torch
from yacs.config import CfgNode
from dbgym.models.gnn import GNN
from dbgym.models.heterognn import HeteroGNN
from dbgym.models.mlp import MLP
from dbgym.models.xgb import xgb
from dbgym.register import dbgym_dict


def create_model(cfg: CfgNode, dataset):
    '''
    Create the model according to configuration

    Args:
    - cfg (CfgNode): The configuration
    - dataset: The dataset for model initialization

    Output:
    - model: The model
    '''

    graph_models = dbgym_dict['graph_model']
    if cfg.model.type in graph_models:
        return graph_models[cfg.model.type](cfg, dataset.hetero).to(torch.device(cfg.device))
    tabular_models = dbgym_dict['tabular_model']
    if cfg.model.type in tabular_models:
        return tabular_models[cfg.model.type](cfg, dataset).to(torch.device(cfg.device))
    if cfg.model.type == "GNN":
        return GNN(cfg, dataset.hetero).to(torch.device(cfg.device))
    if cfg.model.type == "HGNN":
        return HeteroGNN(cfg, dataset.hetero).to(torch.device(cfg.device))
    if cfg.model.type == "MLP":
        return MLP(cfg, dataset).to(torch.device(cfg.device))
    if cfg.model.type == "XGBoost":
        return xgb(cfg, dataset)
    raise ValueError(f"Model not supported: {cfg.model.type}")
