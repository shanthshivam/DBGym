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
from dbgym.register import module_dict


def create_model(cfg: CfgNode, dataset):
    '''
    Create the model according to configuration

    Args:
    - cfg (CfgNode): The configuration
    - dataset: The dataset for model initialization

    Output:
    - model: The model
    '''

    graph_models = module_dict['graph_model']
    if cfg.model.name in graph_models:
        return graph_models[cfg.model.name](cfg, dataset.graph).to(
            torch.device(cfg.device))
    tabular_models = module_dict['tabular_model']
    if cfg.model.name in tabular_models:
        return tabular_models[cfg.model.name](cfg, dataset).to(
            torch.device(cfg.device))

    if cfg.model.name in ["GCN", "GIN", "GAT", "Sage"]:
        return GNN(cfg, dataset.graph).to(torch.device(cfg.device))
    if cfg.model.name in ["HGT", "HGCN"]:
        return HeteroGNN(cfg, dataset.graph).to(torch.device(cfg.device))
    if cfg.model.name == "MLP":
        return MLP(cfg, dataset).to(torch.device(cfg.device))
    if cfg.model.name == "XGBoost":
        return xgb(cfg, dataset)
    raise ValueError(f"Model not supported: {cfg.model.name}")
