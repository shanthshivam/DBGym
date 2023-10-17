'''
The process of training models.
'''
import time
import torch
from .logger import Logger
from yacs.config import CfgNode
from train_modules import *

train_module_dict = {
    "GNN" : train_GNN
    "HGNN" : train_HGNN
}
'''
Every train_module should return 
'''


def train(loader, model, optimizer, scheduler, logger: Logger, cfg: CfgNode) -> None:
    print("Train process begins.")
    if cfg.model.type == 'GNN':
        train_GNN(loader, model, optimizer, scheduler, logger, cfg)
    elif cfg.model.type == 'HGNN':
        train_HGNN(loader, model, optimizer, scheduler, logger, cfg)
    else:
        raise ValueError("Invalid model type. Must be 'GCN'.")
    
    logger.close()
    return
