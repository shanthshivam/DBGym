"""
dataset.py
This module contains some dataset functions.
"""

import os
import argparse
from typing import Tuple
from deepgym.db import DataBase
from deepgym.db2pyg import DB2PyG
from yacs.config import CfgNode


def from_csv(cfg: CfgNode) -> DataBase:
    """
    The seed_everything function, set seed

    Args:
    - cfg: The configuration
    """

    data_dir = "./deepgym/data"
    db = DataBase(os.path.join(data_dir, cfg.dataset.name))
    db.load()
    db.prepare_encoder()
    return db 


# def from_sql(cfg: CfgNode) -> Tuple[argparse.Namespace, CfgNode]:
#     '''
#     The config function, get args and cfg
#     Input: None
#     Output: args, cfg
#     '''
#     return 0, 0


def create_dataset(cfg: CfgNode):
    '''
    The config function, get args and cfg
    Input: None
    Output: args, cfg
    '''
    db = from_csv(cfg)
    print(db)
    if cfg.model.type == "GNN" or cfg.model.type == "HGNN":
        return DB2PyG(db, target_csv=cfg.dataset.file, target_col=cfg.dataset.column, task=cfg.dataset.task)
    raise ValueError(f"Model not supported: {cfg.model}")
