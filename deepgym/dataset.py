"""
dataset.py
This module contains dataset function.
"""

import os
from deepgym.db import DataBase, Tabular
from deepgym.db2pyg import DB2PyG
from yacs.config import CfgNode


def create_dataset(cfg: CfgNode):
    '''
    The dataset function, get dataset

    Args:
    - cfg: The configuration

    Return:
    - dataset: DB2PyG or others
    '''

    data_dir = "Datasets/"
    path = os.path.join(data_dir, cfg.dataset.name)
    if cfg.model.type in ('GNN', 'HGNN'):
        db = DataBase(path)
        db.load()
        db.prepare_encoder()
        return DB2PyG(db, cfg.dataset.file, cfg.dataset.column)
    if cfg.model.type == 'MLP':
        tb = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        tb.load_csv()
        return tb
    raise ValueError(f"Model not supported: {cfg.model}")
