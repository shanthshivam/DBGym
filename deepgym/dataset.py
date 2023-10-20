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
    - dataset: Tabular, DB2PyG or others
    '''

    data_dir = "Datasets/"
    path = os.path.join(data_dir, cfg.dataset.name)
    if cfg.dataset.type == 'single':
        tb = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        tb.load_csv()
        cfg.model.output_dim = tb.output
        return tb
    if cfg.dataset.type == 'join':
        tb = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        tb.load_join()
        cfg.model.output_dim = tb.output
        return tb
    if cfg.dataset.type == 'graph':
        db = DataBase(path)
        db.load()
        db.prepare_encoder()
        data = DB2PyG(db, cfg.dataset.file, cfg.dataset.column)
        cfg.model.output_dim = data.output
        return data
    raise ValueError(f"Model not supported: {cfg.model}")
