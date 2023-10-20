"""
xgboost.py
XGBoost module.
"""

import xgboost
from deepgym.db import Tabular
from yacs.config import CfgNode


def xgb(cfg: CfgNode, data: Tabular):
    """
    XGBoost function
    """

    if cfg.dataset.task == 'classification':
        return xgboost.XGBClassifier(objective='multi:softmax', num_class=len(set(data.y)))
    if cfg.dataset.task == 'regression':
        return xgboost.XGBRegressor(objective='reg:squarederror')
    return 0
