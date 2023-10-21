"""
xgboost.py
XGBoost module.
"""

import xgboost
from dbgym.db import Tabular
from yacs.config import CfgNode


def xgb(cfg: CfgNode, data: Tabular):
    """
    XGBoost function
    """

    if cfg.model.output_dim > 1:
        return xgboost.XGBClassifier(objective='multi:softmax', num_class=len(set(data.y)))
    if cfg.model.output_dim == 1:
        return xgboost.XGBRegressor(objective='reg:squarederror')
