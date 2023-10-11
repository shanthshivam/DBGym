"""
dataset.py
This module contains some dataset functions.
"""

import argparse
from typing import Tuple
from RDBench.db import DataBase
from yacs.config import CfgNode


def from_csv(cfg: CfgNode) -> DataBase:
    """
    The seed_everything function, set seed

    Args:
    - cfg: The configuration
    """

    return 0, 0


def from_sql(cfg: CfgNode) -> Tuple[argparse.Namespace, CfgNode]:
    '''
    The config function, get args and cfg
    Input: None
    Output: args, cfg
    '''
    return 0, 0


def create_dataset(cfg: CfgNode) -> Tuple[argparse.Namespace, CfgNode]:
    '''
    The config function, get args and cfg
    Input: None
    Output: args, cfg
    '''
    return cfg
