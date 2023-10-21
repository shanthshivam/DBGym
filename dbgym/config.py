"""
config.py
This module contains some configuration functions.
"""

import argparse
from typing import Tuple
from yacs.config import CfgNode


def get_args() -> argparse.Namespace:
    """
    This function gets the arguments use by the experiment.

    Returns:
    - args: The arguments use by the experiment
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration path')
    args = parser.parse_args()

    return args


def set_cfg() -> CfgNode:
    """
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    Returns:
    - cfg: An example configuration use by the experiment
    """

    cfg = CfgNode()
    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #
    cfg.seed = 42
    cfg.device = 'auto'

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CfgNode()
    # Name of the dataset
    cfg.dataset.name = 'financial'
    # Target file
    cfg.dataset.file = 'loan'
    # Target column
    cfg.dataset.column = 'status'
    # Dataset type: single, join or graph
    cfg.dataset.type = 'graph'

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CfgNode()
    # Training (and validation) pipeline mode
    cfg.train.mode = 'standard'
    # Training epochs
    cfg.train.epoch = 200

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CfgNode()
    cfg.model.type = 'gnn'

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CfgNode()
    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'
    # Base learning rate
    cfg.optim.lr = 0.01
    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'

    return cfg


def get_config() -> Tuple[argparse.Namespace, CfgNode]:
    """
    This function gets the configurations use by the experiment.

    Returns:
    - args: The arguments use by the experiment.
    - cfg: The configuration use by the experiment.
    """

    args = get_args()
    cfg = set_cfg()

    with open(args.cfg, "r", encoding="utf-8") as f:
        config = CfgNode.load_cfg(f)

    cfg.update(config)

    return args, cfg
