"""
config.py
This module contains some configuration functions.
"""

import argparse
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


def get_config() -> CfgNode:
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
    # Experiment seed
    cfg.seed = 42
    # Log directory
    cfg.log_dir = 'logs'
    # Experiment device
    cfg.device = 'auto'

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CfgNode()
    # Name of the dataset
    cfg.dataset.dir = "Datasets/"
    # Name of the dataset
    cfg.dataset.name = 'rdb2-bank'
    # Target file
    cfg.dataset.file = 'loan'
    # Target column
    cfg.dataset.column = 'Status'
    # Dataset type: tabular or graph
    cfg.dataset.type = 'graph'
    # Dataset format: single or join for tabular data
    cfg.dataset.format = 'homo'

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CfgNode()
    # Training epochs
    cfg.train.epoch = 200

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CfgNode()
    # Model type: GNN, HGNN, MLP, XGBoost
    cfg.model.type = 'GNN'
    # Model name: GCN, GIN, GAT, Sage for GNN, HGCN, HGT for HGNN
    cfg.model.name = 'GCN'
    # Hidden dimension
    cfg.model.hidden_dim = 128
    # Output dimension
    cfg.model.output_dim = 0
    # Number of layers
    cfg.model.layer = 4
    # Number of heads
    cfg.model.head = 4

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CfgNode()
    # Optimizer: adam, sgd
    cfg.optim.optimizer = 'adam'
    # Base learning rate
    cfg.optim.lr = 0.001
    # Weight decay
    cfg.optim.weight_decay = 0.001
    # Momentum in SGD
    cfg.optim.momentum = 0.9
    # Scheduler: none, step, cos
    cfg.optim.scheduler = 'cos'
    # Milestones in step scheduler
    cfg.optim.milestones = [30, 60, 90]
    # Learning rate decay
    cfg.optim.lr_decay: 0.5

    return cfg


def set_config(config: CfgNode) -> CfgNode:
    """
    This function gets the configurations used by the experiment.

    Args:
    - cfg (CfgNode): The configuration used by the experiment.

    Returns:
    - cfg (CfgNode): The configuration used by the experiment.
    """

    cfg = get_config()
    cfg.update(config)
    return cfg


def set_from_path(path: str) -> CfgNode:
    """
    This function gets the configurations used by the experiment.

    Args:
    - cfg (CfgNode): The configuration used by the experiment.

    Returns:
    - cfg (CfgNode): The configuration used by the experiment.
    """

    cfg = get_config()
    print(cfg)
    with open(path, "r", encoding="utf-8") as f:
        config = CfgNode.load_cfg(f)
    cfg.merge_from_other_cfg(config)
    return cfg
