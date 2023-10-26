"""
config.py
This module contains some configuration functions.
"""

import argparse

from yacs.config import CfgNode

from dbgym.register import module_dict


def get_args() -> argparse.Namespace:
    """
    This function gets the arguments use by the experiment.

    Returns:
    - args: The arguments use by the experiment
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default='config.yaml',
                        help='configuration path')
    args = parser.parse_args()

    return args


def get_config() -> CfgNode:
    """
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in dbgym.contrib.config
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
    cfg.log_dir = 'output'
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
    # Query: target_file.target_column
    cfg.dataset.query = 'loan.Status'
    # Dataset type: tabular or graph
    cfg.dataset.type = 'graph'
    # Dataset join parameter
    cfg.dataset.join = 0

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CfgNode()
    # Name of the training function
    cfg.train.name = 'default'
    # Training epochs
    cfg.train.epoch = 200

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CfgNode()
    # Model name: GCN, GIN, GAT, Sage, HGCN, HGT, MLP, XGBoost
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
    cfg.optim.lr_decay = 0.5

    # ----------------------------------------------------------------------- #
    # Loss options
    # ----------------------------------------------------------------------- #
    cfg.loss = CfgNode()
    # Name of the loss function
    cfg.loss.name = 'default'

    # Set user customized cfgs
    for func in module_dict['config'].values():
        func(cfg)

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
    - path (str): The configuration file path.

    Returns:
    - cfg (CfgNode): The configuration used by the experiment.
    """

    cfg = get_config()
    cfg.merge_from_file(path)
    return cfg


if __name__ == '__main__':
    module_dict = {}
    module_dict['config'] = {}
    cfg = get_config()
    with open("./output.yaml", "w") as f:
        f.write(cfg.dump())