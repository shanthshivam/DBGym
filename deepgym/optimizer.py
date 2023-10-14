"""
optimizer.py
This module provides optimizer and scheduler.
"""

from typing import Iterator
from torch import optim, Tensor
from yacs.config import CfgNode

def create_optimizer(cfg: CfgNode, params: Iterator[Tensor]) -> optim.Optimizer:
    """
    Creates a config-driven optimizer.

    Args:
    - cfg: Configuration use by the experiment
    - params: The parameters to optimize

    Returns:
    - optimizer: An optimizer
    """

    params = filter(lambda p: p.requires_grad, params)
    optimz = cfg.optim.optimizer
    if optimz == 'adam':
        optimizer = optim.Adam(params,
                               lr=cfg.optim.lr,
                               weight_decay=cfg.optim.weight_decay)
    elif optimz == 'sgd':
        optimizer = optim.SGD(params,
                              lr=cfg.optim.lr,
                              momentum=cfg.optim.momentum,
                              weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError(f'Optimizer {optimz} not supported')

    return optimizer

def create_scheduler(cfg: CfgNode, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    """
    Creates a config-driven learning rate scheduler.

    Args:
    - cfg: Configuration use by the experiment
    - optimizer: The optimizer to schedule

    Returns:
    - scheduler: A scheduler
    """

    sdlr = cfg.optim.scheduler
    if sdlr == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.optim.max_epoch +
                                              1)
    elif sdlr == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.optim.steps,
                                                   gamma=cfg.optim.lr_decay)
    elif sdlr == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.optim.max_epoch)
    else:
        raise ValueError(f'Scheduler {sdlr} not supported')
    return scheduler
