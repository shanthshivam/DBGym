"""
optimizer.py
This module provides optimizer and scheduler.
"""

from typing import Iterator

from torch import Tensor, optim
from yacs.config import CfgNode

from dbgym.register import module_dict


def create_optimizer(cfg: CfgNode,
                     params: Iterator[Tensor]) -> optim.Optimizer:
    """
    Creates a config-driven optimizer.

    Args:
    - cfg: Configuration use by the experiment
    - params: The parameters to optimize

    Returns:
    - optimizer: Model optimizer
    """

    params = filter(lambda p: p.requires_grad, params)
    optimz = cfg.optim.optimizer
    optimizers = module_dict['optimizer']
    if optimz in optimizers:
        return optimizers[optimz](cfg, params)
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


def create_scheduler(
        cfg: CfgNode,
        optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    """
    Creates a config-driven learning rate scheduler.

    Args:
    - cfg: Configuration use by the experiment
    - optimizer: The optimizer to schedule

    Returns:
    - scheduler: Learning rate scheduler
    """

    sdlr = cfg.optim.scheduler
    schedulers = module_dict['scheduler']
    if sdlr in schedulers:
        return schedulers[sdlr](cfg, optimizer)
    if sdlr == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.train.epoch + 1)
    elif sdlr == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optim.milestones,
            gamma=cfg.optim.lr_decay)
    elif sdlr == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=cfg.train.epoch)
    else:
        raise ValueError(f'Scheduler {sdlr} not supported')
    return scheduler
