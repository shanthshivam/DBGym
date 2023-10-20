"""
loss.py
Module to calculate loss.
"""

import torch
from yacs.config import CfgNode


def compute_loss(cfg: CfgNode, pred: torch.Tensor, true: torch.Tensor):
    """
    Compute loss and prediction score

    Args:
    - pred (torch.tensor): Unnormalized prediction
    - true (torch.tensor): Ground truth

    Returns:
    - loss: loss
    - prediction score: accuracy or mean squared error
    """

    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if cfg.dataset.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
        if cfg.model.type == 'XGBoost':
            accuracy = (pred == true).float().mean().item()
            return accuracy
        loss = criterion(pred, true)
        accuracy = (pred.argmax(dim=1) == true).float().mean().item()
        return loss, accuracy
    if cfg.dataset.task == 'regression':
        criterion = torch.nn.MSELoss()
        loss = criterion(pred, true)
        if cfg.model.type == 'XGBoost':
            return loss.detach()
        return loss, loss.detach()
    raise ValueError("task must be either classification or regression.")
