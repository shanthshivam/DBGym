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

    if cfg.model.output_dim > 1:
        criterion = torch.nn.CrossEntropyLoss()
        pred = pred.float()
        true = true.long()
        if cfg.model.name == 'XGBoost':
            accuracy = (pred == true).float().mean().item()
            return accuracy
        loss = criterion(pred, true)
        accuracy = (pred.argmax(dim=1) == true).float().mean().item()
        return loss, accuracy
    if cfg.model.output_dim == 1:
        criterion = torch.nn.MSELoss()
        pred = pred.float()
        true = true.float()
        loss = criterion(pred, true)
        if cfg.model.name == 'XGBoost':
            return loss.detach().item()
        return loss, loss.detach().item()
