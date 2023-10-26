"""
train.py
The training procedure.
"""

import copy
import time
from tqdm import tqdm

import torch
from yacs.config import CfgNode

from dbgym.logger import Logger
from dbgym.loss import compute_loss
from dbgym.register import module_dict


def train(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode,
          **kwargs):
    """
    The training function
    """
    loss_func = module_dict['loss'].get(cfg.loss.name, compute_loss)

    start = time.time()
    data = dataset.to(torch.device(cfg.device))
    y = data.y
    mask = dataset.mask

    logger.log(f"Device: {cfg.device}")
    stats = {}
    if cfg.model.output_dim > 1:
        stats['metric'] = 'Accuracy'
    elif cfg.model.output_dim == 1:
        stats['metric'] = 'Mean Squared Error'
    flag = cfg.model.output_dim > 1

    for epoch in tqdm(range(cfg.train.epoch), unit="epoch"):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        target = y.squeeze()
        result = {}
        losses = {}
        loss, score = loss_func(cfg, output[mask['train']],
                                target[mask['train']])
        loss.backward()
        val_loss, val_score = loss_func(cfg, output[mask['valid']],
                                        target[mask['valid']])
        test_loss, test_score = loss_func(cfg, output[mask['test']],
                                          target[mask['test']])
        optimizer.step()
        scheduler.step()

        logger.log(
            f"Epoch {epoch}/{cfg.train.epoch}: Use time {time.time() - t:.4f} s"
        )

        if epoch == 0 or (val_score - stats['valid']) * (flag * 2 - 1) > 0:
            stats['best_epoch'] = epoch
            stats['train_loss'] = loss.item()
            stats['train'] = score
            stats['valid_loss'] = val_loss.item()
            stats['valid'] = val_score
            stats['test_loss'] = test_loss.item()
            stats['test'] = test_score
            stats['model'] = copy.deepcopy(model)
            stats['pred'] = output[mask['all']].detach()
            if flag:
                stats['pred'] = stats['pred'].argmax(dim=1)

        logger.log(f"Train {stats['metric']}: " +
                   (f"{score:.2%}" if flag else f"{score:.3f}"))
        logger.log(f"Train Loss: {loss.item():.4f}")
        logger.log(f"Valid {stats['metric']}: " +
                   (f"{val_score:.2%}" if flag else f"{val_score:.3f}"))
        logger.log(f"Test {stats['metric']}: " +
                   (f"{test_score:.2%}" if flag else f"{test_score:.3f}"))

        result['Train'] = score
        losses['Train'] = loss.item()
        result['Valid'] = val_score
        losses['Valid'] = val_loss.item()
        result['Test'] = test_score
        losses['Test'] = test_loss.item()
        logger.log_scalars(stats['metric'], result, epoch)
        logger.log_scalars("Loss", losses, epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
        logger.flush()

    logger.log("")
    logger.log(f"Final Train {stats['metric']}: " +
               (f"{stats['train']:.2%}" if flag else f"{stats['train']:.3f}"))
    logger.log(f"Final Valid {stats['metric']}: " +
               (f"{stats['valid']:.2%}" if flag else f"{stats['valid']:.3f}"))
    logger.log(f"Final Test {stats['metric']}: " +
               (f"{stats['test']:.2%}" if flag else f"{stats['test']:.3f}"))
    logger.log("")

    return stats


def train_xgboost(dataset, model, logger: Logger, cfg: CfgNode, **kwargs):
    """
    The training function for xgboost
    """

    loss_func = module_dict['loss'].get(cfg.loss.name, compute_loss)

    x = torch.concat([dataset.x_c, dataset.x_d], dim=1)
    y = dataset.y
    mask = dataset.mask
    model.fit(x[mask['train']], y[mask['train']])
    stats = {}
    for split in ['train', 'valid', 'test']:
        y_pred = torch.tensor(model.predict(x[mask[split]]))
        y_true = y[mask[split]]
        score = loss_func(cfg, y_pred, y_true)
        if cfg.model.output_dim > 1:
            logger.log(f"{split.capitalize()} Accuracy: {score:.2%}")
        elif cfg.model.output_dim == 1:
            logger.log(f"{split.capitalize()} Mean Squared Error: {score:.3f}")
        stats[split] = score
    stats['pred'] = torch.tensor(model.predict(x[mask[split]]))

    return stats
