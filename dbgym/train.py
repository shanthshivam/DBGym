'''
train.py
The training procedure.
'''

import time
import torch
import copy
from yacs.config import CfgNode
from dbgym.logger import Logger
from dbgym.loss import compute_loss
from dbgym.register import module_dict


def train(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode, **kwargs):
    '''
    The training function
    '''
    loss_func = module_dict['loss'].get(cfg.loss.name, compute_loss)
    
    start = time.time()
    data = dataset.to(torch.device(cfg.device))
    y = data.y
    mask = dataset.mask

    logger.log(f"Device: {cfg.device}")
    # statistics
    s = {}
    if cfg.model.output_dim > 1:
        s['metric'] = 'Accuracy'
    elif cfg.model.output_dim == 1:
        s['metric'] = 'Mean Squared Error'
    # flag
    f = cfg.model.output_dim > 1

    for epoch in range(cfg.train.epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        target = y.squeeze()
        result = {}
        losses = {}
        loss, score = loss_func(cfg, output[mask['train']], target[mask['train']])
        loss.backward()
        val_loss, val_score = loss_func(cfg, output[mask['valid']], target[mask['valid']])
        test_loss, test_score = loss_func(cfg, output[mask['test']], target[mask['test']])
        optimizer.step()
        scheduler.step()

        logger.log(f"Epoch {epoch}/{cfg.train.epoch}: Use time {time.time() - t:.4f} s")

        if epoch == 0 or (val_score - s['valid']) * (f * 2 - 1) > 0:
            s['best_epoch'] = epoch
            s['train_loss'] = loss.item()
            s['train'] = score
            s['valid_loss'] = val_loss.item()
            s['valid'] = val_score
            s['test_loss'] = test_loss.item()
            s['test'] = test_score
            s['model'] = copy.deepcopy(model)
            s['pred'] = output[mask['all']].detach()
            if f:
                s['pred'] = s['pred'].argmax(dim=1)

        logger.log(f"Train {s['metric']}: " + (f"{score:.2%}" if f else f"{score:.3f}"))
        logger.log(f"Train Loss: {loss.item():.4f}")
        logger.log(f"Valid {s['metric']}: " + (f"{val_score:.2%}" if f else f"{val_score:.3f}"))
        logger.log(f"Test {s['metric']}: " + (f"{test_score:.2%}" if f else f"{test_score:.3f}"))

        result['Train'] = score
        losses['Train'] = loss.item()
        result['Valid'] = val_score
        losses['Valid'] = val_loss.item()
        result['Test'] = test_score
        losses['Test'] = test_loss.item()
        logger.log_scalars(s['metric'], result, epoch)
        logger.log_scalars("Loss", losses, epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
        logger.flush()

    logger.log("")
    logger.log(f"Final Train {s['metric']}: " + (f"{s['train']:.2%}" if f else f"{s['train']:.3f}"))
    logger.log(f"Final Valid {s['metric']}: " + (f"{s['valid']:.2%}" if f else f"{s['valid']:.3f}"))
    logger.log(f"Final Test {s['metric']}: " + (f"{s['test']:.2%}" if f else f"{s['test']:.3f}"))
    logger.log("")

    return s


def train_xgboost(dataset, model, logger: Logger, cfg: CfgNode, **kwargs):
    '''
    The training function for xgboost
    '''

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
