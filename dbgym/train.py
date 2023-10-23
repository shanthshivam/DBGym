'''
train.py
The training procedure.
'''

import time
import torch
from yacs.config import CfgNode
from dbgym.logger import Logger
from dbgym.loss import compute_loss


def train(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode):
    '''
    The training function
    '''

    start = time.time()
    data = dataset.to(torch.device(cfg.device))
    y = data.y
    mask = dataset.mask

    logger.log(f"Device: {cfg.device}")
    results = [0, -1e8, 0]
    for epoch in range(cfg.train.epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        target = y.squeeze()
        result = {}
        losses = {}
        loss, score = compute_loss(cfg, output[mask['train']], target[mask['train']])
        loss.backward()
        vl, vs = compute_loss(cfg, output[mask['valid']], target[mask['valid']])
        tl, ts = compute_loss(cfg, output[mask['test']], target[mask['test']])
        optimizer.step()
        scheduler.step()

        logger.log(f"Epoch {epoch} / {cfg.train.epoch}: Use time {time.time() - t:.4f} s")
        if cfg.model.output_dim > 1:
            if vs > results[1]:
                results = [score, vs, ts]
            logger.log(f"Train Accuracy: {score:.2%}")
            logger.log(f"Train Loss: {loss.item():.4f}")
            logger.log(f"Valid Accuracy: {vs:.2%}")
            logger.log(f"Test Accuracy: {ts:.2%}")
            logger.log_scalars("Accuracy", result, epoch)
        elif cfg.model.output_dim == 1:
            if -vs > results[1]:
                results = [-score, -vs, -ts]
            logger.log(f"Train Mean Squared Error: {score:.3f}")
            logger.log(f"Train Loss: {loss.item():.4f}")
            logger.log(f"Valid Mean Squared Error: {vs:.3f}")
            logger.log(f"Test Mean Squared Error: {ts:.3f}")
            logger.log_scalars("Mean Squared Error", result, epoch)

        result['Train'] = score
        losses['Train'] = loss.item()
        result['Valid'] = vs
        losses['Valid'] = vl.item()
        result['Test'] = ts
        losses['Test'] = tl.item()
        logger.log_scalars("Loss", losses, epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
        logger.flush()

    if cfg.model.output_dim > 1:
        logger.log(f"Final Train Accuracy: {results[0]:.2%}")
        logger.log(f"Final Valid Accuracy: {results[1]:.2%}")
        logger.log(f"Final Test Accuracy: {results[2]:.2%}")
    elif cfg.model.output_dim == 1:
        logger.log(f"Final Train Mean Squared Error: {results[0]:.3f}")
        logger.log(f"Final Valid Mean Squared Error: {results[1]:.3f}")
        logger.log(f"Final Test Mean Squared Error: {results[2]:.3f}")

    return results


def train_xgboost(dataset, model, logger: Logger, cfg: CfgNode):
    '''
    The training function for xgboost
    '''

    x = torch.concat([dataset.x_c, dataset.x_d], dim=1)
    y = dataset.y
    mask = dataset.mask
    model.fit(x[mask['train']], y[mask['train']])
    results = []
    for split in ['train', 'valid', 'test']:
        y_pred = torch.tensor(model.predict(x[mask[split]]))
        y_true = y[mask[split]]
        score = compute_loss(cfg, y_pred, y_true)
        if cfg.model.output_dim > 1:
            logger.log(f"{split.capitalize()} Accuracy: {score:.2%}")
        elif cfg.model.output_dim == 1:
            logger.log(f"{split.capitalize()} Mean Squared Error: {score:.3f}")
        results.append(score)

    return results
