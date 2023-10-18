'''
train.py
The training procedure.
'''

import time
import torch
from yacs.config import CfgNode
from deepgym.logger import Logger
from deepgym.loss import compute_loss


def get_data(dataset, cfg: CfgNode):
    '''
    Prepare data with different format
    '''

    if cfg.model.type == 'GNN':
        data = dataset.homo
        y = data.y
        mask = dataset.mask['homo']
    elif cfg.model.type == 'HGNN':
        data = dataset.hetero
        y = data[cfg.dataset.file].y
        mask = dataset.mask['hetero']
    return data, y, mask


def train(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode):
    '''
    The training function
    '''

    start = time.time()
    data, y, mask = get_data(dataset, cfg)

    for epoch in range(cfg.train.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        train_mask = mask['train']
        target = y.squeeze()
        result = {}
        loss, score = compute_loss(cfg, output[train_mask], target[train_mask])
        loss.backward()
        print(loss)
        print(score)
        _, score = compute_loss(cfg, output[mask['valid']], target[mask['valid']])
        print(score)
        _, score = compute_loss(cfg, output[mask['test']], target[mask['test']])
        print(score)
        optimizer.step()
        scheduler.step()

        logger.log_scalars("Accuracy", result, epoch)
        logger.log_scalar("Loss", loss.item(), epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
