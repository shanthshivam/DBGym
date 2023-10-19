'''
train.py
The training procedure.
'''

import time
from yacs.config import CfgNode
from deepgym.logger import Logger
from deepgym.loss import compute_loss


def train(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode):
    '''
    The training function
    '''

    start = time.time()
    if cfg.model.type in ('GNN', 'HGNN'):
        data = dataset.hetero
    elif cfg.model.type == 'MLP':
        data = dataset
    else:
        raise NotImplementedError
    y = data.y
    mask = dataset.mask

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
        result['train'] = score
        _, score = compute_loss(cfg, output[mask['valid']], target[mask['valid']])
        print(score)
        result['valid'] = score
        _, score = compute_loss(cfg, output[mask['test']], target[mask['test']])
        print(score)
        result['test'] = score
        optimizer.step()
        scheduler.step()

        logger.log_scalars("Accuracy", result, epoch)
        logger.log_scalar("Loss", loss.item(), epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
