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
        target = y[train_mask].squeeze()
        result = {}
        loss, score = compute_loss(cfg, output[train_mask], target)
        loss.backward()
        print(loss)
        print(score)
        optimizer.step()
        scheduler.step()

        # with torch.no_grad():
        #     for split in ['val', 'test']:
        #         mask = hetero_mask[split].to(torch.long)
        #         output = model(graph_hete)
        #         target = hete_class[mask].squeeze()
        #         if cfg.dataset.task == 'classification':
        #             preds = output[mask].argmax(dim=1)
        #             correct = (preds == target.to(torch.long)).sum().item()
        #             total = mask.shape[0]
        #             accuracy = correct / total
        #             print(f'{split} Accuracy: {accuracy:.2%}')
        #             result[split] = accuracy
        #         # elif args.task == 'regression':
        #         #     preds = output[mask].squeeze()
        #         #     RMSE = (preds - target).square().mean().sqrt()
        #         #     StdE = (target - target.mean()).square().mean().sqrt()
        #         #     result.append(-RMSE)
        #         #     log.append(f'{split} RMSE (Root Mean Square Error): {RMSE}, where Std Error: {StdE}')
        #         #     print(log[-1])
        # logger.log_scalars("Accuracy", result, epoch)
        logger.log_scalar("Loss", loss.item(), epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
        # result, log = test(model, dataset, args)
        # if result[0] > best:
        #     best = result[0]
        #     logs[0] = 'Best ' + logs[0]
        #     logg = logs + log
        #     patc = 0
        # else:
        #     patc += 1
        #     if patc > args.patience:
        #         break
    # return 
