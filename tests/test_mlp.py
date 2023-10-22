import unittest
from unittest import TestCase

import time
from torch import optim
from yacs.config import CfgNode

import sys 
sys.path.append("") 

from dbgym.config import get_config
from dbgym.utils.seed import seed_everything
from dbgym.logger import Logger
from dbgym.loss import compute_loss
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost
from dbgym.models.mlp import MLP
from dbgym.db import Tabular

def perform(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode):
    '''
    The training function
    '''

    start = time.time()
    if cfg.model.type in ('GNN', 'HGNN'):
        data = dataset.hetero.to(torch.device(cfg.device))
    elif cfg.model.type == 'MLP':
        data = dataset.to(torch.device(cfg.device))
    else:
        raise NotImplementedError
    y = data.y
    mask = dataset.mask

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

class TestDataset(unittest.TestCase):
    
    def SetUp(self):
        # For more extensive testing, possibly requiring initialization
        pass

    def test_mlp(self):
        st = time.time()
        self.args, self.cfg = get_config()
        seed_everything(self.cfg.seed)
        self.logger = Logger(self.cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"Start time: {start}")
        
        #1. Test the dataset
        self.dataset = create_dataset(self.cfg)
        self.assertIsInstance(self.dataset,Tabular)

        #2. Test the transformation to the Tabular class
        # Not testing the connected tables yet.
        self.Correct_dict = {'file': 'loan', 'col': 'Status'}
        self.assertEqual(self.dataset.file, self.Correct_dict['file'])
        self.assertEqual(self.dataset.col, self.Correct_dict['col'])

        #3. Test the model, optimizer and scheduler
        self.model = create_model(self.cfg, self.dataset)

        self.optimizer = create_optimizer(self.cfg, self.model.parameters())
        self.scheduler = create_scheduler(self.cfg, self.optimizer)
        self.assertIsInstance(self.model, MLP)
        self.assertIsInstance(self.optimizer,optim.Adam)
        self.assertIsInstance(self.scheduler,optim.lr_scheduler.CosineAnnealingLR)

        #4.Test the training, validation, and testing process.
        #  If the model's results on the training set, validation set, and test set are all greater than 0.65 
        #  (random guessing is 0.5), it indicates that the model has learned knowledge.

        train(dataset, model, optimizer, scheduler, logger, cfg)
        result = perform(self.dataset, self.model, self.optimizer, self.scheduler, self.logger, self.cfg)
        self.assertTrue(result[0] > 0.65)
        self.assertTrue(result[1] > 0.65)
        self.assertTrue(result[2] > 0.65)


        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"End time: {end}")
        self.logger.log(f"Use time: {time.time() - st:.4f} s")
        self.logger.close()
