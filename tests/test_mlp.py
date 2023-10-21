import unittest
from unittest import TestCase

import time
from torch import optim
from yacs.config import CfgNode

import sys 
sys.path.append("") 

from dbgym.loss import compute_loss
from dbgym.config import get_config
from dbgym.utils import seed_everything
from dbgym.logger import Logger
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.models.mlp import MLP
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.db import Tabular

def perform(dataset, model, optimizer, scheduler, logger: Logger, cfg: CfgNode):
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

    return result

class TestDataset(unittest.TestCase):
    
    def SetUp(self):
        # For more extensive testing, possibly requiring initialization
        pass

    def test_mlp(self):
        start = time.time()

        self.args, self.cfg = get_config()
        seed_everything(self.cfg.seed)
        self.logger = Logger(self.cfg.log_dir)

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
        
        result = perform(self.dataset, self.model, self.optimizer, self.scheduler, self.logger, self.cfg)
        self.assertTrue(result['train'] > 0.65)
        self.assertTrue(result['valid'] > 0.65)
        self.assertTrue(result['test'] > 0.65)

        self.logger.close()

        end = time.time()
        print(f"Load time: {time.time() - start} s")
