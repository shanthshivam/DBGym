import unittest
from unittest import TestCase

import time
import torch
from torch import optim
from yacs.config import CfgNode

import sys 
sys.path.append("") 

from dbgym.config import get_config
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything
from dbgym.logger import Logger
from dbgym.loss import compute_loss
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost
from dbgym.models.mlp import MLP
from dbgym.models.gnn import GNN
from dbgym.models.xgb import xgb
from dbgym.db import Tabular
from dbgym.db2pyg import DB2PyG
from dbgym.train import train, train_xgboost


class TestDataset(unittest.TestCase):
    
    def SetUp(self):
        # For more extensive testing, possibly requiring initialization
        pass

    def test_mlp(self):
        st = time.time()
        self.args, self.cfg = get_config(config_path="tests/test_config_mlp.yaml")
        seed_everything(self.cfg.seed)
        auto_select_device(self.cfg)
    
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

        result = train(self.dataset, self.model, self.optimizer, self.scheduler, self.logger, self.cfg)
        self.assertTrue(result[0] > 0.65)
        self.assertTrue(result[1] > 0.65)
        self.assertTrue(result[2] > 0.65)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"End time: {end}")
        self.logger.log(f"Use time: {time.time() - st:.4f} s")
        self.logger.close()

    def test_gnn(self):
        st = time.time()
        self.args, self.cfg = get_config(config_path="tests/test_config_gnn.yaml")
        seed_everything(self.cfg.seed)
        auto_select_device(self.cfg)
    
        self.logger = Logger(self.cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"Start time: {start}")
        
        #1. Test the dataset
        self.dataset = create_dataset(self.cfg)
        self.assertIsInstance(self.dataset,DB2PyG)

        #2. Test the model, optimizer and scheduler
        self.model = create_model(self.cfg, self.dataset)

        self.optimizer = create_optimizer(self.cfg, self.model.parameters())
        self.scheduler = create_scheduler(self.cfg, self.optimizer)
        self.assertIsInstance(self.model, GNN)
        self.assertIsInstance(self.optimizer,optim.Adam)
        self.assertIsInstance(self.scheduler,optim.lr_scheduler.CosineAnnealingLR)

        #3.Test the training, validation, and testing process.
        #  If the model's results on the training set, validation set, and test set are all greater than 0.65 
        #  (random guessing is 0.5), it indicates that the model has learned knowledge.

        result = train(self.dataset, self.model, self.optimizer, self.scheduler, self.logger, self.cfg)
        self.assertTrue(result[0] > 0.65)
        self.assertTrue(result[1] > 0.65)
        self.assertTrue(result[2] > 0.65)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"End time: {end}")
        self.logger.log(f"Use time: {time.time() - st:.4f} s")
        self.logger.close()

    def test_xgboost(self):
        st = time.time()
        self.args, self.cfg = get_config(config_path="tests/test_config_xgboost.yaml")
        seed_everything(self.cfg.seed)
        auto_select_device(self.cfg)
    
        self.logger = Logger(self.cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"Start time: {start}")
        
        #1. Test the dataset
        self.dataset = create_dataset(self.cfg)
        self.assertIsInstance(self.dataset,Tabular)

        #2. Test the model, optimizer and scheduler
        self.model = create_model(self.cfg, self.dataset)

        #3.Test the training, validation, and testing process.
        #  If the model's results on the training set, validation set, and test set are all greater than 0.65 
        #  (random guessing is 0.5), it indicates that the model has learned knowledge.

        train_xgboost(self.dataset, self.model, self.logger, self.cfg)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.logger.log(f"End time: {end}")
        self.logger.log(f"Use time: {time.time() - st:.4f} s")
        self.logger.close()
