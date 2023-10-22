"""
test.py
Unit test module.
"""

import unittest
import sys

import time
from torch import optim

from dbgym.config import get_config
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything
from dbgym.logger import Logger
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost
from dbgym.models.mlp import MLP
from dbgym.models.gnn import GNN
from dbgym.db import Tabular
from dbgym.db2pyg import DB2PyG

sys.path.append("")


class TestDataset(unittest.TestCase):
    """
    Unit test module.
    """

    def test_gnn(self):
        """
        GNN test function.
        """

        cfg = get_config(config_path="tests/test_config_gnn.yaml")[1]
        seed_everything(cfg.seed)
        auto_select_device(cfg)

        logger = Logger(cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"Start time: {start}")

        # Test the dataset
        dataset = create_dataset(cfg)
        self.assertIsInstance(dataset, DB2PyG)

        # Test the model, optimizer and scheduler
        model = create_model(cfg, dataset)
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        self.assertIsInstance(model, GNN)
        self.assertIsInstance(optimizer,optim.Adam)
        self.assertIsInstance(scheduler,optim.lr_scheduler.CosineAnnealingLR)

        # 3 Test the training, validation, and test process.
        # If the model's results on the train, valid, and test set are all greater than 0.65
        # (random guessing is 0.5), the performance of model is normal.
        results = train(dataset, model, optimizer, scheduler, logger, cfg)
        self.assertTrue(results[0] > 0.65)
        self.assertTrue(results[1] > 0.65)
        self.assertTrue(results[2] > 0.65)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - start:.4f} s")
        logger.close()

    def test_mlp(self):
        """
        MLP test function.
        """

        cfg = get_config(config_path="tests/test_config_mlp.yaml")[1]
        seed_everything(cfg.seed)
        auto_select_device(cfg)

        logger = Logger(cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"Start time: {start}")

        # 1 Test the dataset
        dataset = create_dataset(cfg)
        self.assertIsInstance(dataset,Tabular)

        # 2 Test the model, optimizer and scheduler
        model = create_model(cfg, dataset)
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        self.assertIsInstance(model, MLP)
        self.assertIsInstance(optimizer,optim.Adam)
        self.assertIsInstance(scheduler,optim.lr_scheduler.CosineAnnealingLR)

        # 3 Test the training, validation, and test process.
        # If the model's results on the train, valid, and test set are all greater than 0.65
        # (random guessing is 0.5), the performance of model is normal.
        results = train(dataset, model, optimizer, scheduler, logger, cfg)
        self.assertTrue(results[0] > 0.65)
        self.assertTrue(results[1] > 0.65)
        self.assertTrue(results[2] > 0.65)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - start:.4f} s")
        logger.close()

    def test_xgboost(self):
        """
        XGBoost test function.
        """

        cfg = get_config(config_path="tests/test_config_xgboost.yaml")[1]
        seed_everything(cfg.seed)
        auto_select_device(cfg)

        logger = Logger(cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"Start time: {start}")

        # 1 Test the dataset
        dataset = create_dataset(cfg)
        self.assertIsInstance(dataset, Tabular)

        # 2 Test the model, optimizer and scheduler
        model = create_model(cfg, dataset)

        # 3 Test the training, validation, and test process.
        # If the model's results on the train, valid, and test set are all greater than 0.65
        # (random guessing is 0.5), the performance of model is normal.
        results = train_xgboost(dataset, model, logger, cfg)
        self.assertTrue(results[0] > 0.65)
        self.assertTrue(results[1] > 0.65)
        self.assertTrue(results[2] > 0.65)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - start:.4f} s")
        logger.close()
