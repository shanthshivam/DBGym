"""
test.py
Unit test module.
"""

import sys
import time
import unittest

from torch import optim

from dbgym.config import get_config, set_from_path
from dbgym.dataset import create_dataset
from dbgym.db import Tabular
from dbgym.db2graph import DB2Graph
from dbgym.logger import Logger
from dbgym.model import create_model
from dbgym.models.gnn import GNN
from dbgym.models.mlp import MLP
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.run import run
from dbgym.train import train, train_xgboost
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything

sys.path.append("")


class TestDataset(unittest.TestCase):
    """
    Unit test module.
    """
    def test_gnn(self):
        """
        GNN test function.
        """

        t = time.time()
        cfg = set_from_path("tests/test_config_gnn.yaml")
        seed_everything(cfg.seed)
        auto_select_device(cfg)

        logger = Logger(cfg)
        start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"Start time: {start}")

        # Test the dataset
        dataset = create_dataset(cfg)
        self.assertIsInstance(dataset, DB2Graph)

        # Test the model, optimizer and scheduler
        model = create_model(cfg, dataset)
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        self.assertIsInstance(model, GNN)
        self.assertIsInstance(optimizer, optim.Adam)
        self.assertIsInstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

        # 3 Test the training, validation, and test process.
        # If the model's results on the train, valid, and test set are all greater than 0.5
        # (random guessing is 0.25), the performance of model is normal.
        stats = train(dataset, model, optimizer, scheduler, logger, cfg)
        self.assertTrue(stats['train'] > 0.5)
        self.assertTrue(stats['valid'] > 0.5)
        self.assertTrue(stats['test'] > 0.5)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - t:.4f} s")
        logger.close()

    def test_mlp(self):
        """
        MLP test function.
        """

        t = time.time()
        cfg = set_from_path("tests/test_config_mlp.yaml")
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
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        self.assertIsInstance(model, MLP)
        self.assertIsInstance(optimizer, optim.Adam)
        self.assertIsInstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

        # 3 Test the training, validation, and test process.
        # If the model's results on the train, valid, and test set are all greater than 0.5
        # (random guessing is 0.25), the performance of model is normal.
        stats = train(dataset, model, optimizer, scheduler, logger, cfg)
        self.assertTrue(stats['train'] > 0.5)
        self.assertTrue(stats['valid'] > 0.5)
        self.assertTrue(stats['test'] > 0.5)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - t:.4f} s")
        logger.close()

    def test_xgboost(self):
        """
        XGBoost test function.
        """

        t = time.time()
        cfg = set_from_path("tests/test_config_xgboost.yaml")
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
        # If the model's results on the train, valid, and test set are all greater than 0.5
        # (random guessing is 0.25), the performance of model is normal.
        stats = train_xgboost(dataset, model, logger, cfg)
        self.assertTrue(stats['train'] > 0.5)
        self.assertTrue(stats['valid'] > 0.5)
        self.assertTrue(stats['test'] > 0.5)

        end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        logger.log(f"End time: {end}")
        logger.log(f"Use time: {time.time() - t:.4f} s")
        logger.close()

    def test_run(self):
        """
        run test function.
        """
        config = set_from_path("tests/test_config_gnn.yaml")
        run(config)
        config = set_from_path("tests/test_config_mlp.yaml")
        run(config)
        config = set_from_path("tests/test_config_xgboost.yaml")
        run(config)
