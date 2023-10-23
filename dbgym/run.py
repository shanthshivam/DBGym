"""
run.py
The entry of DBGym reposity.
"""

import time
from yacs.config import CfgNode
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything
from dbgym.logger import Logger
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost


def run(cfg: CfgNode):
    """
    The entry of DBGym reposity.

    Args:
    - cfg (CfgNode): The configuration
    """

    t = time.time()
    seed_everything(cfg.seed)
    auto_select_device(cfg)
    logger = Logger(cfg)
    start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"Start time: {start}")
    dataset = create_dataset(cfg)
    model = create_model(cfg, dataset)
    logger.log(cfg.dump())
    if cfg.model.type == 'XGBoost':
        train_xgboost(dataset, model, logger, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        train(dataset, model, optimizer, scheduler, logger, cfg)
    end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"End time: {end}")
    logger.log(f"Use time: {time.time() - t:.4f} s")
    logger.close()
