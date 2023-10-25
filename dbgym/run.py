"""
run.py
The entry of DBGym reposity.
"""

import time

from yacs.config import CfgNode

from dbgym.dataset import create_dataset
from dbgym.logger import Logger
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.register import module_dict
from dbgym.train import train, train_xgboost
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything


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
    print(f'Loading datasets: {cfg.dataset.name}')
    dt = time.time()
    dataset = create_dataset(cfg)
    dt = time.time() - dt
    print('-------------------- IMPORTANT --------------------')
    print(f'Logs and predictions are saved to {logger.path}')
    print('-------------------- IMPORTANT --------------------')
    model = create_model(cfg, dataset)
    logger.log(cfg.dump())
    tt = time.time()
    if cfg.train.name in module_dict['train']:
        stats = module_dict['train'][cfg.train.name](dataset, model, logger, cfg)
    elif cfg.model.name == 'XGBoost':
        stats = train_xgboost(dataset, model, logger, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        stats = train(dataset, model, optimizer, scheduler, logger, cfg)
    end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"End time: {end}\n")
    logger.log(f"Dataset Use time: {dt:.4f} s")
    logger.log(f"Training Use time: {time.time() - tt:.4f} s")
    logger.log(f"Use time: {time.time() - t:.4f} s")
    logger.close()
    dataset.fill_na(stats['pred'], logger.path)
    return stats
