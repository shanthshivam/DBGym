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

    run_time = time.time()
    seed_everything(cfg.seed)
    auto_select_device(cfg)
    logger = Logger(cfg)
    start_time = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"Start time: {start_time}")
    print(f'Loading datasets: {cfg.dataset.name}')
    dataset_time = time.time()
    dataset = create_dataset(cfg)
    dataset_time = time.time() - dataset_time
    print(f'Logs and predictions are saved to {logger.path}')
    model = create_model(cfg, dataset)
    logger.log(cfg.dump())
    train_time = time.time()
    print(f'Training ... Using device: {cfg.device}')
    if cfg.train.name in module_dict['train']:
        stats = module_dict['train'][cfg.train.name](dataset, model, logger,
                                                     cfg)
    elif cfg.model.name == 'XGBoost':
        stats = train_xgboost(dataset, model, logger, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        stats = train(dataset, model, optimizer, scheduler, logger, cfg)
    end_time = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"End time: {end_time}\n")
    logger.log(f"Dataset Use time: {dataset_time:.4f} s")
    logger.log(f"Training Use time: {time.time() - train_time:.4f} s")
    logger.log(f"Total Use time: {time.time() - run_time:.4f} s")
    logger.close()
    # dataset.fill_na(stats['pred'], logger.path)
    return stats
