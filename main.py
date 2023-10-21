"""
main.py
The entry of DBGym reposity.
"""

import time
from dbgym.config import get_config
from dbgym.utils import seed_everything
from dbgym.logger import Logger
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost


if __name__ == '__main__':
    start = time.time()
    args, cfg = get_config()
    seed_everything(cfg.seed)
    logger = Logger(cfg.log_dir)
    t = time.time()
    dataset = create_dataset(cfg)
    print(f"Dataset Use time: {time.time() - t} s")
    model = create_model(cfg, dataset)
    if cfg.model.type == 'XGBoost':
        train_xgboost(dataset, model, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        train(dataset, model, optimizer, scheduler, logger, cfg)
        logger.close()
    print(f"Use time: {time.time() - start} s")
