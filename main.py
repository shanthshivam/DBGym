"""
main.py
The entry of DBGym reposity.
"""

import time
from dbgym.config import get_config
from dbgym.utils.device import auto_select_device
from dbgym.utils.seed import seed_everything
from dbgym.logger import Logger
from dbgym.dataset import create_dataset
from dbgym.model import create_model
from dbgym.optimizer import create_optimizer, create_scheduler
from dbgym.train import train, train_xgboost


if __name__ == '__main__':
    st = time.time()
    args, cfg = get_config()
    seed_everything(cfg.seed)
    auto_select_device(cfg)
    # cfg.device = 'cpu'
    logger = Logger(cfg)
    start = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"Start time: {start}")
    dataset = create_dataset(cfg)
    model = create_model(cfg, dataset)
    if cfg.model.type == 'XGBoost':
        train_xgboost(dataset, model, logger, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        train(dataset, model, optimizer, scheduler, logger, cfg)
    end = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
    logger.log(f"End time: {end}")
    logger.log(f"Use time: {time.time() - st:.4f} s")
    logger.close()
