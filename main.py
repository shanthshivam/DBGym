"""
main.py
The entry of DeepGym reposity.
"""

import time
from deepgym.config import get_config
from deepgym.utils import seed_everything
from deepgym.logger import Logger
from deepgym.dataset import create_dataset
from deepgym.model import create_model
from deepgym.optimizer import create_optimizer, create_scheduler
from deepgym.train import train, train_xgboost


if __name__ == '__main__':
    start = time.time()
    args, cfg = get_config()
    seed_everything(cfg.seed)
    logger = Logger(cfg.log_dir)
    dataset = create_dataset(cfg)
    model = create_model(cfg, dataset)
    if cfg.model.type == 'XGBoost':
        train_xgboost(dataset, model, cfg)
    else:
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        train(dataset, model, optimizer, scheduler, logger, cfg)
        logger.close()
    print(f"Use time: {time.time() - start} s")
