"""
main.py
This entry of DeepGym reposity.
"""

from deepgym.config import get_config
from deepgym.utils import seed_everything
from deepgym.logger import Logger
from deepgym.dataset import create_dataset
from deepgym.model import create_model
from deepgym.optimizer import create_optimizer, create_scheduler
from deepgym.train import train

if __name__ == '__main__':
    args, cfg = get_config()
    seed_everything(cfg.seed)
    logger = Logger(cfg.log_dir)
    dataset = create_dataset(cfg)
    model = create_model(cfg)
    optimizer = create_optimizer(cfg, model.parameters())
    scheduler = create_scheduler(cfg, optimizer)
    print("train begins")
    train(dataset, model, optimizer, scheduler, logger, cfg)
    
