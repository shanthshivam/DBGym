"""
seed.py
This module contains some seed functions.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    The seed_everything function, set seed

    Args:
    - seed: The seed to set
    """

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
