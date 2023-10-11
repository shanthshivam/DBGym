"""
utils.py
This module contains some utility functions.
"""

import os
import random
import torch
import numpy as np


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
