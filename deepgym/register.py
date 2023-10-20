"""
register.py
This module contains some register functions.
"""

from typing import Any, Dict

graphgym_dict: Dict[str, Dict] = {}
act_dict: Dict[str, Any] = {}
graphgym_dict['act'] = act_dict
model_dict: Dict[str, Any] = {}
graphgym_dict['model'] = model_dict
optimizer_dict: Dict[str, Any] = {}
graphgym_dict['optimizer'] = optimizer_dict
scheduler_dict: Dict[str, Any] = {}
graphgym_dict['scheduler'] = scheduler_dict
loss_dict: Dict[str, Any] = {}
graphgym_dict['loss'] = loss_dict


def register(key: str, name: str, module: Any):
    """
    Base function for registering a module in DeepGym.

    Args:
    - key (string): The type of the module
    - name (string): The name of the module
    - module (Any): The module
    """

    graphgym_dict[key][name] = module
