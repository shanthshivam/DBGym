"""
register.py
This module contains some register functions.
"""

from typing import Any, Dict

dbgym_dict: Dict[str, Dict] = {}
act_dict: Dict[str, Any] = {}
dbgym_dict['act'] = act_dict
graph_model_dict: Dict[str, Any] = {}
dbgym_dict['graph_model'] = graph_model_dict
tabular_model_dict: Dict[str, Any] = {}
dbgym_dict['tabular_model'] = tabular_model_dict
optimizer_dict: Dict[str, Any] = {}
dbgym_dict['optimizer'] = optimizer_dict
scheduler_dict: Dict[str, Any] = {}
dbgym_dict['scheduler'] = scheduler_dict
loss_dict: Dict[str, Any] = {}
dbgym_dict['loss'] = loss_dict


def register(key: str, name: str, module: Any):
    """
    Base function for registering a module in DBGym.

    Args:
    - key (string): The type of the module
    - name (string): The name of the module
    - module (Any): The module
    """

    dbgym_dict[key][name] = module
