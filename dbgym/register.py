"""
register.py
This module save registered modules.
"""

from collections import defaultdict
from typing import Any, Dict

module_dict: Dict[str, Dict] = defaultdict(dict)

def register(module_type: str, name: str, module: Any):
    """
    Base function for registering a module in DBGym.

    Args:
    - module_type (string): The type of the module
    - name (string): The name of the module
    - module (Any): The module
    """

    module_dict[module_type][name] = module
