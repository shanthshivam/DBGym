"""
main.py
The entry of DBGym reposity.
"""

from dbgym.run import run
from dbgym.config import get_config


if __name__ == '__main__':
    config = get_config()
    config.merge_from_list(['dataset.type', 'graph', 'dataset.format', 'homo'])
    config.merge_from_list(['model.type', 'GNN', 'model.name', 'GCN'])
    run(config)
    config = get_config()
    config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'single'])
    config.merge_from_list(['model.type', 'MLP', 'model.name', 'MLP'])
    run(config)
    config = get_config()
    config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'join'])
    config.merge_from_list(['model.type', 'XGBoost', 'model.name', 'XGBoost'])
    run(config)
