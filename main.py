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
    # config.merge_from_list(['dataset.name', 'rdb1-ather', 'train.epoch', 200])
    # config.merge_from_list(['dataset.file', 'entry_examination', 'dataset.column', 'Cholesterol'])
    # config.merge_from_list(['dataset.file', 'death', 'dataset.column', 'DeathReason'])
    stats = run(config)
    config = get_config()
    config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'single'])
    config.merge_from_list(['model.type', 'MLP', 'model.name', 'MLP'])
    stats = run(config)
    config = get_config()
    config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'join'])
    config.merge_from_list(['model.type', 'XGBoost', 'model.name', 'XGBoost'])
    stats = run(config)
