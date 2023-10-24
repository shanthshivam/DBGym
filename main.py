"""
main.py
The entry of DBGym reposity.
"""

from dbgym.run import run
from dbgym.config import get_config, set_from_path


if __name__ == '__main__':
    config = get_config()
    # config.merge_from_list(['dataset.type', 'graph'])
    # config.merge_from_list(['model.name', 'GCN'])
    # config.merge_from_list(['dataset.name', 'rdb1-ather'])
    # config.merge_from_list(['dataset.query', 'entry_examination.Cholesterol'])
    # config.merge_from_list(['dataset.query', 'death.DeathReason'])
    config = set_from_path("tests/test_config_xgboost.yaml")
    stats = run(config)
    # config = get_config()
    # config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'single'])
    # config.merge_from_list(['model.name', 'MLP'])
    # stats = run(config)
    # config = get_config()
    # config.merge_from_list(['dataset.type', 'tabular', 'dataset.format', 'join'])
    # config.merge_from_list(['model.name', 'XGBoost'])
    # stats = run(config)
