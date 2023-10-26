"""
main.py
The entry of DBGym reposity.
"""

from dbgym.config import get_config
from dbgym.run import run
import torch

def experiment(config_path):
    config = get_config()
    config.merge_from_file('./default.yaml')
    config.merge_from_file(config_path)
    stats = run(config)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # config = get_config()
    # config.merge_from_file('./default.yaml')
    # config.merge_from_file('./share.yaml')
    # stats = run(config)
    # print(stats)
    import os
    import sys
    
    cfg_path = './cuda1'
    if len(sys.argv)> 1:
        cfg_path = f'./cuda{sys.argv[1]}'
        for file in os.listdir(cfg_path):
            if file.endswith('.yaml'):
                print(file)
                experiment(os.path.join(cfg_path, file))
    else: 
        experiment('./share.yaml')
