"""
main.py
The entry of DBGym reposity.
"""

from dbgym.run import run
from dbgym.config import get_config, set_from_path


if __name__ == '__main__':
    config = get_config()
    stats = run(config)