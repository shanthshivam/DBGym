"""
main.py
The entry of DBGym reposity.
"""

from dbgym.config import get_config
from dbgym.run import run

if __name__ == '__main__':
    config = get_config()
    stats = run(config)
