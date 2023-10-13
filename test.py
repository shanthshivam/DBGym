"""
test.py
Testing Datasets.
"""

import time
from deepgym.db import DataBase
# from deepgym.db2pyg import DB2PyG

if __name__ == '__main__':
    DATA_DIR = 'Datasets/'
    NAME = 'financial'
    CSV = 'loan'
    COL = 'status'
    print(f'Loading dataset {NAME}, please wait patiently.')
    start = time.time()
    db = DataBase(NAME, DATA_DIR)
    db.load()
    db.prepare_encoder()
    print(f"Load time: {time.time() - start} s")
