"""
test.py
Testing Datasets.
"""

import os
import time
from dbgym.db import DataBase
from dbgym.db2pyg import DB2PyG

if __name__ == '__main__':
    DATA_DIR = 'Datasets/'
    NAME = 'example'
    CSV = 'loan'
    COL = 'Status'
    print(f'Loading dataset {NAME}, please wait patiently.')
    start = time.time()
    path = os.path.join(DATA_DIR, NAME)
    db = DataBase(path)
    db.load()
    db.prepare_encoder()
    converter = DB2PyG(db, CSV, COL)
    print(converter.hetero)
    print(f"Load time: {time.time() - start} s")
