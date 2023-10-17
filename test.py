"""
test.py
Testing Datasets.
"""

import os
import time
from deepgym.db import DataBase
from deepgym.db2pyg import DB2PyG

if __name__ == '__main__':
    DATA_DIR = 'Datasets/'
    NAME = 'financial'
    CSV = 'loan'
    COL = 'status'
    print(f'Loading dataset {NAME}, please wait patiently.')
    start = time.time()
    path = os.path.join(DATA_DIR, NAME)
    db = DataBase(path)
    db.load()
    db.prepare_encoder()
    converter = DB2PyG(db, CSV, COL)
    print(converter.hetero)
    print(converter.homo)
    converter.split(42)
    print(f"Load time: {time.time() - start} s")
