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
    converter = DB2PyG(db, target_csv=CSV, target_col=COL)
    converter.Embedding_hetero()
    converter.Embedding_homo()
    graph_homo = converter.homo
    print(graph_homo)
    print(f"Load time: {time.time() - start} s")
