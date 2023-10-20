"""
example.py
examine the not unique id column
"""

import os
import pandas as pd


def examine_integer(directory):
    """
    examine the non-integer id column
    """

    dirs = os.listdir(directory)
    for d in dirs:
        path = os.path.join(directory, d)
        for file in os.listdir(path):
            name = file.split('.')[0]
            df = pd.read_csv(os.path.join(path, file), low_memory=False)
            dtype = df[name + '_id'].dtype
            if dtype == 'int64':
                continue
            elif dtype == 'float64':
                if df[name + '_id'].apply(lambda x: x.is_integer()).all():
                    continue
            print(os.path.join(path, file))


def examine_unique(directory):
    """
    examine the non-unique id column
    """

    dirs = os.listdir(directory)
    for d in dirs:
        path = os.path.join(directory, d)
        for file in os.listdir(path):
            name = file.split('.')[0]
            df = pd.read_csv(os.path.join(path, file), low_memory=False)
            if not df[name + '_id'].is_unique:
                print(os.path.join(path, file))


if __name__ == '__main__':
    examine_unique('Datasets')
    # examine_integer('Datasets')
