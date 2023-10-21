"""
db.py
DataBase module.
"""

import os
import re
from typing import Dict, List, Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from yacs.config import CfgNode
import torch

MISSING_INT = -114514
MISSING = '1919810'


def reduce_sum_dict(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge and sum two dictionaries, returning a new dictionary
    """

    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict


class IndexMap(dict):
    """
    Map any array to index from 0 - N-1
    """

    def __init__(self, inpt, is_unique: bool = False):
        if not is_unique:
            inpt = np.unique(inpt)
        else:
            inpt = inpt.squeeze(1)
        # create a map from input to 0 - N-1
        inpt = list(inpt)
        if MISSING in inpt:
            inpt.remove(MISSING)
            inpt.append(MISSING)
        if MISSING_INT in inpt:
            inpt.remove(MISSING_INT)
            inpt.append(MISSING_INT)
        self.n = len(inpt)
        super().__init__(zip(inpt, range(self.n)))
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        if self[key] in self.inverse:
            del self.inverse[self[key]]
        super().__delitem__(key)

    def update(self, inpt, is_unique: bool = False):
        if not is_unique:
            inpt = np.unique(inpt)
        else:
            inpt = inpt.squeeze(1)
        for val in inpt:
            if val not in self.keys():
                self[val] = self.n
                self.n += 1

    def merge_dict(self, other_dict: Dict):
        """
        Merges other_dict into current dictionary, updating inverse mapping.
        """
        for key, value in other_dict.items():
            if key not in self.keys():
                self[key] = value
                self.inverse[value] = key

    def __repr__(self):
        return f"{self.__class__.__name__}(num = {self.n})"


def map_value(array, map_dict: IndexMap) -> np.ndarray:
    """
    Map array to index
    """
    if torch.is_tensor(array):
        array = array.numpy()
    null = max(map_dict.values()) + 1
    return np.vectorize(map_dict.get)(array, null)


class Table:
    """
    Table module for single table
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # column types: key, feature
        self.ctypes: Dict[str, str] = {}
        # data types: int64, float64, category, time
        self.dtypes: Dict[str, str] = {}
        # key
        self.key: Dict[str, torch.Tensor] = {}
        # discrete feature
        self.feature_disc: Dict[str, torch.Tensor] = {}
        # continuous feature
        self.feature_cont: Dict[str, torch.Tensor] = {}
        # feature encoder for each column
        self.feat_encoder: Dict[str, Dict[Any, int]] = {}

        self.infer_dtypes()
        self.infer_ctypes()

    def num_rows(self):
        """
        Get the number of rows
        """
        return self.df.shape[0]

    def get_columns(self):
        """
        Get the columns of dataframe
        """
        return self.df.columns.tolist()

    def get_keys(self):
        """
        Get the key columns of dataframe
        """
        return [col for col, val in self.ctypes.items() if val == 'key']

    def get_feats(self):
        """
        Get the feature columns of dataframe
        """
        return [col for col, val in self.ctypes.items() if val == 'feature']

    def infer_dtypes(self, dtypes: Dict[str, str] = None):
        """
        Infer the data type of columns
        """

        self.dtypes = dtypes or {}

        for col in self.get_columns():
            if col in self.dtypes:
                continue
            dtype = self.df[col].dtype
            # dtype: int64, float64, category, time
            if dtype == 'datetime64[ns]':
                dtype = 'time'
            elif dtype == 'int64':
                dtype= 'int64'
            elif dtype == 'float64':
                dtype = 'float64'
                column = self.df[col].dropna()
                if column.apply(lambda x: x.is_integer()).all():
                    dtype = 'int64'
            elif dtype == 'object':
                try:
                    pd.to_datetime(self.df[col][self.df[col].first_valid_index()])
                    dtype = 'time'
                except Exception:
                    dtype = 'category'
            else:
                raise ValueError(f"Unknown dtype: {dtype}")

            self.dtypes[col] = dtype

    def infer_ctypes(
            self,
            ctypes: Dict[str, str] = None,
            int_ratio_max: float = 0.1,
            int_abs_max: int = 1000,
    ):
        """
        Infer the column type of columns
        """

        self.ctypes = ctypes or {}

        for col in self.get_columns():
            # ctype: key, feature
            if col in self.ctypes:
                continue
            if '_' in col:
                ctype = 'key'
            else:
                ctype = 'feature'
                # patch: determine if int feature should be float feature
                if self.dtypes[col] == 'int64':
                    uni = self.df[col].nunique()
                    if uni / len(self.df) > int_ratio_max or uni > int_abs_max:
                        self.dtypes[col] = 'float64'

            self.ctypes[col] = ctype

    def prepare_feature(self):
        """
        Prepare feature & its encoder
        """

        for col in self.get_columns():
            if self.ctypes[col] == 'key':
                continue
            dtype = self.dtypes[col]
            column = self.df[[col]]
            # ignore time features
            if dtype == 'time':
                continue
            if dtype == 'int64':
                column = column.fillna(MISSING_INT).astype(dtype)
                encoder = IndexMap(column)
                self.feat_encoder[col] = encoder
                value = map_value(column, encoder)
                self.feature_disc[col] = torch.tensor(value, dtype=torch.int64)
            elif dtype == 'float64':
                column = column.fillna(column.mean()).astype(dtype)
                encoder = MinMaxScaler((0, 2))
                encoder.fit(column)
                self.feat_encoder[col] = encoder
                value = encoder.transform(column)
                self.feature_cont[col] = torch.tensor(value, dtype=torch.float64)
            elif dtype == 'category':
                column = column.fillna(MISSING).astype(dtype)
                encoder = IndexMap(column)
                self.feat_encoder[col] = encoder
                value = map_value(column, encoder)
                self.feature_disc[col] = torch.tensor(value, dtype=torch.int64)

    def valid_indices(self, col: str):
        """
        Return valid indices of target column
        """
        valid = torch.tensor(pd.notna(self.df[col]))
        return torch.nonzero(valid).flatten()

    def clean(self):
        """
        Reset keys and features
        """
        self.key: Dict[str, torch.Tensor] = {}
        self.feature_disc: Dict[str, torch.Tensor] = {}
        self.feature_cont: Dict[str, torch.Tensor] = {}


class DataBase:
    """
    DataBase module for relational database
    """

    def __init__(self, path: str):
        self.path = path
        self.names = []
        self.tables: Dict[str, Table] = {}
        self.key_to_table: Dict[str, List[str]] = {}
        self.table_to_key: Dict[str, List[str]] = {}
        self.key_to_enc: Dict[str, Any] = {}

    def load(self):
        """
        Load tables
        """

        for file in os.listdir(self.path):
            if file.endswith('.csv'):
                self.names.append(tuple([file.split('.')[0], 'csv']))
            elif file.endswith('.sql'):
                self.names.append(tuple([file.split('.')[0], 'sql']))

        for name, typ in self.names:
            fname = os.path.join(self.path, f'{name}.{typ}')
            if typ == 'csv':
                df = pd.read_csv(fname, low_memory=False)
                df.dropna(axis=1, how='all')
                table = Table(df)
                self.tables[name] = table
                keys = table.get_keys()
                self.table_to_key[name] = keys
                keys = dict(zip(keys, [[name] for _ in range(len(keys))]))
                self.key_to_table = reduce_sum_dict(self.key_to_table, keys)
            elif typ == 'sql':
                # TODO: finish sql support
                continue

    def prepare_encoder(self):
        """
        Prepare encoder for table columns
        """
        for table in self.tables.values():
            table.prepare_feature()

    def clean(self):
        """
        Reset tables
        """
        for table in self.tables.values():
            table.clean()

    @property
    def dtypes(self):
        """
        Return dtypes of tables
        """
        return {name: table.dtypes for name, table in self.tables.items()}

    @property
    def ctypes(self):
        """
        Return ctypes of tables
        """
        return {name: table.ctypes for name, table in self.tables.items()}


class Tabular:
    """
    Tabular data module for relational database
    """

    def __init__(self, path: str, file: str, col: str):
        self.path = path
        self.file = file
        self.col = col
        self.x_d = None
        self.x_c = None
        self.y = None
        self.mask = {}
        self.output = 0

    def load_csv(self):
        """
        load a table from csv
        """

        path = os.path.join(self.path, f'{self.file}.csv')
        df = pd.read_csv(path, low_memory=False)
        df.dropna(axis=1, how='all')
        self.load_df(df)

    def load_join(self):
        """
        load join table
        """

        names = []
        for file in os.listdir(self.path):
            if file.split('.')[1] == 'csv':
                names.append(tuple([file.split('.')[0], 'csv']))

        dfs = []
        for name, typ in names:
            path = os.path.join(self.path, f'{name}.{typ}')
            df = pd.read_csv(path, low_memory=False)
            df.dropna()
            if name == self.file:
                main_df = df
            else:
                dfs.append(df)

        cols = set(main_df.columns.tolist())
        for df in dfs:
            common_keys = list(cols.intersection(set(df.columns.tolist())))
            real_keys = []
            for key in common_keys:
                if 'id' in key and len(df[key]) == len(df[key].unique()):
                    real_keys.append(key)
            if real_keys:
                main_df = main_df.merge(df, on=real_keys, how='left')

        self.load_df(main_df)

    def load_df(self, df: pd.DataFrame):
        """
        load dataframe
        """

        table = Table(df)
        table.prepare_feature()
        if self.col in table.feature_cont:
            self.y = table.feature_cont[self.col]
            self.output = 1
            del table.feature_cont[self.col]
        elif self.col in table.feature_disc:
            self.y = table.feature_disc[self.col]
            self.output = torch.max(self.y[table.valid_indices(self.col)]).item() + 1
            del table.feature_disc[self.col]
        else:
            raise ValueError(f'Column {self.col} not found.')

        # concatenate features
        if table.feature_disc:
            feature = torch.cat([f.view(-1, 1) for f in table.feature_disc.values()], dim=1)
        else:
            feature = torch.zeros((len(table.df), 1))
        self.x_d = feature.to(torch.int32)
        if table.feature_cont:
            feature = torch.cat([f.view(-1, 1) for f in table.feature_cont.values()], dim=1)
        else:
            feature = torch.zeros((len(table.df), 1))
        self.x_c = feature.to(torch.float32)

        # split
        names = ["train", "valid", "test"]
        ratios = [0, 0.8, 0.9, 1]
        valid_indices = table.valid_indices(self.col)
        sizes = [int(ratio * valid_indices.shape[0]) for ratio in ratios]
        indices = torch.randperm(sizes[3])
        for i, name in enumerate(names):
            self.mask[name] = valid_indices[indices[sizes[i]: sizes[i + 1]]]
