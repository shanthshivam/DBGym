import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import os
import os.path as osp
from typing import (Dict, List, Union, Any)
import time

import numpy as np
import torch
import torch.nn as nn
import re


def reduce_sum_dict(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict


def map_value(array, map_dict):
    if torch.is_tensor(array):
        array = array.numpy()
    null = max(map_dict.values()) + 1
    # return np.vectorize(map_dict.__getitem__)(array)
    return np.vectorize(map_dict.get)(array, null)


def insert_dummy_row(df):
    # Create a dictionary with dummy values based on column types
    dummy_row = {}
    for column in df.columns:
        column_type = df[column].dtype
        if column_type == int:
            dummy_row[column] = 999
        elif column_type == float:
            dummy_row[column] = 0.0
        elif column_type == bool:
            dummy_row[column] = False
        elif column_type == object:
            dummy_row[column] = 'dummy'
        # Add more conditions for other column types if needed

    # Insert the dummy row into the DataFrame
    df = pd.concat([df, pd.DataFrame([dummy_row], columns=df.columns)],
                   ignore_index=True)

    return df


class bidict(dict):
    ''' Assuming unique mapping'''
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    def __setitem__(self, key, value):
        super(bidict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        if self[key] in self.inverse:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

    def merge_dict(self, other_dict: Dict):
        for key, value in other_dict.items():
            if key not in self.keys():
                self[key] = value
                self.inverse[value] = key


class IndexMap(dict):
    ''' Map any array to index from 0 - N-1'''
    def __init__(self, input: np.array, is_unique: bool = False):
        if not is_unique:
            input = np.unique(input)
        else:
            input = input.squeeze(1)
        # create a map from input to 0 - N-1
        self.n = input.shape[0]
        super().__init__(zip(input, range(self.n)))
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    def __setitem__(self, key, value):
        super(IndexMap, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        if self[key] in self.inverse:
            del self.inverse[self[key]]
        super(IndexMap, self).__delitem__(key)

    def update(self, input: np.array, is_unique: bool = False):
        if not is_unique:
            input = np.unique(input)
        else:
            input = input.squeeze(1)
        for val in input:
            if val not in self.keys():
                self[val] = self.n
                self.n += 1

    def merge_dict(self, other_dict: Dict):
        for key, value in other_dict.items():
            if key not in self.keys():
                self[key] = value
                self.inverse[value] = key

    def __repr__(self):
        return f"{self.__class__.__name__}(num = {self.n})"


class BaseModule:
    @classmethod
    def _from_dict(cls, dictionary: Dict[str, torch.tensor]):
        r"""
        Creates a data object from a python dictionary.

        Args:
            dictionary (dict): Python dictionary with key (string)
            - value (torch.tensor) pair.

        Returns:
            :class:`deepsnap.graph.Graph`: return a new Graph object
            with the data from the dictionary.
        """
        graph = cls(**dictionary)
        return graph

    def __getitem__(self, key: str):
        r"""
        Gets the data of the attribute :obj:`key`.
        """
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""
        Returns all names of the graph attributes.

        Returns:
            list: List of :class:`deepsnap.graph.Graph` attributes.
        """
        # filter attributes that are not observed by users
        # (1) those with value "None"; (2) those start with '_'
        attr = ['shortcut']
        keys = [
            key for key in self.__dict__.keys()
            if self[key] is not None and key[0] != "_" and key not in attr
        ]
        return keys

    def __len__(self) -> int:
        r"""
        Returns the number of all present attributes.

        Returns:
            int: The number of all present attributes.
        """
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""
        Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data.
        """
        return key in self.keys

    def __iter__(self):
        r"""
        Iterates over all present attributes in the data, yielding their
        attribute names and content.
        """
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""
        Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key: str, value) -> int:
        r"""
        Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if "index" in key else 0

    def __inc__(self, key: str, value) -> int:
        r""""
        Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_items if "index" in key or "id" in key else 0

    @property
    def num_items(self) -> int:
        r"""
        Return number of nodes in the graph.

        Returns:
            int: Number of nodes in the graph.
        """
        for key in self.keys:
            if torch.is_tensor(self[key]):
                return self[key].shape[0]

    def apply_tensor(self, func, *keys):
        r"""
        Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.

        Args:
            func (function): a function can be applied to a PyTorch tensor.
            *keys (string, optional): names of the tensor attributes that will
            be applied.

        Returns:
            :class:`deepsnap.graph.Graph`: Return the
            self :class:`deepsnap.graph.Graph`.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
            elif isinstance(self[key], dict):
                for obj_key, obj_item in self[key].items():
                    if torch.is_tensor(obj_item):
                        self[key][obj_key] = func(obj_item)
        return self

    def contiguous(self, *keys):
        r"""
        Ensures a contiguous memory layout for the attributes specified by
        :obj:`*keys`. If :obj:`*keys` is not given, all present attributes
        are ensured tohave a contiguous memory layout.

        Args:
            *keys (string, optional): tensor attributes which will be in
            contiguous memory layout.

        Returns:
            :class:`deepsnap.graph.Graph`: :class:`deepsnap.graph.Graph`
            object with specified tensor attributes in contiguous memory
            layout.
        """
        return self.apply_tensor(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        r"""
        Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes.

        Args:
            device: Specified device name.
            *keys (string, optional): Tensor attributes which will transfer to
            the specified device.
        """
        return self.apply_tensor(lambda x: x.to(device), *keys)

    def clone(self):
        r"""
        Deepcopy the graph object.

        Returns:
            :class:`deepsnap.graph.Graph`:
            A cloned :class:`deepsnap.graph.Graph` object with deepcopying
            all features.
        """
        dictionary = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                dictionary[k] = v.clone()
        return self.__class__._from_dict(dictionary)

    def _size_repr(self, value):
        r"""
        Returns:
            list: Size of each element in value
        """
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float):
            return [1]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [len(value)]
        elif isinstance(value, dict):
            if len(value) > 0 and torch.is_tensor(next(iter(value.values()))):
                return {key: list(val.size()) for key, val in value.items()}
            else:
                return {len(value)}
        else:
            return []

    def __repr__(self):
        info = [f"{key}={self._size_repr(item)}" for key, item in self]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def slice(self, id):
        self.apply_tensor(lambda x: x[id])


def is_time_col(df, col):
    try:
        pd.to_datetime(df[col], errors='coerce')
        condition = df[col].notnull().any()
        return condition and df[col].astype(str).str.contains('-').any()

    except ValueError:
        return False


class Table(BaseModule):
    def __init__(self,
                 df: pd.DataFrame,
                 name: str = '',
                 known_keys: List = None):
        self.df = df
        self.name = name
        self.known_keys = known_keys or []
        self.key: Dict[str, torch.tensor] = {}
        self.feat_int: Dict[str, torch.tensor] = {}
        self.feat_float: Dict[str, torch.tensor] = {}
        self.ctypes: Dict[str, str] = {}
        self.dtypes: Dict[str, str] = {}
        self.col_to_enc: Dict[str, Dict[Any, int]] = {}

        self.check_cols()
        self.infer_types()
        self.split()

    def num_rows(self):
        return self.df.shape[0]

    def get_columns(self):
        return self.df.columns.tolist()

    def get_keys(self):
        return [col for col, val in self.ctypes.items() if val == 'key']

    def get_feats(self):
        return [col for col, val in self.ctypes.items() if val != 'key']

    def check_cols(self):
        if 'type' in self.get_columns():
            self.df.rename(columns={'type': 'dtype'}, inplace=True)

    def infer_types(
            self,
            dtypes: Dict[str, str] = None,
            ctypes: Dict[str, str] = None,
            int_ratio_max: float = 0.5,
            int_abs_max: int = 10000,
    ):
        self.dtypes = dtypes or {}
        self.ctypes = ctypes or {}
        for col in self.get_columns():
            if col in self.dtypes:
                dtype = self.dtypes[col]
            # or is_time_col(self.df, col)
            elif 'date' in col or 'time' in col:
                dtype = 'datetime'
            elif not is_integer_dtype(
                    self.df[col].dtype) and not is_float_dtype(
                        self.df[col].dtype):
                dtype = 'category'
            else:
                dtype = self.df[col].dtype.name
            self.dtypes[col] = dtype

            # ctype: key, feat_int, feat_float
            if col in self.ctypes:
                ctype = self.ctypes[col]
            elif re.sub(r'^\d+|\d+$', '', col)[-3:] == '_id':
                ctype = 'key'
            elif dtype == 'category' or 'int' in dtype:
                ctype = 'feat_int'
            else:
                ctype = 'feat_float'
            self.ctypes[col] = ctype

            # patch: determine if feat_int should be feat_float
            if 'feat' in ctype and 'int' in dtype:
                uni = self.df[col].nunique()
                if uni / len(self.df) > int_ratio_max or uni > int_abs_max:
                    self.ctypes[col] = 'feat_float'
                    self.dtypes[col] = 'float64'

    def prepare_encoder_feat(self, materialize=True):
        for col in self.get_columns():
            dtype = self.dtypes[col]
            val = self.df[[col]]
            # preprocess datetime
            if self.dtypes[col] == 'datetime':
                val = pd.to_datetime(self.df[col]).to_frame()
                dtype = 'int64'
            if self.ctypes[col] == 'feat_float':
                val = val.astype(dtype)
                enc = MinMaxScaler((0, 2))
                enc.fit(val)
                self.col_to_enc[col] = enc
                if materialize:
                    val = val.fillna(val.mean()).astype(dtype)
                    val = enc.transform(val)
                    self.feat_float[col] = torch.tensor(val, dtype=torch.float)
            elif self.ctypes[col] == 'feat_int':
                if self.dtypes[col] == 'category':
                    val = val.fillna('missing').astype(dtype)
                elif self.dtypes[col] == 'int64':
                    val = val.fillna(-1).astype(dtype)
                enc = IndexMap(val)
                self.col_to_enc[col] = enc
                if materialize:
                    val = map_value(val, enc)
                    self.feat_int[col] = torch.tensor(
                        val, dtype=torch.int64).squeeze(-1)

    def update_encoder_feat(self):
        for col in self.get_columns():
            if self.ctypes[col] == 'feat_int':
                if self.dtypes[col] == 'category':
                    val = self.df[col].fillna('missing').astype(self.dtypes[col])
                elif self.dtypes[col] == 'int64':
                    val = self.df[col].fillna(-1).astype(self.dtypes[col])
                self.col_to_enc[col].update(val)

    def materialize_key(self, enc_dict: Dict[str, IndexMap]):
        for col in self.get_columns():
            if self.ctypes[col] == 'key':
                ## Change "missing" to -1
                val = self.df[[col]].fillna(-1)
                val = map_value(val, enc_dict[col])
                self.key[col] = torch.tensor(val, dtype=torch.int64).squeeze(-1)

    def materialize_feat(self):
        for col, enc in self.col_to_enc.items():
            dtype = self.dtypes[col]
            val = self.df[[col]]
            # preprocess datetime
            if self.dtypes[col] == 'datetime':
                val = pd.to_datetime(self.df[col]).to_frame()
                dtype = 'int64'
            if isinstance(enc, IndexMap):
                val = val.fillna('missing').astype(dtype)
                val = map_value(val, enc)
                self.feat_int[col] = torch.tensor(
                    val, dtype=torch.int64).squeeze(-1)
            else:
                val = val.fillna(val.mean()).astype(dtype)
                val = enc.transform(val)
                self.feat_float[col] = torch.tensor(val, dtype=torch.float)

    def clean(self):
        self.key: Dict[str, torch.tensor] = {}
        self.feat_int: Dict[str, torch.tensor] = {}
        self.feat_float: Dict[str, torch.tensor] = []

    def split(self,
              ratio: List = [0.1, 0.05, 0.05],
              name: List = ['train', 'val', 'test']):
        r""" Get the label mask for train/val/test, for each col"""
        assert sum(ratio) < 1, f"sum of label ratio {ratio} greater than 1"
        n = self.num_rows()
        # todo: add function to load N/A value for test set
        for i, _ in enumerate(ratio):
            self[f'mask_{name[i]}'] = {}
        self['mask_input'] = {}
        self['mask_pred'] = {}
        for col in self.get_feats():
            self.df[col].isna()
            col_na = torch.tensor(self.df[col].isna().values)
            if col_na.any():
                self['mask_pred'][col] = col_na
                ids = (~self['mask_pred'][col]).nonzero().squeeze(1)
                ids = ids[torch.randperm(ids.shape[0])]
            else:
                ids = torch.randperm(n)
            start = 0
            for i, ratio_i in enumerate(ratio):
                end = start + int(ratio_i * n)
                self[f'mask_{name[i]}'][col] = torch.zeros(n, dtype=torch.bool)
                self[f'mask_{name[i]}'][col][ids[start:end]] = True
                start = end
            self['mask_input'][col] = torch.zeros(n, dtype=torch.bool)
            self['mask_input'][col][ids[end:]] = True

    def resplit(self):
        r"""resplit mask_train and mask_input, other masks keep fixed"""
        if 'mask_train' in self:
            for col in self['mask_train'].keys():
                id_train = self['mask_train'][col].nonzero().squeeze(1)
                id_input = self['mask_input'][col].nonzero().squeeze(1)
                ids = torch.cat((id_train, id_input), dim=0)
                ids = ids[torch.randperm(ids.shape[0])]
                id_train = ids[:id_train.shape[0]]
                id_input = ids[id_train.shape[0]:]
                self['mask_train'][col].fill_(False)
                self['mask_input'][col].fill_(False)
                self['mask_train'][col][id_train] = True
                self['mask_input'][col][id_input] = True


class DataBase(BaseModule):
    def __init__(self,
                 name: str,
                 dir: str,
                 file_type: str = 'csv',
                 known_tables: List = None,
                 known_keys: List = None):
        super().__init__()
        self.name = name
        self.dir = osp.join(dir, name)
        self.file_type = file_type
        self.table_names = known_tables or []
        self.known_keys = known_keys or []
        self.tables: Dict[str, Table] = {}
        self.key_to_table: Dict[str, List[str]] = {}
        self.table_to_key: Dict[str, List[str]] = {}
        self.key_to_enc: Dict[str, Any] = {}

    def load(self, sep=','):
        files = os.listdir(self.dir)
        self.table_names = [
            '.'.join(file.split('.')[:-1]) for file in files
            if file.endswith(self.file_type)
        ]

        for table_name in self.table_names:
            fname = osp.join(self.dir, f'{table_name}.{self.file_type}')
            df = pd.read_csv(fname, sep=sep, low_memory=False)
            df.dropna(axis=1, how='all')
            table = Table(df, table_name, self.known_keys)
            self.tables[table_name] = table
            keys = table.get_keys()
            self.table_to_key[table_name] = keys
            keys = dict(zip(keys, [[table_name] for _ in range(len(keys))]))
            self.key_to_table = reduce_sum_dict(self.key_to_table, keys)

    def load_join(self, main_table='', sep=','):
        files = os.listdir(self.dir)
        self.table_names = [
            '.'.join(file.split('.')[:-1]) for file in files
            if file.endswith(self.file_type)
        ]
        table_list = []
        for table_name in self.table_names:
            fname = osp.join(self.dir, f'{table_name}.{self.file_type}')
            df = pd.read_csv(fname, sep=sep, low_memory=False)
            df = df.dropna(axis=1, how='all')
            df.dropna()
            if table_name == main_table:
                main_df = df
            else:
                table_list.append(df)

        main_name = main_table + '_id'
        names = set(main_df.columns.tolist())
        for table in table_list:
            common_keys = list(names.intersection(set(table.columns.tolist())))
            real_keys = []
            for key in common_keys:
                if 'id' in key and len(table[key]) == len(table[key].unique()):
                    real_keys.append(key)
            if real_keys:
                main_df = main_df.merge(table, on=real_keys, how='left')

        main_df = main_df.dropna(axis=1, how='all')
        table = Table(main_df, 'join_table', self.known_keys)
        self.tables['join_table'] = table
        keys = [main_name]
        self.table_to_key['join_table'] = keys
        keys = dict(zip(keys, [['join_table'] for _ in range(len(keys))]))
        self.key_to_table = reduce_sum_dict(self.key_to_table, keys)

        self.prepare_encoder_key()
        table = self.tables['join_table']
        for name in table.ctypes:
            if name != main_name and table.ctypes[name] == 'key':
                # table.dtypes[name] = 'int64'
                table.ctypes[name] = 'feat_int'
        table.materialize_key(self.key_to_enc)
        table.prepare_encoder_feat(materialize=True)

    def prepare_encoder_key(self):
        for key, tables in self.key_to_table.items():
            # infer column with unique keys
            for i, table in enumerate(tables):
                if table in key:
                    break
            table = tables.pop(i)
            tables.insert(0, table)
            # change 'missing' to -1
            if self.tables[tables[0]].dtypes[key] == 'category':
                fillna = '-1'
            else:
                fillna = -1
            self.key_to_enc[key] = IndexMap(
                self.tables[tables[0]].df[key].fillna(fillna))
            for table in tables[1:]:
                if self.tables[table].dtypes[key] == 'category':
                    fillna = '-1'
                else:
                    fillna = -1
                self.key_to_enc[key].update(
                    self.tables[table].df[key].fillna(fillna))

    def prepare_encoder(self, materialize=True):
        self.prepare_encoder_key()
        # encode features
        for name, table in self.tables.items():
            if materialize:
                table.materialize_key(self.key_to_enc)
            table.prepare_encoder_feat(materialize=materialize)

    def update_encoder(self):
        '''When internal value changes'''
        for key, tables in self.key_to_table.items():
            for table in tables:
                self.key_to_enc[key].update(self.tables[table].df[key])
        for table_name, table in self.tables.items():
            table.update_encoder_feat()

    def materialize(self):
        for name, table in self.tables.items():
            table.materialize_key(self.key_to_enc)
            table.materialize_feat()

    def clean(self):
        for name, table in self.tables.items():
            table.clean()

    @property
    def dtypes(self):
        return {name: table.dtypes for name, table in self.tables.items()}

    @property
    def ctypes(self):
        return {name: table.ctypes for name, table in self.tables.items()}


## tests

# id = IndexMap(np.array([1,2,3]))
# id.update(np.array([6, 5]))
# id.update(np.array(['abc','123']))
# breakpoint()

# self.prepare_encoder()
# self.tables['loan'].df = insert_dummy_row(self.tables['loan'].df)
# print(self.key_to_enc)
# self.update_encoder()
# print(self.key_to_enc)

# decode datetime
# if self.dtypes[col] == 'datetime':
#     print(self.df[col])
#     self.df[col] = pd.to_datetime(self.df[col])
#     self.dtypes[col] = 'int64'
#     # self.df[col] = pd.to_datetime(self.df[col])
#     print(self.df[col])
#     enc = MinMaxScaler((0, 2))
#     enc.fit(self.df[[col]])
#     self.df[[col]] = enc.transform(self.df[[col]])
#     print(self.df[col])
#     self.df[[col]] = enc.inverse_transform(self.df[[col]])
#     breakpoint()
#     self.df[col] = pd.to_datetime(self.df[col].astype('int64'))
#     print(self.df[col])
#     breakpoint()
