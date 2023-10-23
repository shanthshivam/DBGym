"""
db2pyg.py
Module to transform database into graph.
"""

import torch
import numpy as np
import pandas as pd
from dbgym.db import DataBase
from torch_geometric.data import HeteroData


class DB2PyG:
    """
    This class is used to convert DataBase in db.py to PyG graphs

    Args:
    - db: The relational DataBase to convert
    - table: The name of target table
    - col: The name of target column
    """

    def __init__(self, db: DataBase, table: str, col: str):
        self.db = db
        self.table = table
        self.col = col

        self.lookup_table = {}
        self.duplicate = {}
        self.graph = HeteroData()
        # Construct a heterogeneous graph using the provided tables and features
        self.output = 0
        self.init_node()
        self.init_edge()

        self.mask = {}
        self.split()

    def init_node(self):
        """
        Construct heterogeneous graph nodes
        """

        graph = self.graph

        names = list(self.db.tables.keys())
        names.remove(self.table)
        names.insert(0, self.table)

        # create nodes in the graph
        for name in names:
            table = self.db.tables[name]
            feat_d = list(table.feature_disc.keys())
            feat_c = list(table.feature_cont.keys())
            if name == self.table:
                if self.col in feat_d:
                    feat_d.remove(self.col)
                    graph.y = table.feature_disc[self.col].squeeze()
                    self.output = torch.max(graph.y[table.valid_indices(self.col)]).item() + 1
                else:
                    feat_c.remove(self.col)
                    graph.y = table.feature_cont[self.col].squeeze()
                    self.output = 1

            # concatenate features
            if feat_d:
                feature = torch.cat([table.feature_disc[col].view(-1, 1) for col in feat_d], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            graph[name].x_d = feature.to(torch.int32)
            if feat_c:
                feature = torch.cat([table.feature_cont[col].view(-1, 1) for col in feat_c], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            graph[name].x_c = feature.to(torch.float32)
            graph[name].num_nodes = len(table.df)

            # set up look up table
            keys = table.df["_" + name].to_list()
            if len(keys) != len(set(keys)):
                lookup_dict = {}
                for i, key in enumerate(keys):
                    if key in lookup_dict:
                        lookup_dict[key].append(i)
                    else:
                        lookup_dict[key] = [i]
                self.lookup_table[name] = lookup_dict
                self.duplicate[name] = True
            else:
                values = list(range(0, len(keys)))
                self.lookup_table[name] = dict(zip(keys, values))
                self.duplicate[name] = False

    def init_edge(self):
        """
        Construct heterogeneous graph edges
        """

        graph = self.graph
        lookup_table = self.lookup_table
        duplicate = self.duplicate

        # create edges in the graph
        for name, table in self.db.tables.items():
            keys = table.get_keys()
            if len(keys) == 1:
                continue

            length = len(self.db.tables[name].df)
            array = torch.arange(length).view(1, -1)
            for key in keys[1:]:
                column = table.df[key]
                key = key[1:]

                if duplicate[key]:
                    edge_index = [[], []]
                    lut = lookup_table[key]
                    for i in range(length):
                        if column[i] in lut:
                            edge_index[0] += len(lut[column[i]]) * [i]
                            edge_index[1] += lut[column[i]]
                    edge_index = torch.tensor(edge_index)
                else:
                    def lookup(x):
                        return lookup_table[key].get(x, -1)
                    point_to = torch.tensor(np.vectorize(lookup)(column))
                    edge_index = torch.cat([array, point_to.view(1, -1)], dim=0)
                    edge_index = edge_index[:, point_to != -1]

                self.create_edge(graph, edge_index, name, key)

    def create_edge(self, graph: HeteroData, edge_index: torch.Tensor, name: str, key: str):
        """
        Construct heterogeneous graph edges
        """
        if graph[name, "to", key].num_edges:
            index = graph[name, "to", key].edge_index
            graph[name, "to", key].edge_index = torch.cat([index, edge_index], dim=1)
        else:
            graph[name, "to", key].edge_index = edge_index
        edge_index = edge_index[[1, 0]]
        if graph[key, "to", name].num_edges:
            index = graph[key, "to", name].edge_index
            graph[key, "to", name].edge_index = torch.cat([index, edge_index], dim=1)
        else:
            graph[key, "to", name].edge_index = edge_index

    def split(self, seed=None):
        """
        Split the dataset into training, validation, and test sets.

        Args:
        - seed: The seed to set

        Returns:
        - masks: A dictionary containing masks for graph
        """

        # ensuring reproducibility
        if seed:
            torch.manual_seed(seed)

        names = ["train", "valid", "test"]
        ratios = [0, 0.8, 0.9, 1]
        valid_indices = self.db.tables[self.table].valid_indices(self.col)
        sizes = [int(ratio * valid_indices.shape[0]) for ratio in ratios]
        indices = torch.randperm(sizes[3])

        # split the dataset into training, validation, and test sets
        for i, name in enumerate(names):
            self.mask[name] = valid_indices[indices[sizes[i]: sizes[i + 1]]]
        self.mask['all'] = torch.arange(len(self.db.tables[self.table].df))

    def to(self, device):
        """
        to device
        """
        return self.graph.to(device)

    def fill_na(self, pred: torch.Tensor, path: str):
        """
        Fill the not available values according to prediction results
        """
        table = self.db.tables[self.table]
        df = table.df
        for col in df.columns.tolist():
            column = df[col].dropna()
            if df[col].dtype == 'float64' and column.apply(lambda x: x.is_integer()).all():
                df[col] = df[col].astype(pd.Int64Dtype())

        na_indices = torch.nonzero(torch.tensor(pd.isna(df[self.col]))).flatten().numpy()
        valid_indices = torch.nonzero(torch.tensor(pd.notna(df[self.col]))).flatten().numpy()
        pred = pred.cpu()
        encoder = table.feat_encoder[self.col]
        if self.col in table.feature_disc:
            pred = np.vectorize(encoder.inverse.get)(pred)
            if na_indices.shape[0]:
                df[self.col][na_indices] = pred[na_indices]
        elif self.col in table.feature_cont:
            pred = encoder.inverse_transform(pred).squeeze()
            if df[self.col][valid_indices].dtype == 'Int64':
                pred = pd.Series(pred.round(), dtype='int64')
            if na_indices.shape[0]:
                df[self.col][na_indices] = pred[na_indices]
        df.to_csv(f"{path}/{self.table}.csv", index=False)
