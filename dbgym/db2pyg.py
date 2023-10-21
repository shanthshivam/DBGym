"""
db2pyg.py
Module to transform database into graph.
"""

import time
import torch
import numpy as np
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
        self.hetero = HeteroData()
        # Construct a heterogeneous graph using the provided tables and features
        self.output = 0
        self.init_node()
        self.init_edge()

        self.mask = {}
        self.split()
        print("Done initialisation.")

    def init_node(self):
        """
        Construct heterogeneous graph nodes
        """

        hetero = self.hetero

        start = time.time()
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
                    hetero.y = table.feature_disc[self.col].squeeze()
                    self.output = torch.max(hetero.y[table.valid_indices(self.col)]).item() + 1
                else:
                    feat_c.remove(self.col)
                    hetero.y = table.feature_cont[self.col].squeeze()
                    self.output = 1

            # concatenate features
            if feat_d:
                feature = torch.cat([table.feature_disc[col].view(-1, 1) for col in feat_d], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            hetero[name].x_d = feature.to(torch.int32)
            if feat_c:
                feature = torch.cat([table.feature_cont[col].view(-1, 1) for col in feat_c], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            hetero[name].x_c = feature.to(torch.float32)
            hetero[name].num_nodes = len(table.df)

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

        print(f'Node use time: {time.time() - start} s')

    def init_edge(self):
        """
        Construct heterogeneous graph edges
        """

        hetero = self.hetero
        lookup_table = self.lookup_table
        duplicate = self.duplicate

        start = time.time()
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
                if '.' in key:
                    key = key.split('.')[0]

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

                if hetero[name, "to", key].num_edges:
                    index = hetero[name, "to", key].edge_index
                    hetero[name, "to", key].edge_index = torch.cat([index, edge_index], dim=1)
                else:
                    hetero[name, "to", key].edge_index = edge_index
                edge_index = edge_index[[1, 0]]
                if hetero[key, "to", name].num_edges:
                    index = hetero[key, "to", name].edge_index
                    hetero[key, "to", name].edge_index = torch.cat([index, edge_index], dim=1)
                else:
                    hetero[key, "to", name].edge_index = edge_index

        print(f'Edge use time: {time.time() - start} s')

    def split(self, seed=None):
        """
        Split the dataset into training, validation, and test sets.

        Args:
        - seed: The seed to set

        Returns:
        - masks: A dictionary containing masks for hetero graph and homo graph
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
