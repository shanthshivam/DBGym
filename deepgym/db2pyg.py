"""
db2pyg.py
Module to transform database into graph.
"""

import re
import torch
import numpy as np
from deepgym.db import DataBase
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
        self.hetero = HeteroData()
        # Construct a heterogeneous graph using the provided tables and features
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
                else:
                    feat_c.remove(self.col)
                    hetero.y = table.feature_cont[self.col].squeeze()

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
            keys = table.df[name + "_id"].to_list()
            values = list(range(0, len(keys)))
            self.lookup_table[name] = dict(zip(keys, values))

        # 2 create edges in the graph

        return hetero

    def init_edge(self):
        """
        Construct heterogeneous graph edges
        """

        hetero = self.hetero
        lookup_table = self.lookup_table

        for name, table in self.db.tables.items():
            keys = table.get_keys()
            if len(keys) == 1:
                continue
            edge_matrix = self.db.tables[name].df[keys].values.T

            for i, key in enumerate(keys):
                key = re.sub(r'\d+$', '', key)[:-3]
                edge_matrix[i] = np.vectorize(lookup_table[key].get)(edge_matrix[i])
                if i == 0:
                    continue
                edge_index = edge_matrix[[0, i]]
                edge_index = edge_index[:, np.all(edge_index != None, axis=0)]
                edge_index = torch.from_numpy(edge_index)
                hetero[name, "to", key].edge_index = edge_index
                hetero[key, "to", name].edge_index = edge_index[[1, 0]]

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
