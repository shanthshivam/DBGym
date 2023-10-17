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

        self.place = 0
        self.mask = {}
        self.hetero = self.init_hetero()
        self.homo = self.init_homo()
        print("Done initialisation.")

    def init_hetero(self):
        """
        Construct a heterogeneous graph using the provided tables and features
        """

        hetero = HeteroData()
        lookup_table = {}
        place = 0

        # 1 create nodes in the graph
        for name, table in self.db.tables.items():
            feat_cols = list(table.feature.keys())
            if name == self.table:
                feat_cols.remove(self.col)
                hetero[name].y = table.feature[self.col].squeeze()
                self.place = place
            place += len(table.df)

            # concatenate features
            if feat_cols:
                feature = torch.cat([table.feature[col].view(-1, 1) for col in feat_cols], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            hetero[name].x = feature.to(torch.float32)

            # set up look up table
            lut_key = table.df[name + "_id"].to_list()
            lut_value = list(range(0, len(lut_key)))
            lookup_table[name] = dict(zip(lut_key, lut_value))

        # 2 create edges in the graph
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

        return hetero

    def init_homo(self):
        """
        Convert the constructed heterogeneous graph to a homogeneous graph.
        """

        hetero = self.hetero
        homo = hetero.to_homogeneous()

        rows = homo.node_type.shape[0]
        cols = 0
        for node in hetero.node_types:
            cols += hetero[node].x.shape[1]
        x = torch.zeros((rows, cols))
        sr = 0
        sc = 0
        for node in hetero.node_types:
            shape = hetero[node].x.shape
            x[sr: sr + shape[0], sc: sc + shape[1]] = hetero[node].x
            sr += shape[0]
            sc += shape[1]

        y = torch.zeros(rows)
        hetero_y = hetero[self.table].y
        y[self.place: self.place + hetero_y.shape[0]] = hetero_y

        homo.x = x
        homo.y = y
        return homo

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
        print(sizes)
        indices = torch.randperm(sizes[3])

        # split the dataset into training, validation, and test sets
        hetero_mask = {}
        homo_mask = {}
        for i, name in enumerate(names):
            hetero_mask[name] = valid_indices[indices[sizes[i]: sizes[i + 1]]]
            homo_mask[name] = hetero_mask[name] + self.place

        self.mask['hetero'] = hetero_mask
        self.mask['homo'] = homo_mask

        return self.mask
