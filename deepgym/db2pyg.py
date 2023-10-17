"""
db2pyg.py
Module to transform database into graph.
"""

import re
from typing import Dict, List
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

        self.hetero = self.init_hetero()
        self.homo = self.init_homo()
        print("Done initialisation.")

    def init_hetero(self):
        """
        Construct a heterogeneous graph using the provided tables and features
        """

        hetero = HeteroData()
        lookup_table = {}

        # 1 create nodes in the graph
        for name, table in self.db.tables.items():
            feat_cols = list(table.feature.keys())
            if name == self.table:
                feat_cols.remove(self.col)
                self.hetero_y = table.feature[self.col].squeeze()

            # concatenate features
            if feat_cols:
                feature = torch.cat([table.feature[col].view(-1, 1) for col in feat_cols], dim=1)
            else:
                feature = torch.zeros((len(table.df), 1))
            hetero[name].x = feature.to(torch.float32)

            # set up a look up table
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
        return self.hetero.to_homogeneous()

    # These functions below are used to obtain embeddings.
    def Embedding_hetero(self, int_dimension=0):
        """
        Perform embedding on the heterogeneous node features.

        Args:
            int_dimension (int): The dimension of the integer embeddings. If 0, automatically determined based on max value.

        Returns:
            dict: A dictionary containing the embedded features for each heterogeneous node type.
        """

        self.embedding_hetero = {}
        # print(self.len_featint)
        # print(self.db.names)
        for node_type, typ in self.db.names:
            # print(graph[node_type+'_id'].x.shape[1])
            
            if self.task == "classification":
                if node_type not in self.len_featint.keys():
                    len_int = 0
                else:
                    len_int = self.len_featint[node_type]
                len_float = self.hetero[node_type].x.shape[1] - len_int
            elif self.task == "regression":
                if node_type not in self.len_featint.keys():
                    len_int = 0
                else:
                    len_int = self.len_featint[node_type]
                # len_float = self.hetero[node_type+'_id'].x.shape[1]-len_int-1
                # I don't see why Yang add a -1 here.
                len_float = self.hetero[node_type].x.shape[1] - len_int

            feature_float = self.hetero[node_type].x[:, :len_float]
            feature_int = self.hetero[node_type].x[:, len_float : len_float + len_int]

            if len_int > 0:
                embed_dict = {}
                for col in range(len_int):
                    # print(feature_int[:,col].max())
                    max_int = feature_int[:, col].max().item() + 1
                    inti_dimension = int_dimension if int_dimension else max_int
                    embed_dict[str(col)] = torch.nn.Embedding(
                        int(inti_dimension), int(len_int)
                    )
                    self.embeds.append(embed_dict[str(col)])
                embed_dict = torch.nn.ModuleDict(embed_dict)

                feature_embed = 0.0
                for col in range(len_int):
                    feature_embed += embed_dict[str(col)](feature_int[:, col].long())

                if len_float > 0:
                    feature_embed = torch.cat((feature_float, feature_embed), dim=1)
                # break
            else:
                feature_embed = feature_float
            self.embedding_hetero[node_type] = feature_embed

        return self.embedding_hetero

    def Embedding_homo(self, embed_nodetype=False, dimen_nodetype=0):
        """
        Embed heterogeneous node features into a homogeneous embedding tensor.

        Args:
            embed_nodetype (bool, optional): Whether to embed node types. Defaults to False.
            dimen_nodetype (int, optional): Dimensionality of node type embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Homogeneous embedding tensor containing concatenated heterogeneous embeddings.
        """
        # Make sure self.embedding_hetero exists.
        if self.embedding_hetero is None:
            self.Embedding_hetero()
            
        num_row, num_col = 0, 0
        row_starts, col_starts = [], []  # Using lists instead of dictionaries

        # Treat node type as feat_int, we create additional embeddings for node_type
        if embed_nodetype:
            embed_dict = {}
            max_int = self.homo.node_type.max().item() + 1
            embed_dict["node_type"] = torch.nn.Embedding(int(max_int), int(dimen_nodetype))
            self.embeds.append(embed_dict["node_type"])
            embed_dict = torch.nn.ModuleDict(embed_dict)
            embed_node = embed_dict["node_type"](self.homo.node_type.long())
            self.embedding_hetero["homo_node_type"] = embed_node

        for key in self.embedding_hetero.keys():
            # Record the starting row and column indices of each block
            row_starts.append(num_row)
            col_starts.append(num_col)

            # Accumulate the number of rows and columns
            num_row += self.embedding_hetero[key].shape[0]
            num_col += self.embedding_hetero[key].shape[1]

        # Record the ending row and column indices of the last block
        row_starts.append(num_row)
        col_starts.append(num_col)

        # Initialize the homogeneous result tensor
        self.embedding_homo = torch.zeros((num_row, num_col))

        for block_idx, key in enumerate(self.embedding_hetero.keys()):
            # Copy the content of each block to the corresponding position in the homogeneous result tensor
            row_start, row_end = row_starts[block_idx], row_starts[block_idx + 1]
            col_start, col_end = col_starts[block_idx], col_starts[block_idx + 1]
            self.embedding_homo[
                row_start:row_end, col_start:col_end
            ] = self.embedding_hetero[key]

        return self.embedding_homo

    def Infer(self):
        index = 0
        embedding_hetero = {}
        for node_type, typ in self.db.names:
            len_int = self.len_featint[node_type]
            len_float = self.hetero[node_type + "_id"].x.shape[1] - len_int
            feature_float = self.hetero[node_type + "_id"].x[:, :len_float]
            feature_int = self.hetero[node_type + "_id"].x[:, len_float : len_float + len_int]
            if len_int > 0:
                feature_embed = 0.0
                for col in range(len_int):
                    feature_embed += self.embeds[index + col](feature_int[:, col].long())
                index += len_int
                if len_float > 0:
                    feature_embed = torch.cat((feature_float, feature_embed), dim=1)
            else:
                feature_embed = feature_float
            embedding_hetero[node_type + "_id"] = feature_embed

        num_row, num_col = 0, 0
        row_starts, col_starts = [], []  # Using lists instead of dictionaries
        for key in embedding_hetero.keys():
            row_starts.append(num_row)
            col_starts.append(num_col)
            num_row += embedding_hetero[key].shape[0]
            num_col += embedding_hetero[key].shape[1]

        # Record the ending row and column indices of the last block
        row_starts.append(num_row)
        col_starts.append(num_col)

        # Initialize the homogeneous result tensor
        embedding_homo = torch.zeros((num_row, num_col))

        for block_idx, key in enumerate(embedding_hetero.keys()):
            # Copy the content of each block to the corresponding position in the homogeneous result tensor
            row_start, row_end = row_starts[block_idx], row_starts[block_idx + 1]
            col_start, col_end = col_starts[block_idx], col_starts[block_idx + 1]
            embedding_homo[
                row_start:row_end, col_start:col_end
            ] = embedding_hetero[key]

        return embedding_homo

    def split(
        self,
        split_ratio: List = [0.8, 0.1, 0.1],
        name: List = ["train", "val", "test"],
        type="Hetero",
        order=False,
        seed=0
    ):
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split_ratio (List, optional): List of split ratios for train, validation, and test sets.
                                        Defaults to [0.8, 0.1, 0.1].
            name (List, optional): List of names for train, validation, and test sets.
                                Defaults to ['train', 'val', 'test'].
            type (str, optional): Type of graph, either 'Hetero' or 'Homo'. Defaults to 'Hetero'.

        Returns:
            dict: A dictionary containing masks for train, validation, and test sets.

        Raises:
            ValueError: If the input type is not 'Hetero' or 'Homo'.
            ValueError: If the sum of split ratios exceeds the total number of samples.
        """

        if type not in ("Hetero", "Homo"):
            raise ValueError("Input string must be 'Hetero' or 'Homo'")

        # By this way we can assure reproducibility.
        if seed:
            torch.manual_seed(seed)

        valid_indices = self.db.tables[self.table].valid_indices(self.col)
        total_samples = valid_indices.shape[0]
            
        split_sizes = [int(ratio * total_samples) for ratio in split_ratio]

        if 0 in split_sizes:
            print("Warning: There's one ratio too small that its partition would be empty.")

        if sum(split_sizes) > total_samples:
            raise ValueError("Sum of split ratios exceeds total number of samples.")

        if order:
            indices = torch.tensor([i for i in range(0, total_samples)])
        else:
            indices = torch.randperm(total_samples)

        self.hetero_mask = {}
        start_idx = 0

        # Split the dataset into train, validation, and test sets
        for i, split_size in enumerate(split_sizes):
            end_idx = start_idx + split_size
            split_name = name[i]
            self.hetero_mask[split_name] = torch.tensor([valid_indices[x].item() for x in indices[start_idx:end_idx]])
            start_idx = end_idx

        if type == "Hetero":
            return self.hetero_mask

        self.homo_mask = {}

        # Map the splits to the corresponding classes in a homogeneous graph
        for i, split_size in enumerate(split_sizes):
            split_name = name[i]
            self.homo_mask[split_name] = (
                self.hetero_mask[split_name] + self.classbef_num
            )

        self.homo_y = torch.zeros(len(self.homo.node_type))
        self.homo_y[
            self.classbef_num : self.classbef_num + len(self.hetero_y)
        ] = self.hetero_y

        return self.homo_mask
