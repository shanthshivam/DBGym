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
    - db: The DataBase to convert
    - table: The name of target table
    - col: The column you want to predict
    """

    # These functions below are for initialisation.
    def __init__(self, db: DataBase, table: str, col: str):

        self.db = db

        self.len_featint = {}
        self.classbef_num = 0

        self.task = ''
        self.missing_csv = target_csv
        self.missing_col = target_col
        self.missing_FK = None
        self.embeds = []
        self.embedding_hetero = None
        self.embedding_homo = None
        self.homo_mask = None

        self.init_hetero()
        self.init_homo()
        print("Done initialisation.")

    def init_hetero(self):
        """
        Construct a heterogeneous graph using the provided tables and features.

        Args:
            reverse_connect (bool, optional): Whether to create reverse connections. Defaults to False.
            concat_feature (bool, optional): Whether to concatenate features. Defaults to True.

        Returns:
            HeteroData: Constructed heterogeneous graph data.
        """
        self.hetero = HeteroData()
        missing_csv = self.missing_csv
        missing_col = self.missing_col
        task = self.task

        reverse_connect = self.reverse_connect

        data = self.hetero
        hash = {}

        # Count the number of features before the target csv.
        for node_type, typ in self.db.names:
            if node_type == missing_csv:
                break
            self.classbef_num += self.db.tables[node_type].df.shape[0]

        # Firstly, we create nodes in the graph.
        for node_type, typ in self.db.names:
            table = self.db.tables[node_type]

            # Handle special case for missing CSV and column.
            if node_type == missing_csv:
                # Infer the type of the task.
                if missing_col in table.feat_cont:
                    self.task = "regression"
                else:
                    self.task = "classification"

                for col, val in table.df.items():

                    if col == missing_col:
                        if task == "classification":
                            if col.endswith("_id"):
                                # The second condition is used to tell whether missing col is an FK,
                                # which indicates that the task is classification.
                                self.missing_FK = (node_type + "_id", "connect", col)
                                self.hetero_y = torch.tensor(val.values)
                            else:
                                # Map values to natural number.
                                str_val = val.values.astype(str)
                                unique_str = np.unique(str_val)
                                string_to_int = {
                                    string: idx for idx, string in enumerate(unique_str)
                                }
                                int_array = np.array(
                                    [string_to_int[string] for string in str_val]
                                )
                                self.hetero_y = torch.tensor(int_array)
                        elif task == "regression":
                            self.hetero_y = table.feat_cont[col].squeeze()

                        if self.task == "classification":
                            # Find indices of non-max values,
                            # since in db.py nan is treated as an extra kind mapped to the maximum integer.
                        
                            # note from zzz: please first find out if there is any nan value
                            if self.db.tables[self.missing_csv].df[self.missing_col].isna().sum() != 0:
                                # Find the index of the maximum value
                                max_index = torch.argmax(self.hetero_y)

                                # Find indices of non-maximum values
                                non_max_indices = torch.nonzero(self.hetero_y != self.hetero_y[max_index]).flatten()
                            else:
                                non_max_indices = torch.nonzero(self.hetero_y == self.hetero_y).flatten()
                        else:
                            # Find indices of non-NaN values
                            non_max_indices = torch.nonzero(~torch.isnan(self.hetero_y)).flatten()

                        self.valid_label_index = non_max_indices

            # Set node features.

            # Collect non-label float and integer columns from the table.
            # By this mean, we manage to place feat_float before feat_int.
            nonlabel_col_float = [
                col
                for col in table.feat_cont
                if col != missing_col or node_type != missing_csv
            ]

            nonlabel_col_int = [
                col
                for col in table.feat_disc
                if col != missing_col or node_type != missing_csv
            ]

            # Concatenate float features if available.
            if len(nonlabel_col_float) > 0:
                # All NaN in feat_float is substituted by non-NaN mean.
                # However, it seems that NaN is coped in db.py
                for col in nonlabel_col_float:
                    # print(table.feat_cont[col])
                    data_np = table.feat_cont[col].numpy()
                    mean_non_nan = np.nanmean(data_np)

                    # Replace NaN values with the mean
                    table.feat_cont[col][
                        table.feat_cont[col] != table.feat_cont[col]
                    ] = torch.tensor(mean_non_nan)

                hetero_feature_float = torch.cat(
                    [
                        table.feat_cont[col].clone().detach().view(-1, 1)
                        for col in nonlabel_col_float
                    ],
                    dim=1,
                )

            # Concatenate integer features if available.
            if len(nonlabel_col_int) > 0:
                hetero_feature_int = torch.cat(
                    [
                        table.feat_disc[col].clone().detach().view(-1, 1)
                        for col in nonlabel_col_int
                    ],
                    dim=1,
                )
                self.len_featint[node_type] = hetero_feature_int.shape[1]
                # if node_type == missing_csv and self.task == "classification":
                #     self.len_featint[node_type] = self.len_featint[node_type] - 1

            # Concatenate float and integer features if both are available.
            if len(nonlabel_col_float) > 0 and len(nonlabel_col_int) > 0:
                data[node_type + "_id"].x = torch.cat(
                    (hetero_feature_float, hetero_feature_int), dim=1
                )
            elif len(nonlabel_col_float) > 0:
                data[node_type + "_id"].x = hetero_feature_float
            elif len(nonlabel_col_int) > 0:
                data[node_type + "_id"].x = hetero_feature_int
            else: # This means that this node_type has no features, but they exist!
                # Lugar fills its feature with a single 0.
                data[node_type + "_id"].x = torch.zeros((len(table.df[node_type + "_id"]),1))
            
            # Set up a hash map for mapping node IDs to indices.
            hash_key = self.db.tables[node_type].df[node_type + "_id"].to_list()
            hash_value = [i for i in range(0, len(hash_key))]
            hash[node_type + "_id"] = dict(zip(hash_key, hash_value))

        if self.missing_FK:
            # We still need to map hetero_y (one end of edges).
            for i in range(0, self.hetero_y.shape[0]):
                if not torch.isnan(self.hetero_y[i]):
                    self.hetero_y[i] = hash[missing_col][self.hetero_y[i].item()]

        # Secondly, we create edges.
        for node_type, typ in self.db.names:
            keys = self.db.tables[node_type].get_keys()
            if len(keys) == 1:
                # It has no foreign key.
                continue
            edge_matrix = self.db.tables[node_type].df[keys].values

            # Map the original ids in the edge_matrix to indices.
            for i in range(0, edge_matrix.shape[0]):
                for j in range(0, edge_matrix.shape[1]):
                    other_node_type = keys[j]

                    # Remove numbers from the end using regular expression
                    other_node_type = re.sub(r'\d+$', '', other_node_type)
                    # print(other_node_type)

                    if edge_matrix[i][j] in hash[other_node_type].keys():
                        # Assure this node exist.
                        if (not isinstance(edge_matrix[i][j], (float, int, np.floating, np.integer))) or (not np.isnan(edge_matrix[i][j])):
                            edge_matrix[i][j] = hash[other_node_type][edge_matrix[i][j]]

            edge_matrix = edge_matrix.astype(float)
            if np.issubdtype(edge_matrix.dtype, np.floating):
                if np.isnan(edge_matrix).any():
                    edge_matrix = edge_matrix[~np.isnan(edge_matrix).any(axis=1)]
            edge_matrix = torch.tensor(edge_matrix.astype(np.int64))

            for i in range(1, len(keys)):
                # Take out the edge row
                index1, index2 = np.ix_(
                    [i for i in range(0, edge_matrix.shape[0])], [0, i]
                )
                edge_index = edge_matrix[index1, index2]
                # When foreign key is missing, we do not know where the edge goes.
                # So we need to pick NaN out of the edge_index.
                # Assume that a missing value shall be represented as torch.nan
                non_missing_index = ~torch.isnan(edge_index).any(axis=1)
                non_missing_index = non_missing_index.type(torch.bool)
                edge_index = edge_index[non_missing_index].T

                # We make it bi-directional edges.

                other_node_type = keys[i]

                # Remove numbers from the end using regular expression
                other_node_type = re.sub(r'\d+$', '', other_node_type)
                
                edge_storage = data[node_type + "_id", "connect", other_node_type]
                if hasattr(edge_storage, "edge_index"):
                    # Cope with multiple key column pointing to one type.
                    edge_storage.edge_index = torch.cat(
                        (edge_storage.edge_index, edge_index), dim=1
                    )
                else:
                    edge_storage.edge_index = edge_index
                
                edge_index_reverse = np.array([edge_index[1], edge_index[0]])  # Swap rows of indexes.
                edge_index_reverse = torch.tensor(edge_index_reverse)
                edge_name = "reverse-connect" if reverse_connect else "connect"
                edge_storage = data[other_node_type, edge_name, node_type + "_id"]
                if hasattr(edge_storage, "edge_index"):
                    edge_storage.edge_index = torch.cat(
                        (edge_storage.edge_index, edge_index_reverse), dim=1
                    )
                else:
                    edge_storage.edge_index = edge_index_reverse

    def init_homo(self):
        """
        Convert the constructed heterogeneous graph to a homogeneous graph.

        Returns:
            None
        """
        self.homo = self.hetero.to_homogeneous()
        return

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
                len_float = self.hetero[node_type + "_id"].x.shape[1] - len_int
            elif self.task == "regression":
                if node_type not in self.len_featint.keys():
                    len_int = 0
                else:
                    len_int = self.len_featint[node_type]
                # len_float = self.hetero[node_type+'_id'].x.shape[1]-len_int-1
                # I don't see why Yang add a -1 here.
                len_float = self.hetero[node_type + "_id"].x.shape[1] - len_int

            feature_float = self.hetero[node_type + "_id"].x[:, :len_float]
            feature_int = self.hetero[node_type + "_id"].x[:, len_float : len_float + len_int]

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
            self.embedding_hetero[node_type + "_id"] = feature_embed

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
            max_int = self.homo_graph.node_type.max().item() + 1
            embed_dict["node_type"] = torch.nn.Embedding(int(max_int), int(dimen_nodetype))
            self.embeds.append(embed_dict["node_type"])
            embed_dict = torch.nn.ModuleDict(embed_dict)
            embed_node = embed_dict["node_type"](self.homo_graph.node_type.long())
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

    # These functions below support scaling on the target(y).
    def normalize_y(self, type: str):
        if type == "Homo":
            tensor = self.homo_y
        elif type == "Hetero":
            tensor = self.hetero_y
        else:
            raise ValueError("type string should be either Homo or Hetero")
        # Filter out NaN and zero values
        filtered_values = tensor[~torch.isnan(tensor) & (tensor != 0)]

        # Calculate mean and standard deviation
        mean = torch.mean(filtered_values).item()
        std = torch.std(filtered_values).item()
        return (tensor - mean) / std

    def inverse_normalize_y(self, ans_tensor, type: str):
        if type == "Homo":
            tensor = self.homo_y
        elif type == "Hetero":
            tensor = self.hetero_y
        else:
            raise ValueError("type string should be either Homo or Hetero")
        # Filter out NaN and zero values
        filtered_values = tensor[~torch.isnan(tensor) & (tensor != 0)]

        # Calculate mean and standard deviation
        mean = torch.mean(filtered_values).item()
        std = torch.std(filtered_values).item()
        return ans_tensor * std + mean

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

        total_samples = self.valid_label_index.shape[0]
            
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
            self.hetero_mask[split_name] = torch.tensor([self.valid_label_index[x].item() for x in indices[start_idx:end_idx]])
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
