"""
PyTorch specification for the hit graph dataset.
"""

# System imports
import os
import logging
import pandas as pd

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# Local imports
from datasets.graph import load_graph

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None, preproc_df_path=None):
        self.input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
                     if f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)
        self.is_concrete_files = False
        self.df = None
        if preproc_df_path is not None:
            self.preproc_files(preproc_df_path)
            self.is_concrete_files = True
            pass

    def preproc_files(self, df_path):
        self.df = pd.read_csv(df_path)
        self.df = self.df.astype({'event': 'int64'})
        filenames = 'graph_' + self.df['event'].astype('str').values + '.npz'
        self.df = self.df.assign(filenames = filenames)
        self.df = self.df[(self.df.edge_count < 1e4) & (self.df.node_count < 1e4) & (self.df.edge_factor > 0)]
        self.n_train = int(len(self.df) * 0.8)
        self.n_valid = int(len(self.df) * 0.2)
        self.df = self.df[:(self.n_train + self.n_valid)]
        print(self.df)
        pass

    def __getitem__(self, index):
        if self.is_concrete_files:
            return load_graph(os.path.join(self.input_dir, self.df.iloc[index][['filenames']].values[0]))
        return load_graph(self.filenames[index])

    def __len__(self):
        if self.is_concrete_files:
            return len(self.df)
        return len(self.filenames)

def get_datasets(input_dir, n_train, n_valid, preproc_df_path):
    data = HitGraphDataset(input_dir, n_train + n_valid, preproc_df_path)
    # Split into train and validation
    if data.is_concrete_files:
        train_data, valid_data = random_split(data, [data.n_train, data.n_valid])
    else:
        train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data

def collate_fn(graphs):
    """
    Collate function for building mini-batches from a list of hit-graphs.
    This function should be passed to the pytorch DataLoader.
    It will stack the hit graph matrices sized according to the maximum
    sizes in the batch and padded with zeros.

    This implementation could probably be optimized further.
    """
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
        batch_target = torch.from_numpy(g.y[None]).float()
        return batch_inputs, batch_target

    # Get the matrix sizes in this batch
    n_features = graphs[0].X.shape[1]
    n_nodes = np.array([g.X.shape[0] for g in graphs])
    n_edges = np.array([g.y.shape[0] for g in graphs])
    max_nodes = n_nodes.max()
    max_edges = n_edges.max()

    # Allocate the tensors for this batch
    batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    batch_Ri = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_Ro = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_y = np.zeros((batch_size, max_edges), dtype=np.float32)

    # Loop over samples and fill the tensors
    for i, g in enumerate(graphs):
        batch_X[i, :n_nodes[i]] = g.X
        batch_Ri[i, :n_nodes[i], :n_edges[i]] = g.Ri
        batch_Ro[i, :n_nodes[i], :n_edges[i]] = g.Ro
        batch_y[i, :n_edges[i]] = g.y

    batch_inputs = [torch.from_numpy(bm) for bm in [batch_X, batch_Ri, batch_Ro]]
    batch_target = torch.from_numpy(batch_y)
    return batch_inputs, batch_target
