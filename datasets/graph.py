"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y.
"""

from collections import namedtuple

import numpy as np

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])

def graph_to_sparse_new(graph):
    Ri_rows_full, Ri_cols_full   = np.where(graph.Ri > 0.5)
    Ri_rows_small, Ri_cols_small = np.where(graph.Ri == 0.5)
    Ro_rows_full, Ro_cols_full   = np.where(graph.Ro > 0.5)
    Ro_rows_small, Ro_cols_small = np.where(graph.Ro == 0.5)
    return dict(X=graph.X, y=graph.y,
                Ri_rows_full = Ri_rows_full,
                Ri_rows_small =Ri_rows_small,
                Ro_rows_full = Ro_rows_full,
                Ro_rows_small =Ro_rows_small,
                Ri_cols_full = Ri_cols_full,
                Ri_cols_small =Ri_cols_small,
                Ro_cols_full = Ro_cols_full,
                Ro_cols_small =Ro_cols_small)

def sparse_to_graph_new(X,
                    Ri_rows_full,
                    Ri_rows_small,
                    Ro_rows_full,
                    Ro_rows_small,
                    Ri_cols_full,
                    Ri_cols_small,
                    Ro_cols_full,
                    Ro_cols_small
                    ,y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], y.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows_small, Ri_cols_small] = 0.5
    Ri[Ri_rows_full, Ri_cols_full] = 1
    Ro[Ro_rows_small, Ro_cols_small] = 0.5
    Ro[Ro_rows_full, Ro_cols_full] = 1
    return Graph(X, Ri, Ro, y)

def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols)

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def save_graphs_new(graphs_filenames):
    for graph, filename in graphs_filenames:
        if graph is None:
            continue
        save_graph(graph, filename)

def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))

def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]
