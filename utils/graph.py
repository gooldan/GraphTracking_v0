import networkx as nx
import numpy as np
import pandas as pd
def weight_func_default(data_x, data_y):
    a = data_x['x':'y'].values
    b = data_y['x':'y'].values
    return np.linalg.norm(a - b)

def dist_tuple_default(row, name1='x_x', name2='ph'):
    a = np.array([row.x_prev, row.y_prev])
    b = np.array([row.x_next, row.y_next])
    return np.linalg.norm(a - b)

def to_nx_graph(single_event_df, weight_func=weight_func_default, dist_tuple=dist_tuple_default):
    assert single_event_df.event.nunique() == 1
    grouped = single_event_df.groupby('track')
    G = nx.DiGraph()
    edges = []
    single_event_df['index_old'] = single_event_df.index
    by_stations = [df.rename(columns={'station': 'station_' + str(ind)}) for (ind, df) in
                   single_event_df.groupby('station')]
    for i in range(1, len(by_stations)):
        cartesian_product = pd.merge(by_stations[i - 1], by_stations[i], on='event', suffixes=('_prev','_next'))
        elems = [(row.index_old_prev, row.index_old_next, dist_tuple(row)) for row in cartesian_product.itertuples()]
        edges.extend(elems)
    elems_self_loop_dropped = [elem for elem in edges if elem[0]!=elem[1]]
    G.add_weighted_edges_from(elems_self_loop_dropped)
    return G

def zip_index(index1, index2):
    assert index1 >= 0 and index2 >= 0 and index1 < int(np.uint32(-1)) and index2 < int(np.uint32(-1))
    return int(np.uint64(np.uint32(index1) << 32 | np.uint32(index2)))

def unzip_index(index_zipped):
    return index_zipped >> 32, (index_zipped & np.uint32(-1))

# graph should have weights on edges
def to_line_graph(nx_di_graph):
    # iterate over the edges with weights
    new_graph = nx.DiGraph()
    for from_node in nx_di_graph.edges(data=True):
        # from_node is: (u,v,w)
        new_graph.add_node(from_node)
        for to_node in  nx_di_graph.edges(from_node[1]):
            new_graph.add_edge(from_node, to_node)