import networkx as nx

import numpy as np
import pandas as pd

LABEL_X = 'x'
LABEL_Y = 'y'
LABEL_Z = 'z'
LABEL_DX = 'dx'
LABEL_DY = 'dy'
LABEL_DZ = 'dz'

def weight_func_default(data_x, data_y):
    a = data_x['x':'y'].values
    b = data_y['x':'y'].values
    return np.linalg.norm(a - b)

def dist_tuple_default(row, name1='x_x', name2='ph'):
    a = np.array([row.x_prev, row.y_prev])
    b = np.array([row.x_next, row.y_next])
    return np.linalg.norm(a - b)

def nodes_info_default(df_from, G):
    G.add_nodes_from(df_from[['x','y', 'track']].to_dict(orient='index').items())
    return G

def to_pandas_graph_df(single_event_df, weight_func=weight_func_default, dist_tuple=dist_tuple_default, suffx=None, compute_is_true_track=False):
    if suffx is None:
        suffx = ('_prev','_current')
    assert single_event_df.event.nunique() == 1
    my_df = single_event_df.copy()
    my_df['index_old'] = my_df.index
    by_stations = [df for (ind, df) in my_df.groupby('station')]
    cartesian_product = pd.DataFrame()
    for i in range(1, len(by_stations)):
        cartesian_product = cartesian_product.append(pd.merge(by_stations[i - 1], by_stations[i], on='event', suffixes=suffx),
                                                     ignore_index=True, sort=False)
    if compute_is_true_track:
        pid1 = cartesian_product["track" + suffx[0]].values
        pid2 = cartesian_product["track" + suffx[1]].values
        cartesian_product['track'] = ((pid1 == pid2) & (pid1 != -1))
    return cartesian_product

def to_nx_graph(single_event_df, weight_func=weight_func_default, dist_tuple=dist_tuple_default):
    assert single_event_df.event.nunique() == 1
    G = nx.DiGraph()
    edges = []
    G = nodes_info_default(single_event_df, G)
    my_df = single_event_df.copy()
    my_df['index_old'] = my_df.index
    by_stations = [df.rename(columns={'station': 'station_' + str(ind)}) for (ind, df) in
                   my_df.groupby('station')]
    for i in range(1, len(by_stations)):
        cartesian_product = pd.merge(by_stations[i - 1], by_stations[i], on='event', suffixes=('_prev','_next'))
        elems = [(row.index_old_prev, row.index_old_next, dist_tuple(row)) for row in cartesian_product.itertuples()]
        edges.extend(elems)
    elems_self_loop_dropped = [elem for elem in edges if elem[0]!=elem[1]]
    G.add_weighted_edges_from(elems_self_loop_dropped)
    return G

def zip_index(index1, index2):
    return int(np.uint64(np.uint32(index1) << 32 | np.uint32(index2)))

def unzip_index(index_zipped):
    return index_zipped >> 32, (index_zipped & np.uint32(-1))

def get_weight_default(di_graph_old, node_data_i, node_data_j, node_data_l):
    # -1 is the track_id
    a, b, c = np.fromiter(di_graph_old.nodes[node_data_i].values(), dtype=np.float)[:-1],\
              np.fromiter(di_graph_old.nodes[node_data_j].values(), dtype=np.float)[:-1],\
              np.fromiter(di_graph_old.nodes[node_data_l].values(), dtype=np.float)[:-1]
    ba = b - a
    cb = c - b
    norm_ba = np.linalg.norm(ba)
    norm_cb = np.linalg.norm(cb)
    w = np.dot(ba, cb) / ( norm_ba * norm_cb )
    return w / (norm_ba + norm_cb)


# graph should have weights on edges
def to_line_graph(nx_di_graph):
    # iterate over the edges with weights
    new_graph = nx.DiGraph()
    for from_node, current_node in nx_di_graph.edges():
        # from_node is: (u,v,w)
        new_node_index = zip_index(from_node,current_node)
        new_graph.add_node(new_node_index)
        for to_node in nx_di_graph.edges(current_node):
            true_superedge = nx_di_graph.nodes[from_node]['track'] == nx_di_graph.nodes[current_node]['track'] == nx_di_graph.nodes[to_node[1]]['track'] and nx_di_graph.nodes[from_node]['track'] != -1
            new_graph.add_edge(new_node_index, zip_index(to_node[0], to_node[1]), weight=get_weight_default(nx_di_graph, from_node, to_node[0], to_node[1]), true=true_superedge)
    return new_graph

def get_linegraph_superedges_stat(nx_di_graph):
    # iterate over the edges with weights
    true_SE = []
    false_SE = []
    for from_node, current_node in nx_di_graph.edges():
        for to_node in nx_di_graph.edges(current_node):
            true_superedge = nx_di_graph.nodes[from_node]['track'] == nx_di_graph.nodes[current_node]['track'] == nx_di_graph.nodes[to_node[1]]['track'] and nx_di_graph.nodes[from_node]['track'] != -1
            if true_superedge:
                true_SE.append(get_weight_default(nx_di_graph, from_node, to_node[0], to_node[1]))
            else:
                false_SE.append(get_weight_default(nx_di_graph, from_node, to_node[0], to_node[1]))
    return true_SE, false_SE

rename_scheme={
    'x_prev':'x_current',
    'y_prev':'y_current',
    'track_prev':'track_current',
    'index_old_prev':'index_old_current',
    'x_current':'x_next',
    'y_current':'y_next',
    'index_old_current':'index_old_next',
    'track_current':'track_next',
}

# edge row should contain info about x,y,z
def get_weight_pandas_edgegraph_default(prev_edge, next_edge):
    prev_edges = prev_edge[['x_prev', 'y_prev', 'track_prev', 'index_old_prev', 'index_old_current']]
    next_edge = next_edge.rename(columns=rename_scheme)
    next_edges = next_edge[['x_current', 'y_current', 'track_current', 'index_old_current',
                            'x_next', 'y_next', 'track_next', 'index_old_next',]]
    line_graph_edges = pd.merge(prev_edges, next_edges, on='index_old_current', suffixes=('_p','_c'))

    line_graph_edges = line_graph_edges.assign(true_superedge=False)
    index_true_superedge = line_graph_edges[(line_graph_edges.track_prev == line_graph_edges.track_current) &
                                            (line_graph_edges.track_prev == line_graph_edges.track_next) &
                                            (line_graph_edges.track_prev != -1)].index
    line_graph_edges.loc[index_true_superedge, 'true_superedge'] = True

    a = line_graph_edges[['x_prev', 'y_prev']].values
    b = line_graph_edges[['x_current', 'y_current']].values
    c = line_graph_edges[['x_next', 'y_next']].values


    ba = b - a
    cb = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_cb = np.linalg.norm(cb, axis=1)
    dot = np.einsum('ij,ij->i',ba, cb)
    w = dot / (norm_ba * norm_cb)
    line_graph_edges['weight'] = w
    indexation = ['weight', 'index_old_prev',
                   'index_old_current', 'index_old_next', 'true_superedge']
    return line_graph_edges[indexation]


def check_validity_single_event_in_graph(df, check=True):
    if check:
        assert df.event.nunique() == 1

def get_linegraph_stats_from_pandas(pd_graph):
    check_validity_single_event_in_graph(pd_graph)
    by_stations = [df for (ind, df) in pd_graph.groupby('station_prev')]
    weights_true = []
    weights_false = []
    for i in range(1, len(by_stations)):
        prev_edges = by_stations[i - 1]
        next_edges = by_stations[i]
        edge_graph = get_weight_pandas_edgegraph_default(prev_edges, next_edges)
        trues_weights = edge_graph[edge_graph.true_superedge == True][['weight']].values
        false_weights = edge_graph[edge_graph.true_superedge == False][['weight']].values
        weights_true.extend(np.unique(trues_weights))
        weights_false.extend(np.unique(false_weights))

    return weights_true, weights_false

def zip_index_pd(pd_col1_int_vals, pd_col2_int_vals):
    assert pd_col1_int_vals.dtype == np.uint32 and pd_col2_int_vals.dtype == np.uint32
    packed = np.left_shift(pd_col1_int_vals, 32, dtype=np.uint64)
    packed = np.bitwise_or(packed, pd_col2_int_vals.astype(np.uint32), dtype=np.uint64)
    return packed

def get_supernodes_df(one_station_edges, with_station_info, station=-1, STAION_COUNT=5):
    ret = pd.DataFrame()
    x0_y0_z0 = one_station_edges[[LABEL_X + '_prev', LABEL_Y + '_prev', LABEL_Z + '_prev']].values
    x1_y1_z1 = one_station_edges[[LABEL_X + '_current', LABEL_Y +'_current', LABEL_Z +'_current']].values
    dx_dy_dz = x1_y1_z1 - x0_y0_z0

    ret = ret.assign(dx=dx_dy_dz[:, 0], dy=dx_dy_dz[:, 1],
                     x_p=one_station_edges.x_prev.values,
                     x_c=one_station_edges.x_current.values,
                     y_p=one_station_edges.y_prev.values,
                     y_c=one_station_edges.y_current.values,
                     dz=dx_dy_dz[:, 2], z=(station+1)/STAION_COUNT)
    ret = ret.assign(from_ind=one_station_edges[['index_old_prev']].values.astype(np.uint32))
    ret = ret.assign(to_ind=one_station_edges[['index_old_current']].values.astype(np.uint32))
    ret = ret.assign(track=-1)
    ret = ret.set_index(one_station_edges.index)
    index_true_superedge = one_station_edges[(one_station_edges.track_prev == one_station_edges.track_current) &\
                                            (one_station_edges.track_prev != -1)]
    ret.loc[index_true_superedge.index, 'track'] = index_true_superedge.track_prev.values
    # ret = ret.assign(ind=zip_index_pd(one_station_edges[['index_old_prev']].values.astype(np.uint32),
    #                                   one_station_edges[['index_old_current']].values.astype(np.uint32)))
    if with_station_info and station >= 0:
        ret = ret.assign(station=station)
    return ret

def get_edges_from_supernodes(sn_from, sn_to):
    prev_edges = sn_from.rename(columns={'to_ind':'cur_ind'})
    next_edges = sn_to.rename(columns={'from_ind':'cur_ind'})
    prev_edges['edge_index'] = prev_edges.index
    next_edges['edge_index'] = next_edges.index
    line_graph_edges = pd.merge(prev_edges, next_edges, on='cur_ind', suffixes=('_p', '_c'))

    line_graph_edges = line_graph_edges.assign(true_superedge=-1)
    index_true_superedge = line_graph_edges[(line_graph_edges.track_p == line_graph_edges.track_c) &
                                            (line_graph_edges.track_p != -1)]
    line_graph_edges.loc[index_true_superedge.index, 'true_superedge'] = index_true_superedge.track_p.values



    ba = line_graph_edges[['dx_p', 'dy_p']].values
    cb = line_graph_edges[['dx_c', 'dy_c']].values
    # norm_ba = np.linalg.norm(ba, axis=1)
    # norm_cb = np.linalg.norm(cb, axis=1)
    # dot = np.einsum('ij,ij->i', ba, cb)
    w = np.linalg.norm(ba - cb, axis=1)
    line_graph_edges['weight'] = w
    indexation = ['weight', 'true_superedge', 'edge_index_p', 'edge_index_c']
    return line_graph_edges[indexation]

def get_pd_line_graph(pd_edges, with_station_info=True, restriction_func=None, reduce_output=False):
    if reduce_output:
        assert restriction_func is not None
    nodes = pd.DataFrame()
    edges = pd.DataFrame()
    by_stations = [df for (ind, df) in pd_edges.groupby('station_prev')]

    mean_purity = []
    mean_reduce = []
    for i in range(1, len(by_stations)):
        supernodes_from = get_supernodes_df(by_stations[i - 1], with_station_info, i-1)
        supernodes_to = get_supernodes_df(by_stations[i], with_station_info, i)

        if restriction_func:
            new_from = restriction_func(supernodes_from)
            new_to = restriction_func(supernodes_to)
            if not supernodes_to[supernodes_to.track != -1].empty and not new_to.empty:
                mean_purity.append(len(new_from[new_from.track != -1])/len(supernodes_from[supernodes_from.track != -1]))
                mean_purity.append(len(new_to[new_to.track != -1])/len(supernodes_to[supernodes_to.track != -1]))
                mean_reduce.append(len(supernodes_from) / len(new_from))
                mean_reduce.append(len(supernodes_to) / len(new_to))
        if reduce_output:
            supernodes_from = new_from
            supernodes_to = new_to
        nodes = nodes.append(supernodes_from, sort=False)
        nodes = nodes.append(supernodes_to, sort=False)
        superedges = get_edges_from_supernodes(supernodes_from, supernodes_to)
        edges = edges.append(superedges, ignore_index=True, sort=False)

    nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
    return nodes, edges, mean_purity, mean_reduce

def get_reduced_df_graph(pd_graph, hit_event_graph, get_bars_info = True):
    check_validity_single_event_in_graph(pd_graph)
    check_validity_single_event_in_graph(hit_event_graph)
    by_stations = [df for (ind, df) in pd_graph.groupby('station_prev')]
    res_df = pd.DataFrame()
    weights_true = []
    weights_false = []
    for i in range(1, len(by_stations)):
        prev_edges = by_stations[i - 1]
        next_edges = by_stations[i]
        edge_graph = get_weight_pandas_edgegraph_default(prev_edges, next_edges)
        if get_bars_info:
            trues_weights = edge_graph[edge_graph.true_superedge == True][['weight']].values
            false_weights = edge_graph[edge_graph.true_superedge == False][['weight']].values
            weights_true.extend(np.unique(trues_weights))
            weights_false.extend(np.unique(false_weights))
        left_edges = edge_graph[edge_graph.weight > 0.95]
        res_df = res_df.append(hit_event_graph.loc[np.unique(left_edges[['index_old_prev']].values)])
        res_df = res_df.append(hit_event_graph.loc[np.unique(left_edges[['index_old_current']].values)])
        res_df = res_df.append(hit_event_graph.loc[np.unique(left_edges[['index_old_next']].values)])
    # dropping duplicates
    res_df = res_df.loc[~res_df.index.duplicated(keep='first')]
    if get_bars_info:
        return weights_true, weights_false, res_df
    return res_df



def get_weight_stats(nx_di_graph):
    new_test_data = {}
    for ind, edge in enumerate(nx_di_graph.edges(data=True)):
        i, j = unzip_index(edge[0])
        _, l = unzip_index(edge[1])
        new_test_data[ind] = {
            'from_zipped': edge[0],
            'to_zipped': edge[1],
            'i':i, 'j':j, 'l':l, 'weight':edge[2]['weight'], 'true_edge':edge[2]['true']
        }
    stat_df = pd.DataFrame.from_dict(new_test_data, orient='index')
    return stat_df[stat_df.true_edge == True].weight.values, stat_df[stat_df.true_edge == False].weight.values

def run_mbt_graph(pd_node_graph, pd_edge_graph):
    new_weights = 1 / pd_edge_graph.weight.values
    pd_edge_graph = pd_edge_graph.assign(newweight=new_weights, indexx=pd_edge_graph.index)
    nx_gr = nx.from_pandas_edgelist(pd_edge_graph, 'edge_index_p', 'edge_index_c', ['newweight', 'indexx'], create_using=nx.DiGraph())
    br = nx.algorithms.tree.maximum_branching(nx_gr, preserve_attrs=True)
    reduced = nx.to_pandas_edgelist(br)
    ret = pd_edge_graph.loc[reduced.indexx]
    return ret, (len(ret[ret.true_superedge != -1])/len(pd_edge_graph[pd_edge_graph.true_superedge != -1])), \
           (len(pd_edge_graph) / len(ret))



