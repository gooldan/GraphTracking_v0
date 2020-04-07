from utils.old_visualizer import Visualizer
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df, \
    calc_purity_reduce_factor, apply_edge_restriction, \
        apply_node_restriction, normalize_convert_to_r_phi_z
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph, run_mbt_graph, calc_dphi


import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import pandas as pd

def get_like_hitgraph_from_linegraph(df):
    df = df.rename(columns={'dx':'x', 'dy':'y', 'dz':'z'})
    return df



def evaluate_mbt_algorithm(event):
    G = to_pandas_graph_df(event, suffx=['_p', '_c'], compute_is_true_track=True)
    G['dz'] = G.z_c.values - G.z_p.values
    G['dr'] = G.r_c.values - G.r_p.values
    G['dphi'] = calc_dphi(G.phi_p.values, G.phi_c.values) / np.pi
    G['weight'] = 1. / np.linalg.norm(G[['dz', 'dphi']].values, axis=1)
    G['edge_index'] = G.index
    nx_G = nx.from_pandas_edgelist(G, 'index_old_p', 'index_old_c', ['weight', 'edge_index'], create_using=nx.DiGraph())
    br = nx.algorithms.tree.maximum_branching(nx_G, preserve_attrs=True)
    reduced = nx.to_pandas_edgelist(br)
    reduced_G = G.loc[reduced.edge_index].copy().reset_index(drop=True)
    vis = Visualizer(event, cfg['visualize'], title='BES EVENT', random_seed=14)
    vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    vis.draw(show=False)
    vis1 = Visualizer(event, cfg['visualize'], title='EVENT GRAPH', random_seed=14)
    vis1.init_draw(draw_all_hits=False, draw_all_tracks_from_df=False)
    ax = vis1.draw_2d(None)
    edges_true = reduced_G[reduced_G.track]
    edges_false = reduced_G[reduced_G.track != True]

    def drop_postfix(df, postfix):
        return df.rename(lambda name: name[:name.index(postfix) if postfix in name else len(name)], axis='columns')

    n_from_true = drop_postfix(edges_true[['r_p', 'phi_p', 'z_p', 'track_p']], '_p')
    n_from_false = drop_postfix(edges_false[['r_p', 'phi_p', 'z_p', 'track_p']], '_p')
    n_to_true = drop_postfix(edges_true[['r_c', 'phi_c', 'z_c', 'track_c']], '_c')
    n_to_false = drop_postfix(edges_false[['r_c', 'phi_c', 'z_c', 'track_c']], '_c')
    vis1.draw_edges_from_nodes_2d(ax, n_from_false, n_to_false,
                                  color=[0.2, 0.2, 0.2, 0.5], pnt_color=[0.2, 0.2, 0.2, 0.5],
                                  z_line=3, z_dot=5, line_width=2)
    vis1.draw_edges_from_nodes_2d(ax, n_from_true, n_to_true, 'orange',
                                  pnt_color=[0.2, 0.8, 0.2, 1.0], z_line=6,
                                  z_dot=10, line_width=2)
    plt.show()
    print(reduced)
    exit()


if __name__ == '__main__':
    # visualize with the cylindrical and normalized coordinates in 2D
    #reader = ConfigReader("configs/cgem_init_config_2d_normalized_cylindrical.yaml")

    # visualize without any modifications in interactive 3D scene
    reader = ConfigReader("configs/cgem_init_config_3D_visualize.yaml")
    cfg = reader.cfg

    df = parse_df(cfg['df'])
    df = normalize_convert_to_r_phi_z(df, True)
    events_df = get_events_df(cfg['df']['take'], df, preserve_fakes=True)

    # idx = 0
    # max = 0
    # for id, event in events_df.groupby('event'):
    #     if len(event) > max:
    #         max = len(event)
    #         idx = id
    #
    # print(idx, max)
    # exit()
    res_false = []
    res_true = []
    draw_bars = True
    lg_nodes = pd.DataFrame()
    lg_edges = pd.DataFrame()
    mean_purity = []
    mean_reduce = []
    mean_edge_reduce = []
    mean_edge_purity = []
    def restrict_func(df):
        return apply_node_restriction(df, [-50, 50], [-50, 50])

    for id, event in events_df.groupby('event'):
        vis = Visualizer(event, cfg['visualize'], title='BES EVENT', random_seed=14)
        vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
        vis.draw(show=True)
        exit()

        # G = to_pandas_graph_df(event)
        # lg_nodes_t, lg_edges_t, mean_purity_t, mean_reduce_t = get_pd_line_graph(G,
        #                                                                          with_station_info=True,
        #                                                                          restriction_func=restrict_func,
        #                                                                          reduce_output=True)
        #
        # like_original_df = get_like_hitgraph_from_linegraph(lg_nodes_t)
        # edges_filtered = apply_edge_restriction(lg_edges_t, 15)
        #
        # #
        # true_edg = lg_edges_t[lg_edges_t.true_superedge != -1]
        # if len(true_edg[true_edg.weight > 15]):
        #     print("FOUND IT!!!", id)
        #     print(true_edg)
        #     print(event)
        #     exit()
        # #
        # # print(len(lg_edges_t))
        # #         # print(len(edges_filtered))
        #
        # #gg, p, m = run_mbt_graph(lg_nodes_t, edges_filtered)
        # p, m = calc_purity_reduce_factor(lg_edges_t, edges_filtered)
        # mean_edge_reduce.append(m)
        # mean_edge_purity.append(p)
        # #new_nodes = apply_node_restriction(lg_nodes_t, [-0.2, 0.2], [-10, 10])
        # mean_reduce.extend(mean_reduce_t)
        # mean_purity.extend(mean_purity_t)
        # lg_nodes = lg_nodes.append(lg_nodes_t, ignore_index=True, sort=False)
        # lg_edges = lg_edges.append(lg_edges_t, ignore_index=True, sort=False)
        #
        # # # vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
        # # # vis.init_draw(draw_all_hits=False, draw_all_tracks_from_df=False)
        # # # vis.add_edges_data(gg)
        # # # vis.draw(show=True)
        # # print("Purity:", p)
        # # print("Reduce:", m)
        # print(id)

    # # edges_filtered = apply_edge_restriction(lg_edges, 0.09)
    # # purity, reduce_factor = calc_purity_reduce_factor(lg_edges, edges_filtered)
    # # print('edges -- purity:', purity)
    # # print('edges -- reduce_factor:', reduce_factor)
    # print("nodes -- reduce:", np.mean(mean_reduce))
    # print("nodes -- purity:", np.mean(mean_purity))
    # print("edges -- reduce:", np.mean(mean_edge_reduce))
    # print("edges -- purity:", np.mean(mean_edge_purity))
    # # vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
    # #     # vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    # #     # vis.draw(show=False)
    # #
    # for id, event in events_df.groupby('event'):
    #      G = to_pandas_graph_df(event)
    #      st_true, st_false, res_ev_df = get_reduced_df_graph(G, event)
    #      res_true.extend(st_true)
    #      res_false.extend(st_false)
    #      print(event.event.values[0])
    #      vis = Visualizer(res_ev_df, cfg['visualize'])
    #      vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #      vis.draw(show=False)
    #
    #      vis = Visualizer(event, cfg['visualize'])
    #      vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #      vis.draw(show=True)
    # exit()
    # plt.figure(figsize=(12, 6))
    # res_true = lg_edges[lg_edges.true_superedge != -1][['weight']].values
    # res_false = lg_edges[lg_edges.true_superedge == -1][['weight']].values
    # plt.subplot(131)
    # binning = dict(bins=300) #, range=(-0.05, 5.1)
    # plt.hist(res_false, label='fake', log=True, **binning)
    # plt.hist(res_true, label='true', **binning)
    # plt.xlabel('weight')
    # plt.ylabel('count of edges in supergraph')
    # plt.legend(loc=0)
    # plt.subplot(132)
    # res_true1 = lg_nodes[lg_nodes.track != -1][['dx']].values
    # res_false1 = lg_nodes[lg_nodes.track == -1][['dx']].values
    # binning = dict(bins=300) #, range=(-1.5, 1.5)
    # plt.hist(res_false1, label='fake', log=True, **binning)
    # plt.hist(res_true1, label='true', **binning)
    # plt.xlabel('dx')
    # plt.ylabel('count of nodes in supergraph')
    # plt.legend(loc=0)
    # plt.subplot(133)
    # res_true1 = lg_nodes[lg_nodes.track != -1][['dy']].values
    # res_false1 = lg_nodes[lg_nodes.track == -1][['dy']].values
    # binning = dict(bins=300) #, range=(-1.5, 1.5)
    # plt.hist(res_false1, label='fake', log=True, **binning)
    # plt.hist(res_true1, label='true', **binning)
    # plt.xlabel('dy')
    # plt.ylabel('count of nodes in supergraph')
    # plt.legend(loc=0)
    # plt.show()
    # vis = Visualizer(events_df, cfg['visualize'])
    # vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=False)
    # vis.add_graph_data(G)
    # vis.draw()
