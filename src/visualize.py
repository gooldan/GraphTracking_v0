from utils.old_visualizer import Visualizer
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df, calc_purity_reduce_factor, apply_edge_restriction, apply_node_restriction
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph, run_mbt_graph


import matplotlib.pyplot as plt
import numpy as np
def get_like_hitgraph_from_linegraph(df):
    df = df.rename(columns={'dx':'x', 'dy':'y', 'dz':'z'})
    return df


import pandas as pd
if __name__ == '__main__':
    reader = ConfigReader("configs/init_config.yaml")
    cfg = reader.cfg
    df = parse_df(cfg['df'])
    events_df = get_events_df(cfg['df'], df, preserve_fakes=True)
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
        return apply_node_restriction(df, [-0.15, 0.25], [-0.3, 0.22])

    for id, event in events_df.groupby('event'):
        G = to_pandas_graph_df(event)
        lg_nodes_t, lg_edges_t, mean_purity_t, mean_reduce_t = get_pd_line_graph(G,
                                                                                 with_station_info=True,
                                                                                 restriction_func=restrict_func,
                                                                                 reduce_output=False)

        like_original_df = get_like_hitgraph_from_linegraph(lg_nodes_t)
        edges_filtered = apply_edge_restriction(lg_edges_t, 0.09)
        # vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
        # vis.init_draw(draw_all_hits=False, draw_all_tracks_from_df=False)
        # vis.add_edges_data(edges_filtered)
        # vis.draw(show=False)

        #gg, p, m = run_mbt_graph(lg_nodes_t, edges_filtered)
        p, m = calc_purity_reduce_factor(lg_edges_t, edges_filtered)
        mean_edge_reduce.append(m)
        mean_edge_purity.append(p)
        #new_nodes = apply_node_restriction(lg_nodes_t, [-0.2, 0.2], [-10, 10])
        mean_reduce.extend(mean_reduce_t)
        mean_purity.extend(mean_purity_t)
        lg_nodes = lg_nodes.append(lg_nodes_t, ignore_index=True, sort=False)
        lg_edges = lg_edges.append(lg_edges_t, ignore_index=True, sort=False)

        # # vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
        # # vis.init_draw(draw_all_hits=False, draw_all_tracks_from_df=False)
        # # vis.add_edges_data(gg)
        # # vis.draw(show=True)
        # print("Purity:", p)
        # print("Reduce:", m)
        print(id)

    # edges_filtered = apply_edge_restriction(lg_edges, 0.09)
    # purity, reduce_factor = calc_purity_reduce_factor(lg_edges, edges_filtered)
    # print('edges -- purity:', purity)
    # print('edges -- reduce_factor:', reduce_factor)
    print("nodes -- reduce:", np.mean(mean_reduce))
    print("nodes -- purity:", np.mean(mean_purity))
    print("edges -- reduce:", np.mean(mean_edge_reduce))
    print("edges -- purity:", np.mean(mean_edge_purity))
    # vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
    #     # vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #     # vis.draw(show=False)
    #
    # for id, event in events_df.groupby('event'):
    #     G = to_pandas_graph_df(event)
    #     st_true, st_false, res_ev_df = get_reduced_df_graph(G, event)
    #     res_true.extend(st_true)
    #     res_false.extend(st_false)
    #     print(event.event.values[0])
    #     vis = Visualizer(res_ev_df, cfg['visualize'])
    #     vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #     vis.draw(show=False)
    #
    #     vis = Visualizer(event, cfg['visualize'])
    #     vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #     vis.draw(show=True)

    plt.figure(figsize=(12, 6))
    res_true = lg_edges[lg_edges.true_superedge != -1][['weight']].values
    res_false = lg_edges[lg_edges.true_superedge == -1][['weight']].values
    plt.subplot(131)
    binning = dict(bins=300, range=(-0.05, 5.1))
    plt.hist(res_false, label='fake', log=True, **binning)
    plt.hist(res_true, label='true', **binning)
    plt.xlabel('weight')
    plt.ylabel('count of edges in supergraph')
    plt.legend(loc=0)
    plt.subplot(132)
    res_true1 = lg_nodes[lg_nodes.track != -1][['dx']].values
    res_false1 = lg_nodes[lg_nodes.track == -1][['dx']].values
    binning = dict(bins=300, range=(-1.5, 1.5))
    plt.hist(res_false1, label='fake', log=True, **binning)
    plt.hist(res_true1, label='true', **binning)
    plt.xlabel('dx')
    plt.ylabel('count of nodes in supergraph')
    plt.legend(loc=0)
    plt.subplot(133)
    res_true1 = lg_nodes[lg_nodes.track != -1][['dy']].values
    res_false1 = lg_nodes[lg_nodes.track == -1][['dy']].values
    binning = dict(bins=300, range=(-1.5, 1.5))
    plt.hist(res_false1, label='fake', log=True, **binning)
    plt.hist(res_true1, label='true', **binning)
    plt.xlabel('dy')
    plt.ylabel('count of nodes in supergraph')
    plt.legend(loc=0)
    plt.show()
    # vis = Visualizer(events_df, cfg['visualize'])
    # vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=False)
    # vis.add_graph_data(G)
    # vis.draw()
