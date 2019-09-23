from utils.old_visualizer import Visualizer
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph


import matplotlib.pyplot as plt
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
    for id, event in events_df.groupby('event'):
        G = to_pandas_graph_df(event)
        lg_nodes_t, lg_edges_t = get_pd_line_graph(G, cfg['df'], with_station_info=True)
        like_original_df = get_like_hitgraph_from_linegraph(lg_nodes_t)
        lg_nodes = lg_nodes.append(lg_nodes_t, ignore_index=True, sort=False)
        lg_edges = lg_edges.append(lg_edges_t, ignore_index=True, sort=False)
        print(id)
    #     vis = Visualizer(like_original_df, cfg['visualize'], title='EVENT LINEGRAPH', random_seed=14)
    #     vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=True)
    #     vis.draw(show=False)
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
    plt.subplot(121)
    binning = dict(bins=300, range=(-0.05, 5.1))
    plt.hist(res_false, label='fake', log=True, **binning)
    plt.hist(res_true, label='true', **binning)
    plt.xlabel('weight')
    plt.ylabel('count of edges in supergraph')
    plt.legend(loc=0)
    plt.subplot(122)
    res_true1 = lg_nodes[lg_nodes.track != -1][['dx']].values
    res_false1 = lg_nodes[lg_nodes.track == -1][['dx']].values
    binning = dict(bins=300, range=(-1.5, 1.5))
    plt.hist(res_false1, label='fake', log=True, **binning)
    plt.hist(res_true1, label='true', **binning)
    plt.xlabel('dphi')
    plt.ylabel('count of nodes in supergraph')
    plt.legend(loc=0)
    plt.show()
    # vis = Visualizer(events_df, cfg['visualize'])
    # vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=False)
    # vis.add_graph_data(G)
    # vis.draw()
