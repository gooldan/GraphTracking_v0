
# by e_shchavelev
# inspired from HEP.TrkX

from utils.old_visualizer import Visualizer, draw_single
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df, calc_purity_reduce_factor, apply_edge_restriction, apply_node_restriction
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph, run_mbt_graph
import os
import logging
import numpy as np
import pandas as pd
from datasets.graph import Graph, save_graphs_new, load_graph
import time
import sys

def construct_output_graph(hits, edges, feature_names):
    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = edges.shape[0]
    X = (hits[feature_names].values).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.float32)
    Ro = np.zeros((n_hits, n_edges), dtype=np.float32)
    y = np.zeros(n_edges, dtype=np.float32)
    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[edges.edge_index_p].values
    seg_end = hit_idx.loc[edges.edge_index_c].values
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.track.loc[edges.edge_index_p].values
    pid2 = hits.track.loc[edges.edge_index_c].values
    y[:] = ((pid1 == pid2) & (pid1 != -1))
    return Graph(X, Ri, Ro, y)


def process_event(event_id, prepare_cfg, event_df, output_dir):

    G = to_pandas_graph_df(event_df)

    def restrict_func(df):
        return apply_node_restriction(df,
                                      prepare_cfg['restrictions']['x_min_max'],
                                      prepare_cfg['restrictions']['y_min_max'])

    lg_nodes_t, lg_edges_t, \
    mean_purity_t, mean_reduce_t = get_pd_line_graph(G, cfg['df'], restriction_func=restrict_func, reduce_output=True)
    edges_filtered = apply_edge_restriction(lg_edges_t, prepare_cfg['restrictions']['weight_max'])
    e_purity, e_reduce = calc_purity_reduce_factor(lg_edges_t, edges_filtered)
    G = construct_output_graph(lg_nodes_t, edges_filtered, ['dx', 'dy', 'z'])
    save_graphs_new([(G, (output_dir + '/graph_%d' % (event_id)))])
    return mean_reduce_t, mean_purity_t, e_reduce, e_purity

def draw_graph_result(G = None, graph_path = None):
    if G:
        X, Ri, Ro, y = G
    elif graph_path:
        X, Ri, Ro, y = load_graph(graph_path)
    else:
        assert False and "Nothing to draw"

    draw_single(X, Ri, Ro, y, xcord1=(0, 'x'), xcord2=(1, 'y'), ycord=(2, 'z'))

def prepare_events(base_cfg, config_prepare, events_df):
    os.makedirs(config_prepare['output_dir'], exist_ok=True)
    logfilename = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(filename=(config_prepare['output_dir'] + "/process_log.log"), level=logging.INFO, format=logfilename)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("="*20)
    logging.info("="*20)
    logging.info("="*20)
    logging.info("\nStarting...")
    print("prepare")
    logging.info("Config:")
    logging.info(base_cfg)

    logging.info("Started processing.")
    reduce = []
    purity = []
    reduce_edge = []
    purity_edge = []

    start_time = time.time()
    count = 0
    for id, event in events_df.groupby('event'):
        one_event_start_time = time.time()
        logging.info("Processing event #%09d ...." % id)
        reduce_t, purity_t, e_reduce, e_purity = process_event(id, config_prepare, event, config_prepare['output_dir'])
        reduce.extend(reduce_t)
        purity.extend(purity_t)
        reduce_edge.append(e_reduce)
        purity_edge.append(e_purity)
        logging.info("Done (%.5f sec).  p: %.5f r: %.5f ep: %.5f er: %.5f" %((time.time() - one_event_start_time), np.mean(purity_t), np.mean(reduce_t), e_purity, e_reduce))
        count+=1
    result_time = time.time() - start_time
    logging.info("="*20)
    logging.info("Processing done.")
    logging.info("Processed: %d events;    Total time: %d sec;     Speed: %f ev/s" % (count, result_time, count / result_time))
    logging.info("Total purity: %.6f" % (np.mean(purity)*np.mean(purity_edge)))
    logging.info("Node purity: %.6f" % (np.mean(purity)))
    logging.info("Edge purity: %.6f" % (np.mean(purity_edge)))
    logging.info("Node reduce: %.2f times" % (np.mean(reduce)))
    logging.info("Edge reduce: %.2f times" % (np.mean(reduce_edge)))





if __name__ == '__main__':
    reader = ConfigReader("configs/prepare_config.yaml")
    cfg = reader.cfg
    df = parse_df(cfg['df'])
    events_df = get_events_df(cfg['df'], df, preserve_fakes=True)
    prepare_events(cfg, cfg['prepare'], events_df)
    pass