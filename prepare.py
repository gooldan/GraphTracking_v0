
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

import matplotlib.pyplot as plt

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


def process_event(event_id, prepare_cfg, event_df, output_dir, logging):
    logging.info("Composing to graph...")
    G = to_pandas_graph_df(event_df)

    def restrict_func(df):
        return apply_node_restriction(df,
                                      prepare_cfg['restrictions']['x_min_max'],
                                      prepare_cfg['restrictions']['y_min_max'])

    logging.info("Constructing linegraph...")
    lg_nodes_t, lg_edges_t, \
    mean_purity_t, mean_reduce_t = get_pd_line_graph(G, cfg['df'], restriction_func=restrict_func, reduce_output=True)
    edges_filtered = apply_edge_restriction(lg_edges_t, prepare_cfg['restrictions']['weight_max'])
    if lg_edges_t[lg_edges_t.true_superedge != -1].empty or edges_filtered[edges_filtered.true_superedge != -1].empty:
        return mean_reduce_t, mean_purity_t, 1, 0, len(edges_filtered), len(lg_nodes_t), 0
    e_purity, e_reduce = calc_purity_reduce_factor(lg_edges_t, edges_filtered)
    logging.info("Constructing output result...")
    G = construct_output_graph(lg_nodes_t, edges_filtered, ['x_p', 'x_c', 'y_p', 'y_c' 'z'])
    #draw_graph_result(G)
    logging.info("Saving result...")
    save_graphs_new([(G, (output_dir + '/graph_%d' % (event_id)))])
    edge_factor = len(edges_filtered[edges_filtered.true_superedge == -1]) / len(edges_filtered[edges_filtered.true_superedge != -1])
    return mean_reduce_t, mean_purity_t, e_reduce, e_purity, len(edges_filtered), len(lg_nodes_t), edge_factor

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

    def stdout_write(*args):
        sys.stdout.write(*args)

    def empty_write(*args):
        pass

    console_write = stdout_write

    if base_cfg['with_stdout']:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        console_write = empty_write

    logging.info("="*20)
    logging.info("="*20)
    logging.info("="*20)
    logging.info("\nStarting...")
    console_write("\nstarting\n")
    logging.info("Config:")
    logging.info(base_cfg)

    logging.info("Started processing.")
    info_dict = {
        'event': [],
        'reduce': [],
        'purity': [],
        'reduce_edge': [],
        'purity_edge': [],
        'edge_count': [],
        'node_count': [],
        'edge_factor': [],
        'process_time': []
    }
    # reduce = []
    # purity = []
    # reduce_edge = []
    # purity_edge = []
    # edge_count = []
    # node_count = []
    start_time = time.time()
    count = 0
    for id, event in events_df.groupby('event'):
            one_event_start_time = time.time()
            logging.info("Processing event #%09d ...." % id)
            #console_write('\r Processing event #%09d ....' % id)
            try:
                reduce_t, purity_t, e_reduce, \
                e_purity, edge_count_, node_count_, edge_factor = process_event(id, config_prepare, event, config_prepare['output_dir'], logging)
            except MemoryError:
                logging.info("MEMORY ERROR ON %d event" % id)
                console_write('\n\n mem error event %d \n\n' % id)
                continue
            except KeyboardInterrupt:
                break
            except ValueError:
                continue
            info_dict['event'].append(id)
            info_dict['reduce'].append(np.mean(reduce_t))
            info_dict['purity'].append(np.mean(purity_t))
            info_dict['reduce_edge'].append(e_reduce)
            info_dict['purity_edge'].append(e_purity)
            info_dict['edge_count'].append(edge_count_)
            info_dict['edge_factor'].append(edge_factor)
            info_dict['node_count'].append(node_count_)
            info_dict['process_time'].append(time.time() - one_event_start_time)

            console_write('\r Processing event #%09d .... p: %03.5f r: %03.5f ep: %03.5f er: %03.5f, f: %03.5f' %
                          (id, np.mean(purity_t), np.mean(reduce_t), e_purity, e_reduce, edge_factor))

            logging.info("Done (%.5f sec).  p: %.5f r: %.5f ep: %.5f er: %.5f f: %03.5f" %
                         ((time.time() - one_event_start_time), np.mean(purity_t), np.mean(reduce_t), e_purity, e_reduce, edge_factor))
            count += 1


    result_time = time.time() - start_time
    logging.info("="*20)
    logging.info("Processing done.")
    logging.info("Processed: %d events;    Total time: %d sec;     Speed: %f ev/s" % (count, result_time, count / result_time))
    logging.info("Total purity: %.6f" % (np.mean(info_dict['purity'])*np.mean(info_dict['purity_edge'])))
    logging.info("Node purity: %.6f" % (np.mean(info_dict['purity'])))
    logging.info("Edge purity: %.6f" % (np.mean(info_dict['purity_edge'])))
    logging.info("Node reduce: %.2f times" % (np.mean(info_dict['reduce'])))
    logging.info("Edge reduce: %.2f times" % (np.mean(info_dict['reduce_edge'])))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    binning = dict(bins=300)
    plt.hist(info_dict['node_count'], label='node_count', log=True, **binning)
    plt.xlabel('counts')
    plt.ylabel('count of events')
    plt.legend(loc=0)

    plt.subplot(122)
    binning = dict(bins=300)
    plt.hist(info_dict['edge_count'], label='edge_count', log=True, **binning)
    plt.xlabel('dx')
    plt.ylabel('count of nodes in supergraph')
    plt.legend(loc=0)
    plt.savefig(config_prepare['output_dir'] + '/result.png', bbox_inches='tight', dpi=500)
    stat_dict = pd.DataFrame.from_dict(info_dict)
    stat_dict.to_csv(config_prepare['output_dir'] + '/prepare_stats.csv')
    plt.show()



if __name__ == '__main__':
    reader = ConfigReader("configs/prepare_config.yaml")
    cfg = reader.cfg
    df = parse_df(cfg['df'])
    events_df = get_events_df(cfg['df'], df, preserve_fakes=True)
    prepare_events(cfg, cfg['prepare'], events_df)
    pass