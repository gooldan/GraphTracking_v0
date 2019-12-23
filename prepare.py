
# by e_shchavelev
# inspired from HEP.TrkX

from utils.old_visualizer import Visualizer, draw_single
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df, calc_purity_reduce_factor, apply_edge_restriction, apply_node_restriction, apply_segments_restriction
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph, run_mbt_graph, calc_dphi
import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
from datasets.graph import Graph, save_graphs_new, load_graph
import time
import sys

import matplotlib.pyplot as plt


def construct_output_graph(hits, edges, feature_names, feature_scale, index_label_prev='edge_index_p', index_label_current='edge_index_c'):
    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = edges.shape[0]
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.float32)
    Ro = np.zeros((n_hits, n_edges), dtype=np.float32)
    y = np.zeros(n_edges, dtype=np.float32)
    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[edges[index_label_prev]].values
    seg_end = hit_idx.loc[edges[index_label_current]].values
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.track.loc[edges[index_label_prev]].values
    pid2 = hits.track.loc[edges[index_label_current]].values
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
    mean_purity_t, mean_reduce_t = get_pd_line_graph(G, with_station_info=False, restriction_func=restrict_func, reduce_output=True)
    edges_filtered = apply_edge_restriction(lg_edges_t, prepare_cfg['restrictions']['weight_max'])
    if lg_edges_t[lg_edges_t.true_superedge != -1].empty or edges_filtered[edges_filtered.true_superedge != -1].empty:
        return mean_reduce_t, mean_purity_t, 1, 0, len(edges_filtered), len(lg_nodes_t), 0
    e_purity, e_reduce = calc_purity_reduce_factor(lg_edges_t, edges_filtered)
    logging.info("Constructing output result...")
    G = construct_output_graph(lg_nodes_t, edges_filtered, ['x_p', 'x_c', 'y_p', 'y_c', 'z'])
    #draw_graph_result(G)
    logging.info("Saving result...")
    save_graphs_new([(G, (output_dir + '/graph_%d' % (event_id)))])
    edge_factor = len(edges_filtered[edges_filtered.true_superedge == -1]) / len(edges_filtered[edges_filtered.true_superedge != -1])
    return mean_reduce_t, mean_purity_t, e_reduce, e_purity, len(edges_filtered), len(lg_nodes_t), edge_factor

def draw_graph_result(G = None, graph_path = None):
    if G is not None:
        X, Ri, Ro, y = G
    elif graph_path is not None:
        X, Ri, Ro, y = load_graph(graph_path)
    else:
        assert False and "Nothing to draw"

    draw_single(X, Ri, Ro, y, c_fake = (0,0,0,0.1), xcord1=(0, 'z'), xcord2=(2, 'phi'), ycord=(4, 'r'))

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
    if 'mode' in config_prepare and config_prepare['mode'] == 'cgem':
        prepare_cgem(config_prepare, console_write, events_df, info_dict)
    else:
        prepare_bmn(config_prepare, console_write, events_df, info_dict)


def prepare_bmn(config_prepare, console_write, events_df, info_dict):
    start_time = time.time()
    count = 0
    for id, event in events_df.groupby('event'):
        one_event_start_time = time.time()
        logging.info("Processing event #%09d ...." % id)
        # console_write('\r Processing event #%09d ....' % id)
        try:
            reduce_t, purity_t, e_reduce, \
            e_purity, edge_count_, node_count_, edge_factor = process_event(id, config_prepare, event,
                                                                            config_prepare['output_dir'], logging)
        except MemoryError:
            logging.info("MEMORY ERROR ON %d event" % id)
            console_write('\n\n mem error event %d \n\n' % id)
            continue
        except KeyboardInterrupt:
            break
        except:
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
                     ((time.time() - one_event_start_time), np.mean(purity_t), np.mean(reduce_t), e_purity, e_reduce,
                      edge_factor))
        count += 1
    result_time = time.time() - start_time
    logging.info("=" * 20)
    logging.info("Processing done.")
    logging.info(
        "Processed: %d events;    Total time: %d sec;     Speed: %f ev/s" % (count, result_time, count / result_time))
    logging.info("Total purity: %.6f" % (np.mean(info_dict['purity']) * np.mean(info_dict['purity_edge'])))
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

def get_edges_from_supernodes_ex(sn_from, sn_to):
    prev_edges = sn_from.rename(columns={'to_ind':'cur_ind'})
    next_edges = sn_to.rename(columns={'from_ind':'cur_ind'})
    prev_edges['edge_index'] = prev_edges.index
    next_edges['edge_index'] = next_edges.index
    line_graph_edges = pd.merge(prev_edges, next_edges, on='cur_ind', suffixes=('_p', '_c'))

    line_graph_edges = line_graph_edges.assign(true_superedge=-1)
    index_true_superedge = line_graph_edges[(line_graph_edges.track_p == line_graph_edges.track_c) &
                                            (line_graph_edges.track_p != -1)]
    line_graph_edges.loc[index_true_superedge.index, 'true_superedge'] = index_true_superedge.track_p.values



    ba = line_graph_edges[['dx_p', 'dy_p', 'dz_p']].values
    cb = line_graph_edges[['dx_c', 'dy_c', 'dz_c']].values
    # norm_ba = np.linalg.norm(ba, axis=1)
    # norm_cb = np.linalg.norm(cb, axis=1)
    # dot = np.einsum('ij,ij->i', ba, cb)
    w = np.linalg.norm(ba - cb, axis=1)
    line_graph_edges['weight'] = w
    indexation = ['weight', 'true_superedge', 'edge_index_p', 'edge_index_c']
    return line_graph_edges[indexation]


def get_supernodes_df_ex(linegraph_cfg, one_station_segments, station=-1, STAION_COUNT=0, pi_fix=False):
    ret = pd.DataFrame()
    axes = linegraph_cfg['axes']
    suffix_p, suffix_c = linegraph_cfg['suffix']

    x0_y0_z0 = one_station_segments[[axes[0] + suffix_p, axes[1] + suffix_p, axes[2] + suffix_p]].values
    x1_y1_z1 = one_station_segments[[axes[0] + suffix_c, axes[1] + suffix_c, axes[2] + suffix_c]].values
    dx_dy_dz = x1_y1_z1 - x0_y0_z0

    if pi_fix:
        dy_fixed = calc_dphi(one_station_segments[axes[1] + suffix_p].values,
                             one_station_segments[axes[1] + suffix_c].values)
        ret = ret.assign(dx=dx_dy_dz[:, 0], dy=dy_fixed,

                         z_p=one_station_segments[axes[2] + suffix_p].values,
                         z_c=one_station_segments[axes[2] + suffix_c].values,
                         y_p=one_station_segments[axes[1] + suffix_p].values,
                         y_c=one_station_segments[axes[1] + suffix_c].values,

                         dz=dx_dy_dz[:, 2], z=(station + 1) / STAION_COUNT)
    else:
        ret = ret.assign(dx=dx_dy_dz[:, 0], dy=dx_dy_dz[:, 1],

                         z_p=one_station_segments[axes[2] + suffix_p].values,
                         z_c=one_station_segments[axes[2] + suffix_c].values,
                         y_p=one_station_segments[axes[1] + suffix_p].values,
                         y_c=one_station_segments[axes[1] + suffix_c].values,

                         dz=dx_dy_dz[:, 2], z=(station + 1) / STAION_COUNT)

    ret = ret.assign(from_ind=one_station_segments[['index_old' + suffix_p]].values.astype(np.uint32))
    ret = ret.assign(to_ind=one_station_segments[['index_old' + suffix_c]].values.astype(np.uint32))
    ret = ret.assign(track=-1)
    ret = ret.set_index(one_station_segments.index)
    index_true_superedge = one_station_segments[
        (one_station_segments['track' + suffix_p] == one_station_segments['track' + suffix_c]) & \
        (one_station_segments['track' + suffix_p] != -1)]
    ret.loc[index_true_superedge.index, 'track'] = index_true_superedge['track' + suffix_p].values
    ret = ret.assign(station=station)
    return ret


def get_pd_line_graph_ex(prepare_cfg, segments, restrictions_func):
    nodes = pd.DataFrame()
    edges = pd.DataFrame()

    restrictions_0, restrictions_1 = prepare_cfg['linegraph']['restrictions']

    by_stations = [df for (ind, df) in segments.groupby('station' + prepare_cfg['linegraph']['suffix'][0])]
    for i in range(1, len(by_stations)):
        supernodes_from = get_supernodes_df_ex(prepare_cfg['linegraph'], by_stations[i - 1], i - 1, 3, pi_fix=True)
        supernodes_to = get_supernodes_df_ex(prepare_cfg['linegraph'], by_stations[i], i, 3, pi_fix=True)

        if restrictions_func:
            supernodes_from = restrictions_func(supernodes_from, restrictions_0, restrictions_1)
            supernodes_to = restrictions_func(supernodes_to, restrictions_0, restrictions_1)

        nodes = nodes.append(supernodes_from, sort=False)
        nodes = nodes.append(supernodes_to, sort=False)

        superedges = get_edges_from_supernodes_ex(supernodes_from, supernodes_to)
        edges = edges.append(superedges, ignore_index=True, sort=False)

    nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
    return nodes, edges

def prepare_cgem(config_prepare, console_write, events_df, info_dict):

    def apply_nodes_restrictions(nodes, restrictions_0, restrictions_1):
        return nodes[
            (nodes.dy > restrictions_0[0]) & (nodes.dy < restrictions_0[1]) &
            (nodes.dz > restrictions_1[0]) & (nodes.dz < restrictions_1[1])
            ]

    start_time = time.time()
    count = 0
    output_dir = config_prepare['output_dir']
    for id, event in events_df.groupby('event'):
        one_event_start_time = time.time()
        logging.info("Processing event #%09d ...." % id)
        console_write('\r Processing event #%09d ....' % id)
        try:
            G = to_pandas_graph_df(event, suffx=['_p', '_c'], compute_is_true_track=True)

            nodes_t, edges_t = get_pd_line_graph_ex(config_prepare, G, apply_nodes_restrictions)

            # skip completely broken events
            if edges_t.empty:
                logging.info("SKIPPED broken %d event" % id)
                console_write('\n\n skipping broken event %d \n\n' % id)
                continue

            edges_filtered = apply_edge_restriction(edges_t, config_prepare['linegraph']['weight_r'])


            count_true = len(edges_filtered[edges_filtered.true_superedge != -1])

            if count_true == 0:
                purity_, reduce_ = (1, 0)
            else:
                purity_, reduce_ = calc_purity_reduce_factor(edges_t, edges_filtered, 'true_superedge', -1)
            info_dict['reduce'].append(reduce_)
            info_dict['purity'].append(purity_)

            out = construct_output_graph(nodes_t, edges_filtered, ['y_p', 'y_c', 'z_p', 'z_c', 'z'], [np.pi, np.pi, 1., 1., 1.], 'edge_index_p', 'edge_index_c')


            info_dict['node_count'].append(len(nodes_t))
            info_dict['edge_count'].append(len(edges_filtered))
            count_true = count_true if count_true > 0 else 1
            info_dict['edge_factor'].append((len(edges_filtered) - count_true) / count_true )
            # draw_graph_result(out)
            # exit()
        except MemoryError:
            logging.info("MEMORY ERROR ON %d event" % id)
            console_write('\n\n mem error event %d \n\n' % id)
            continue
        except KeyboardInterrupt:
            break
        except KeyError as err:
            logging.info("KEY ERROR ON %d event err:%s" % (id, err))
            console_write('\n\n key error event %d err:%s\n\n' % (id, err))
            continue
        except AttributeError:
            logging.info("ATTRIBUTE ERROR ON %d event" % id)
            console_write('\n\n attr error event %d \n\n' % id)
            continue
        save_graphs_new([(out, (output_dir + '/graph_%d' % (id)))])
        # except Exception as ex:
        #     logging.error(ex)
        #     console_write('\nexception!! event=%d\n' % id)
        #     console_write(str(ex))
        #     print(sys.exc_info()[2])
        #     console_write('\n=========\n')
        #     continue
        info_dict['process_time'].append(time.time() - one_event_start_time)
        count+=1

    result_time = time.time() - start_time
    logging.info("=" * 20)
    logging.info("Processing done.")
    logging.info("Total processed %d events. Mean node count: %.6f; Mean segments count: %.6f" % (count, np.mean(info_dict['node_count']), np.mean(info_dict['edge_count'])))
    logging.info("After filtration mean purity is %.6f and reduce factor is %.6f." % (np.mean(info_dict['purity']), np.mean(info_dict['reduce'])))
    logging.info("Speed is %.3f events per second" % (count / result_time))
    logging.info("Fake edge ratio is %.6f" % np.mean(info_dict['edge_factor']))

if __name__ == '__main__':

    draw_graph_result(graph_path="output/cgem_4k_LINEGRAPH_final/graph_3413.npz")
    exit()
    reader = ConfigReader("configs/cgem_prepare_config.yaml")
    cfg = reader.cfg
    df = parse_df(cfg['df'])
    events_df = get_events_df(cfg['df']['take'], df, preserve_fakes=True)
    prepare_events(cfg, cfg['prepare'], events_df)
    pass