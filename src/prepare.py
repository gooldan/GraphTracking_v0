
# by e_shchavelev
# inspired from HEP.TrkX

from utils.old_visualizer import Visualizer
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df, calc_purity_reduce_factor, apply_edge_restriction, apply_node_restriction
from utils.graph import to_nx_graph, to_line_graph, get_weight_stats, \
    get_linegraph_superedges_stat, to_pandas_graph_df, get_linegraph_stats_from_pandas, \
    get_reduced_df_graph, get_pd_line_graph, run_mbt_graph
import os
import logging
import numpy as np
import pandas as pd

def construct_output_graph(hits, edges, feature_names=['dx', 'dy', 'dz']):
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
	y[:] = ((pid1 == pid2) & (pid1 != -1)) | ((pid1 == -2) & (pid2 != -1))
	return Graph(X, Ri, Ro, y)


def process_event(prepare_cfg, event_df):
	G = to_pandas_graph_df(event_df)

	def restrict_func(df):
		return apply_node_restriction(df,
									  prepare_cfg['restrictions']['x_min_max'],
									  prepare_cfg['restrictions']['y_min_max'])

	lg_nodes_t, lg_edges_t, \
	mean_purity_t, mean_reduce_t = get_pd_line_graph(G, cfg['df'], restriction_func=restrict_func, reduce_output=True)
	edges_filtered = apply_edge_restriction(lg_edges_t, prepare_cfg['restrictions']['weight_max'])


def prepare_events(base_cfg, config_prepare, df):
	logfilename = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(level=logging.INFO, format=logfilename)
	logging.info("Starting...")
	logging.info("Config:")
	logging.info(base_cfg)

	os.makedirs(config_prepare['output_dir'], exist_ok=True)
	logging.info("Started processing.")
	for id, event in events_df.groupby('event'):
		logging.info("Processing event #%09d ....")

		process_event(event)




if __name__ == '__main__':
	reader = ConfigReader("configs/prepare_config.yaml")
	cfg = reader.cfg
	df = parse_df(cfg['df'])
	events_df = get_events_df(cfg['df'], df, preserve_fakes=True)
	prepare_events(cfg, cfg['prepare'], events_df)
	pass