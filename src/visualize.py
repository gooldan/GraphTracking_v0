from utils.old_visualizer import Visualizer
from utils.config_reader import ConfigReader
from utils.utils import get_events_df, parse_df
from utils.graph import to_nx_graph, to_line_graph

if __name__ == '__main__':
    reader = ConfigReader("configs/init_config.yaml")
    cfg = reader.cfg
    df = parse_df(cfg['df'])
    events_df = get_events_df(cfg['df'], df, preserve_fakes=False)
    G = to_nx_graph(events_df)
    G1 = to_line_graph(G)
    vis = Visualizer(events_df, cfg['visualize'])
    vis.init_draw(draw_all_hits=True, draw_all_tracks_from_df=False)
    vis.add_graph_data(G)
    vis.draw()
