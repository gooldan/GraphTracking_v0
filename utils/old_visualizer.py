import matplotlib

if not 'inline' in matplotlib.get_backend():
    matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
import matplotlib.cm as cm
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d


def revert_types(df):
    df = df.astype({'track':'int32', 'station':'int32', 'event':'int32'})
    df.index = df.index.map(int)
    return df

class Visualizer:
    Z_ORDER_FAKE_HIT = 3
    Z_ORDER_TRUE_HIT = 4
    Z_ORDER_FAKE_EDGE = 1
    Z_ORDER_TRUE_EDGE = 2

    def __init__(self, df, cfg, title = "EVENT GRAPH", random_seed=13):
        np.random.seed(random_seed)
        self.__df = df
        self.__z_coord = cfg['z_coord']
        self.__x_coord = cfg['x_coord']
        self.__y_coord = cfg['y_coord']
        self.__draw_cfg = cfg
        self.__axs = []
        self.__color_map = {-1: np.array([[0.1, 0.1, 0.1, 1.]])}
        self.__adj_track_list = []
        self.__reco_adj_list = []
        self.__fake_hits = np.empty(shape=(0,3))
        self.__nn_preds = []
        self.__coord_planes = []
        self.__nx_edges = []
        self.__draw_all_hits = False
        self.__draw_all_tracks_from_df = False
        self.__title = title
        self.__nx_line_edges = []
        self.__pd_line_edges = pd.DataFrame()
        self.__pd_line_edges_ex = pd.DataFrame()

    def init_draw(self, reco_tracks = None, draw_all_tracks_from_df = False, draw_all_hits = False):
        self.__draw_all_hits = draw_all_hits
        self.__draw_all_tracks_from_df = draw_all_tracks_from_df
        grouped = self.__df.groupby('track')
        # prepare adjacency list for tracks
        for i, gp in grouped:
            if gp.track.values[0] == -1:
                self.__fake_hits = np.append(self.__fake_hits, gp[[self.__z_coord, self.__x_coord, self.__y_coord]].values, axis=0)
                continue

            if not self.__draw_cfg['draw_scatters_for_tracks']:
                self.__fake_hits = np.append(self.__fake_hits, gp[[self.__z_coord, self.__x_coord, self.__y_coord]].values, axis=0)

            for row in range(1, len(gp.index)):
                elem = (gp.index[row - 1], gp.index[row], 1)
                self.__adj_track_list.append(elem)

        if reco_tracks is not None:
            for track in reco_tracks:
                for i in range(0, len(track) - 2):
                    if track[i] == -1 or track[i+1] == -1:
                        break
                    self.__reco_adj_list.append((track[i], track[i + 1], 1))

    def add_nn_pred(self, z_ell_coord, from_idx, pred_X_Y_Station, pred_R1_R2):
         self.__nn_preds.append([z_ell_coord,from_idx, pred_X_Y_Station, pred_R1_R2])

    def add_coord_planes(self, coord_planes_arr):
        self.__coord_planes = coord_planes_arr

    def add_graph_data(self, nx_graph):
        self.__nx_edges = nx_graph.edges

    def add_line_graph_data(self, nx_line_graph):
        assert len(self.__nx_line_edges) == 0
        self.__nx_line_edges = nx_line_graph.edges

    def add_edges_data(self, edges_df):
        assert len(self.__pd_line_edges) == 0
        self.__pd_line_edges = edges_df[['edge_index_p', 'edge_index_c', 'true_superedge']]

    def add_edges_data_ex(self, edges_df):
        self.__pd_line_edges_ex = edges_df

    def set_title(self, title = "EVENT GRAPH"):
        self.__title = title

    def draw_3d(self):
        matplotlib.rcParams['legend.fontsize'] = 10
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.__title)
        ax.set_xlabel(self.__z_coord)
        ax.set_ylabel(self.__x_coord)
        ax.set_zlabel(self.__y_coord)
        legends = {}
        if self.__draw_all_hits:
            ax.scatter(self.__fake_hits[:,0], self.__fake_hits[:,1], self.__fake_hits[:,2], c=self.__color_map[-1], marker='o')

        if self.__draw_all_tracks_from_df:
            for adj_val in self.__adj_track_list:
                col, lab, tr_id = self.draw_edge_3d(adj_val, ax)
                if int(tr_id) not in legends:
                    legends[int(tr_id)] = mpatches.Patch(color=col, label=lab)
                    legends[int(tr_id)] = mpatches.Patch(color=col, label=lab)

        for adj_val in self.__reco_adj_list:
            col, lab, tr_id = self.draw_edge_3d(adj_val, ax)
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)

        for ell_data in self.__nn_preds:
            ell = Ellipse(xy=ell_data[2], width=ell_data[3][0], height=ell_data[3][1], color='red')
            ax.add_patch(ell)
            art3d.pathpatch_2d_to_3d(ell, z=ell_data[0], zdir="x")
            col, lab, tr_id = self.draw_edge_3d_from_idx_to_pnt(ell_data[1], [ell_data[0], ell_data[2][0], ell_data[2][1]],
                                                             ax)
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)

        for station_id, coord_planes in enumerate(self.__coord_planes):
            for rect_data in coord_planes:
                rect = Rectangle(xy=(rect_data[0] - rect_data[2] / 2, rect_data[1] - rect_data[3] / 2),
                                 width=rect_data[2], height=rect_data[3], linewidth=1, edgecolor='black',
                                 facecolor='none')
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=station_id, zdir="x")

        fig.legend(handles=list(legends.values()))



    def draw_2d(self, ax):
        assert len(self.__nn_preds) == 0 and "Can not draw ellipses on 2d plot"
        fig = plt.figure(figsize=(6, 6))
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_title(self.__title)
        ax.set_xlabel(self.__x_coord)
        ax.set_ylabel(self.__z_coord)
        legends = {}
        if self.__draw_all_hits:
            ax.scatter(self.__fake_hits[:,1], self.__fake_hits[:,0], c=self.__color_map[-1], marker='o')

        if self.__draw_all_tracks_from_df:
            for adj_val in self.__adj_track_list:
                col, lab, tr_id = self.draw_edge_2d(adj_val, ax, drop_fake_percent=0.8)
                if int(tr_id) not in legends:
                    legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)

        for adj_val in self.__reco_adj_list:
            col, lab, tr_id = self.draw_edge_2d(adj_val, ax)
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)

        for edge in self.__nx_edges:
            col, lab, tr_id = self.draw_edge_2d(edge, ax, drop_fake_percent=0.8)
            if col is None:
                continue
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)

        if len(self.__pd_line_edges) > 0:
            self.draw_edges_robust_2d(ax)

        if len(self.__pd_line_edges_ex) > 0:
            self.draw_edges_robust_2d_ex(ax)

        for edge in self.__nx_line_edges:
            col, lab, tr_id = self.draw_edge_2d(edge, ax, drop_fake_percent=0.8)
            if col is None:
                continue
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col[0], label=lab)
        fig.legend(handles=list(legends.values()))
        return ax

    def draw_edges_robust_2d(self, ax):
        assert False, 'deprecated'
        nodes_true = self.__df[self.__df.track != -1]
        nodes_false = self.__df[self.__df.track == -1]

        nodes_from_true = nodes_true.loc[self.__pd_line_edges.edge_index_p.values]
        nodes_from_false = nodes_false.loc[self.__pd_line_edges.edge_index_p.values]

        nodes_to_true = nodes_true.loc[self.__pd_line_edges.edge_index_c.values]
        nodes_to_false = nodes_false.loc[self.__pd_line_edges.edge_index_c.values]

        c = self.__pd_line_edges.apply(lambda x: 'b'
                                            if x.true_superedge != -1
                                            else 'r', axis=1)
        z = self.__pd_line_edges.apply(lambda x: self.Z_ORDER_TRUE_EDGE
                                            if x.true_superedge != -1
                                            else self.Z_ORDER_FAKE_EDGE, axis=1)

        self.draw_edges_from_nodes_2d(ax, nodes_from_false, nodes_to_false, [0.1, 0.1, 0.1, 0.5],  [0.1, 0.1, 0.1, 1],
                                      self.Z_ORDER_FAKE_EDGE, self.Z_ORDER_FAKE_HIT, 1)

        self.draw_edges_from_nodes_2d(ax, nodes_from_true, nodes_to_true, 'orange',  [0.1, 0.1, 0.1, 1],
                                      self.Z_ORDER_TRUE_EDGE, self.Z_ORDER_TRUE_HIT, 2)

    def draw_edges_from_nodes_2d(self, ax, nodes_from, nodes_to, color, pnt_color, z_line, z_dot, line_width ):

        ax.scatter(nodes_from[self.__x_coord].values, nodes_from[self.__z_coord].values, c=pnt_color, marker='o', zorder=z_dot)
        ax.scatter(nodes_to[self.__x_coord].values, nodes_to[self.__z_coord].values, c=pnt_color, marker='o')

        x0 = nodes_from[[self.__x_coord]].values
        y0 = nodes_from[[self.__z_coord]].values
        x1 = nodes_to[[self.__x_coord]].values
        y1 = nodes_to[[self.__z_coord]].values
        lines = np.dstack((np.hstack((x0, x1)), np.hstack((y0, y1))))
        lk = LineCollection(lines, color=[color]*len(lines), linewidths=[line_width]*len(lines), zorder=z_line)
        ax.add_collection(lk)

    def draw(self, show=True, ax=None):
        if self.__draw_cfg['mode'] == '3d':
            ax = self.draw_3d()
        else:
            ax = self.draw_2d(ax)

        if show:
            plt.draw_all()
            plt.tight_layout()
            plt.show()
        return ax

    def draw_edge_3d(self, adj_val, ax):
        hit_from = self.__df.loc[adj_val[0]]
        hit_to = self.__df.loc[adj_val[1]]
        color, label, tr_id = self.generate_color_label_3d(int(hit_from.track), int(hit_to.track))
        marker_1 = 'h' if hit_from.track == -1 else 'o'
        marker_2 = 'h' if hit_to.track == -1 else 'o'
        ax.plot((hit_from[self.__z_coord], hit_to[self.__z_coord]), (hit_from[self.__x_coord], hit_to[self.__x_coord]), zs=(hit_from[self.__y_coord], hit_to[self.__y_coord]), c=color)
        if self.__draw_cfg['draw_scatters_for_tracks']:
            ax.scatter(hit_from[self.__z_coord], hit_from[self.__x_coord], hit_from[self.__y_coord], c=self.__color_map[int(hit_from.track)], marker=marker_1)
            ax.scatter(hit_to[self.__z_coord], hit_to[self.__x_coord], hit_to[self.__y_coord], c=self.__color_map[int(hit_to.track)], marker=marker_2)
        return color, label, tr_id

    def draw_edge_2d(self, adj_val, ax, drop_fake_percent=0.):
        hit_from = self.__df.loc[adj_val[0]]
        hit_to = self.__df.loc[adj_val[1]]
        if drop_fake_percent > 0:
            if hit_from.track == -1 or hit_to.track == -1 and np.random.random_sample() < drop_fake_percent:
                return None, None, None
        color, label, tr_id = self.generate_color_label_2d(int(hit_from.track), int(hit_to.track))
        marker_1 = 'h' if hit_from.track == -1 else 'o'
        marker_2 = 'h' if hit_to.track == -1 else 'o'
        zorder_edge = Visualizer.Z_ORDER_TRUE_EDGE if tr_id != -1 else Visualizer.Z_ORDER_FAKE_EDGE
        zorder_hit = Visualizer.Z_ORDER_TRUE_HIT if tr_id != -1 else Visualizer.Z_ORDER_FAKE_HIT
        ax.plot((hit_from[self.__x_coord], hit_to[self.__x_coord]), (hit_from[self.__z_coord], hit_to[self.__z_coord]), c=color[0], zorder=zorder_edge)
        if self.__draw_cfg['draw_scatters_for_tracks']:
            ax.scatter(hit_from[self.__x_coord], hit_from[self.__z_coord], c=self.__color_map[int(hit_from.track)], marker=marker_1, zorder=zorder_hit)
            ax.scatter(hit_to[self.__x_coord], hit_to[self.__z_coord], c=self.__color_map[int(hit_to.track)], marker=marker_2, zorder=zorder_hit)
        return color, label, tr_id

    def draw_edge_3d_from_idx_to_pnt(self, from_idx,
                                  to_coord_STATXY, ax,
                                  line_color=np.random.rand(3,),
                                  marker='h',
                                  pnt_color='yellow'):
        hit_from = self.__df.loc[from_idx]
        ax.plot((hit_from[self.__z_coord], to_coord_STATXY[0]), (hit_from[self.__x_coord], to_coord_STATXY[1]),
                zs=(hit_from[self.__y_coord], to_coord_STATXY[2]), c=line_color)
        ax.scatter(to_coord_STATXY[0], to_coord_STATXY[1], to_coord_STATXY[2],
                   c=pnt_color, marker=marker)
        return line_color, 'test edge from ev_id:' + str(int(hit_from.track)), int(hit_from.track)

    def redraw_all(self):
        pass

    def generate_color_label_3d(self, tr_id_from, tr_id_to, use_bad_conn=True):
        if tr_id_from not in self.__color_map:
            self.__color_map[tr_id_from] = np.random.rand(3,)
        if tr_id_to not in self.__color_map:
            self.__color_map[tr_id_to] = np.random.rand(3,)
        if tr_id_from != tr_id_to:
            return (1, 0.1, 0.1), 'bad connection', tr_id_from<<16|tr_id_to
        if tr_id_from == -1:
            return (0.1, 0.1, 0.1), 'fake connection', -1
        return self.__color_map[tr_id_from], 'tr_id: ' + str(int(tr_id_from)), tr_id_from

    def generate_color_label_2d(self, tr_id_from, tr_id_to):
        if tr_id_from not in self.__color_map:
            self.__color_map[tr_id_from] = np.array(np.append(np.random.rand(3,),1)).reshape((1,4))
        if tr_id_to not in self.__color_map:
            self.__color_map[tr_id_to] = np.array(np.append(np.random.rand(3,),1)).reshape((1,4))
        if tr_id_from != tr_id_to or tr_id_from == -1:
            return (0.1, 0.1, 0.1), 'fake connection', -1
        return self.__color_map[tr_id_from], 'tr_id: ' + str(int(tr_id_from)), tr_id_from


def draw_single(X, Ri, Ro, y, c_true = 'green', c_fake = (0,0,0,0.1), xcord1 = (2, 'x'), xcord2 = (1, 'y'), ycord=(0, 'z'), draw_fake=True):
    feats_o = X[np.where(Ri.T)[1]]
    feats_i = X[np.where(Ro.T)[1]]
    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    #                0    1  2  3  4  5
    # Draw the hits (r, phi, z, x, y, z)
    # colMap = np.zeros_like(X)
    # colMap[:, 0] = X[:, 0]*1.7 + 100
    # colMap *= 1.0 / colMap.max()
    ax0.scatter(X[:, xcord1[0]],
                X[:, ycord[0]], c='black')
    ax1.scatter(X[:, xcord2[0]],
                X[:, ycord[0]], c='black')

    # Draw the segments
    for j in range(y.shape[0]):
        ax0.plot([feats_o[j, xcord1[0]], feats_i[j, xcord1[0]]],
                 [feats_o[j, ycord[0]], feats_i[j, ycord[0]]], '-', c=c_true if y[j] > 0.5 else c_fake, zorder=10 if y[j] > 0.5 else 1)
        ax1.plot([feats_o[j, xcord2[0]], feats_i[j, xcord2[0]]],
                 [feats_o[j, ycord[0]], feats_i[j, ycord[0]]], '-', c=c_true if y[j] > 0.5 else c_fake, zorder=10 if y[j] > 0.5 else 1)
    # Adjust axes
    ax0.set_xlabel('$%s$' % xcord1[1])
    ax0.set_ylabel('$%s$' % ycord[1])

    ax1.set_xlabel('$%s$' % xcord2[1])
    ax1.set_ylabel('$%s$' % ycord[1])
    plt.tight_layout()
    plt.show()