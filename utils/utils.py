import pandas as pd
import numpy as np



def get_stations_constraints(df, stations_sizes):
    x_min_max = [-1, 1]
    y_min_max = [1, 1]
    z_min_max = [32, 194]
    for station_rects in stations_sizes:
        for station_rect in station_rects:
            station_center_xy = station_rect[0:2]
            station_width = station_rect[2]
            station_height = station_rect[3]
            left_x = station_center_xy[0] - station_width / 2
            left_y = station_center_xy[1] - station_height / 2
            right_x = station_center_xy[0] + station_width / 2
            right_y = station_center_xy[1] + station_height / 2
            if left_x < x_min_max[0]:
                x_min_max[0] = left_x
            if left_y < y_min_max[0]:
                y_min_max[0] = left_y
            if right_y > y_min_max[1]:
                y_min_max[1] = right_y
            if right_x > x_min_max[1]:
                x_min_max[1] = right_x
    assert df.x.min() > x_min_max[0] and df.x.max() < x_min_max[1] and \
           df.y.min() > y_min_max[0] and df.y.max() < y_min_max[1] and \
           df.z.min() > z_min_max[0] and df.z.max() < z_min_max[1]
    return x_min_max, y_min_max, z_min_max

def normalize_convert_to_r_phi_z(df, stations_sizes, convert_to_polar=False):

    x_min_max, y_min_max, z_min_max = get_stations_constraints(df, stations_sizes)

    x_min, x_max = x_min_max
    y_min, y_max = y_min_max
    z_min, z_max = z_min_max
    x_norm = 2 * (df.x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (df.y - y_min) / (y_max - y_min) - 1
    z_norm = (df.z - z_min) / (z_max - z_min)
    if convert_to_polar:
        df = df.assign(y_old=df.y, x_old=df.x, z_old=df.z)
        del df['z']
        df = df.assign(x=x_norm, y=y_norm, z=z_norm)
        r = np.sqrt(df.y ** 2 + df.x ** 2)
        phi = np.arctan2(df.y, df.x)
        df = df.assign(r=r, phi=phi, z=df.z)
    else:
        df = df.assign(x=x_norm, y=y_norm, z=z_norm)

    return df

def dropBroken(df, preserve_fakes, drop_full_tracks):
    assert df.event.nunique() == 1
    def drop(x):
        return np.all(np.diff(x.station.values) == 1.) and x.station.values[0] == 0.

    if not preserve_fakes:
        df = df[df.track != -1]

    ret = df.groupby('track', as_index=False).filter(
        lambda x: drop(x) or preserve_fakes and x.track.values[0] == -1
        # if preserve_fakes == False, we are leaving only matched events, no fakes
    )
    if drop_full_tracks:
        ret = ret.groupby('track', as_index=False).filter(
        lambda x: x.station.nunique() < 6 or preserve_fakes and x.track.values[0] == -1)
    return ret

def parse_df(config_df):
    df_path = config_df['df_path']
    if config_df['read_only_first_lines']:
        nrows = config_df['read_only_first_lines']
        return pd.read_csv(df_path, encoding='utf-8', sep='\t', nrows=nrows)
    return pd.read_csv(df_path, encoding='utf-8', sep='\t')

def get_events_df(config_df, hits_df, preserve_fakes=True, drop_full_tracks=False):
    eventIdsArr = config_df['event_ids']

    def parseSingleArrArg(arrArg):
        if '..' in arrArg:
            args = arrArg.split('..')
            assert len(args) == 2 and "It should have form '%num%..%num%' ."
            return np.arange(int(args[0]), int(args[1])), False
        if ':' in arrArg:
            return -1, True
        return [int(arrArg)], False

    res = np.array([])
    for elem in eventIdsArr:
        toAppend, is_all = parseSingleArrArg(elem)
        if is_all:
            return hits_df
        res = np.append(res, toAppend)

    hits = hits_df[hits_df.event.isin(res)].copy()
    if config_df['drop_broken_tracks']:
        for id, event in hits.groupby('event'):
            ev_hits = dropBroken(event, preserve_fakes=preserve_fakes, drop_full_tracks=drop_full_tracks)
            hits.loc[hits.event == id] = ev_hits
    else:
        assert preserve_fakes and drop_full_tracks and "Error, you are not dropping broken but attempting to 'drop_full_tracks' or 'preserve_fakes'"
    if config_df['convert_to_polar'] or config_df['normalize']:
        hits = normalize_convert_to_r_phi_z(hits, config_df['stations_sizes'], config_df['convert_to_polar'])
        pass
    return hits


