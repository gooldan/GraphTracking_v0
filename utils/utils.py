import pandas as pd
import numpy as np

def dropBroken(df, preserve_fakes, drop_full_tracks):
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
            assert len(args) == 3 and "It should have form '%num%..%num%' ."
            return np.arange(int(args[0]), int(args[2]))
        if ':' in arrArg:
            return -1
        return [int(arrArg)]

    res = np.array([])
    for elem in eventIdsArr:
        toAppend = parseSingleArrArg(elem)
        if toAppend == -1:
            return hits_df
        res = np.append(res, toAppend)

    hits = hits_df[hits_df.event.isin(res)]
    if config_df['drop_broken_tracks']:
        hits = dropBroken(hits, preserve_fakes=preserve_fakes, drop_full_tracks=drop_full_tracks)
    else:
        assert preserve_fakes and drop_full_tracks and "Error, you are not dropping broken but attempting to 'drop_full_tracks' or 'preserve_fakes'"
    return hits


