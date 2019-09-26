"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader

from . import hitgraphs


def preprocess_data(hit_graph_dataset, stat_df, filenames):

    pass

def get_data_loaders(train_cfg):
    batch_size = train_cfg['batch_size']
    input_dir = train_cfg['input_dir']
    n_train = train_cfg['n_train']
    n_valid = train_cfg['n_valid']
    preproc_df_path = None
    if train_cfg['preprocess_dataset']:
        preproc_df_path = train_cfg['input_dir'] + '/' + train_cfg['preprocess_dataset']['filename']
    train_dataset, valid_dataset = hitgraphs.get_datasets(input_dir, n_train, n_valid, preproc_df_path)
    collate_fn = hitgraphs.collate_fn


    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn)
    train_data_loader = DataLoader(train_dataset, **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
