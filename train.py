
from utils.config_reader import ConfigReader

import os
import logging
import numpy as np
import sys
# Locals
from datasets import get_data_loaders
from trainers.gnn import GNNTrainer
import argparse
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--device', default='cpu')
    add_arg('--config', default='configs/cgem_train_config.yaml')
    add_arg('--out_dir_colab', default='')
    return parser.parse_args()

if __name__ == '__main__':

    args_in = parse_args()

    reader = ConfigReader(args_in.config)
    cfg = reader.cfg
    config_train = cfg['train']

    if args_in.out_dir_colab != "":
        out_dir = args_in.out_dir_colab #"/gdrive/My Drive/graph/result_colab"
    else:
        out_dir = config_train['result_dir']

    logfilename = '%(asctime)s %(levelname)s %(message)s'

    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(filename=(out_dir + "/process_log.log"), level=logging.INFO,
                        format=logfilename)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("=" * 20)
    logging.info("=" * 20)
    logging.info("=" * 20)
    logging.info("\nStarting...")
    logging.info("Config:")
    logging.info(cfg)

    train_data_loader, valid_data_loader = get_data_loaders(cfg['train'])
    logging.info('Loaded %g training samples', len(train_data_loader.dataset))
    if valid_data_loader is not None:
        logging.info('Loaded %g validation samples', len(valid_data_loader.dataset))

    # Load the trainer
    trainer = GNNTrainer( cfg['trainer'], output_dir=out_dir,
                          device=args_in.device, train_loader=train_data_loader)
    # Build the model and optimizer
    trainer.build_model(**cfg.get('model', {}))
    trainer.print_model_summary()
    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            n_epochs=config_train['n_epochs'])
    trainer.write_summaries()

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %g s rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = np.mean(summary['valid_time'])
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)


    pass