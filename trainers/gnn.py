"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time

# Externals
import torch
import numpy as np
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model
from sklearn import metrics
import sys

from sys import stdout
class GNNTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, cfg_trainer,  **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        self.real_weight = cfg_trainer['real_weight']
        self.fake_weight = cfg_trainer['fake_weight']

    def build_model(self, name='gnn_segment_classifier',
                    optimizer='Adam', learning_rate=0.001,
                    loss_func='binary_cross_entropy', **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)

        # TODO: LR scaling
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=learning_rate)
        # Functional loss functions
        self.loss_func = getattr(nn.functional, loss_func)

#    @profile
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        start_time = time.time()
        # Loop over training batches
        stdout.write('\n\n========\n\n')
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch_target * self.real_weight
            batch_weights_fake = (1 - batch_target) * self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target, weight=batch_weights)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            stdout.write('\r  batch %i, loss %f' %(i, batch_loss.item()))

        stdout.write('\n\n========\n\n')
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.info(' \nProcessed %i batches for %d sec' % ((i + 1), summary['train_time']))
        self.logger.info('  Training loss: %f' % summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        start_time = time.time()
        # Loop over batches
        prec = 0
        rec = 0
        acc = 0
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            sum_loss += self.loss_func(batch_output, batch_target).item()
            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            arr1 = ((batch_target > 0.5).cpu().numpy() != 0)
            arr2 = ((batch_output > 0.5).cpu().numpy() != 0)
            prec += np.average([metrics.precision_score(arr1, arr2) for (arr1, arr2) in zip(arr1, arr2)])
            rec += np.average([metrics.recall_score(arr1, arr2) for (arr1, arr2) in zip(arr1, arr2)])
            acc += np.average([metrics.accuracy_score(arr1, arr2) for (arr1, arr2) in zip(arr1, arr2)])
            #self.logger.debug(' batch %d prec %f', )
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total

        summary['valid_prec_t'] = prec /  (i + 1)
        summary['valid_rec_t'] = rec /  (i + 1)
        summary['valid_acc_t'] = acc /  (i + 1)
        self.logger.info('\n Validation: Processed %d samples in %d batches for %f sec\n' % (len(data_loader.sampler), i + 1, summary['valid_time']))
        str_to_write = "\nValidation loss: %f acc: %f valid_prec_t: %f valid_rec_t: %f valid_acc_t: %f'\n" % (summary['valid_loss'], summary['valid_acc'], summary['valid_prec_t'], summary['valid_rec_t'], summary['valid_acc_t'])
        self.logger.info(str_to_write)
        return summary

def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
