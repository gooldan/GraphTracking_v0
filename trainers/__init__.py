"""
Python module for holding our PyTorch trainers.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

from .gnn import GNNTrainer

def get_trainer(name, **trainer_args):
    return GNNTrainer(**trainer_args)
