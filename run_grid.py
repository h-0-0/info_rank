import argparse
import slune
from utils import dict_to_ls, ls_to_dict
import os
import matplotlib.pyplot as plt
from train import train
import numpy as np
import torch
from typing import Optional

def none_or_int(value):
    if value is None:
        return None
    elif value == 'None':   
        return None
    try:
        i = int(value)
        return i
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")
    
def none_or_float(value):
    if value is None:
        return None
    elif value == 'None':
        return None
    try:
        f = float(value)
        return f
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a float")

def none_or_str(value):
    if value is None:
        return None
    elif value == 'None':
        return None
    return value

if  __name__ == "__main__":
    # Parse input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Name of the benchmark to use', default="written_spoken_digits")
    parser.add_argument('--model', type=str, help='Model to use', default="FusionModel")
    parser.add_argument('--learning_rate', type=float, help='Learning rate to use', default=0.01)
    parser.add_argument('--num_epochs', type=float, help='Number of epochs to train for', default=200)
    parser.add_argument('--batch_size', type=int, help='Batch size to use', default=256)
    parser.add_argument('--est', type=str, help='Estimator to use', default="info_rank")
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=10)
    parser.add_argument('--temperature', type=float, help='Temperature for NCE', default=1.0)
    parser.add_argument('--output_dim', type=int, help='Output dimension of the model', default=64)
    parser.add_argument('--optimizer', type=str, help='Optimizer to use', default="SGD")
    parser.add_argument('--eval_lr', type=none_or_float, help='Learning rate for evaluation training, if None then uses 0.1 * batch_size / 256', default=None)
    parser.add_argument('--eval_num_epochs', type=none_or_int, help='Number of epochs to train for evaluation training, if None then uses 50', default=None)
    parser.add_argument('--eval_patience', type=none_or_int, help='Patience for early stopping during evaluation training, if None then no early stopping', default=None)
    parser.add_argument('--sigma', type=float, help='Variance of the gaussian noise to add to the data', default=0.1)
    parser.add_argument('--grad_clip', type=none_or_float, help='Gradient clipping value', default=None)
    parser.add_argument('--scheduler', type=none_or_str, help='Scheduler to use', default=None)
    args = parser.parse_args()

    config = {
        'benchmark': [args.benchmark],
        'est': [args.est],
        'model': [args.model],
        'output_dim': [args.output_dim],
        'num_epochs': [args.num_epochs],
        'batch_size': [args.batch_size],
        'patience': [args.patience],
        'temperature': [args.temperature],
        'learning_rate': [args.learning_rate],
        'optimizer': [args.optimizer],
    }
    if args.eval_lr is not None:
        config['eval_lr'] = [args.eval_lr]
    if args.eval_num_epochs is not None:
        config['eval_num_epochs'] = [args.eval_num_epochs]
    if args.eval_patience is not None:
        config['eval_patience'] = [args.eval_patience]
    if args.grad_clip is not None:
        config['grad_clip'] = [args.grad_clip]
    if args.scheduler is not None:
        config['scheduler'] = [args.scheduler]

    print("Searching Over: ", config, flush=True)
    if args.benchmark == 'written_spoken_digits':
        grid = slune.searchers.SearcherGrid(config, runs=10)
    elif (args.benchmark == 'written_spoken_digits_weak_image') or (args.benchmark == 'written_spoken_digits_weak_audio') or (args.benchmark == 'written_spoken_digits_noisy_pairing'):
        config['sigma'] = [args.sigma]
        grid = slune.searchers.SearcherGrid(config, runs=20)
    else:
        grid = slune.searchers.SearcherGrid(config, runs=1)
    # grid.check_existing_runs(slune.get_csv_saver(root_dir='results'))
    for g in grid:
        # Add the net to the config
        # Train the model
        model, linear_classifier = train(**g)
        # Create save location using slune
        saver = slune.get_csv_saver(root_dir='results', params=g)
        print("path: ", g, flush=True)
        path = os.path.dirname(saver.getset_current_path())
        
        # Clear GPU memory
        torch.cuda.empty_cache()