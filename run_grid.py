import argparse
import slune
from utils import dict_to_ls, ls_to_dict
import os
import matplotlib.pyplot as plt
from train import train
import numpy as np
import torch

if  __name__ == "__main__":
    # Parse input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Name of the benchmark to use', default="written_spoken_digits")
    parser.add_argument('--model', type=str, help='Model to use', default="FusionModel")
    parser.add_argument('--learning_rate', type=float, help='Learning rate to use', default=0.01)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for', default=200)
    parser.add_argument('--batch_size', type=int, help='Batch size to use', default=256)
    parser.add_argument('--est', type=str, help='Estimator to use', default="info_rank")
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=10)
    parser.add_argument('--temperature', type=float, help='Temperature for NCE', default=0.1)
    args = parser.parse_args()

    config = {
        'benchmark': args.benchmark,
        'model': args.model,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'est': args.est,
        'patience': args.patience,
        'temperature': args.temperature,
    }

    to_grid ={
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    }
    
    grid = slune.searchers.SearcherGrid(to_grid)
    for g in grid:
        # Add the net to the config
        g = ls_to_dict(g)
        config = {**config, **g}
        # Train the model
        losses, model = train(**config)
        # Create save location using slune
        saver = slune.get_csv_saver(root_dir='results')
        print("path: ", config)
        path = os.path.dirname(saver.get_path(dict_to_ls(**config)))

        # Plot the loss
        plt.figure()
        plt.plot(np.arange(len(losses)), np.array(losses))
        plt.savefig(os.path.join(path, "loss.png"))
        plt.close()
        
        # Clear GPU memory
        torch.cuda.empty_cache()