# import parent directory 
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from train import train


config = {
    'benchmark': 'written_spoken_digits',
    'model': 'ImageOnly',
    'learning_rate': 0.05,
    'num_epochs': 1,
    'batch_size': 128,
    'est': 'info_critic',
    'patience': -1,
    'temperature': 1,
    'output_dim': 64,
}
model = train(**config)