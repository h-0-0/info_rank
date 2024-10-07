import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['mosi'],
        'model': ['MosiFusion'],
        'num_epochs': [300],
        'batch_size': [128],
        'patience': [20],

        'optimizer': ['SGD'],

        'eval_num_epochs': [300],
        'eval_lr': [1e-1],
        'eval_patience': [30],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [5e-2, 1e-2, 1e-3, 1e-4],
        'temperature': [1], 
        'output_dim': [64, 128, 256, 512, 1024],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [5e-1, 1e-1, 5e-2, 1e-2],
        'temperature' : [1],
        'output_dim': [64, 128, 256, 512, 1024],
    }
    supervised = {
        'est': ['supervised'],
        'learning_rate': [5e-1, 1e-1, 1e-2, 5e-3],
        'output_dim': [64, 128, 256, 512, 1024],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    # to_search.update(info_critic)
    to_search.update(supervised)

    # Trying shallower thinner GRUs, no dropout in fusion layer, smaller output dim and thinned lr
    # Now trying better range of lrs and increased unsupervised num_epochs and batch size, also wide range of output dims
    # Now added layer norm, previous 7acc 19%
    # Changed regression model so that it halves from input dimension, didnt work
    # Now adding dropout to regression model
    # Focusing on boosting supervised performance 
    # MOSEI working well so it may be that backbone bad, going to gradually make it like MOSEIs
    # Started by removing dropout from regression model and using MOSEIs fusion model
    # Performance roughly the same, now added batchnorm to projection, 25.7 test, 47.8 train
    # Trying hidden size 64, num layers 1, Got 26% test, 43.5% train
    # Trying hidden size 128, num layers 1, Got 28% test, 58% train
    # Trying hidden size 128, num layers 2, Got 25% test, 43.9% train
    # Trying hidden size 256, num layers 2, Got 29.5% test, 59.9% train
    # Trying hidden size 512, num layers 2, Got 29.9% test, 62.1% train
    # Trying hidden size 128, num layers 3, Got 27.7 test, 52% train
    # Trying hidden size 256, num layers 3, Got 29.3% test, 58.3% train
    # Trying hidden size 512, num layers 3, Got 29.2% test, 65% train
    

    # Try hidden size 512, num layers 2, 
        # Stripped back fusion MLP 
            # Got 30.9 % test, 83.1% train
        # Verbose fusion MLP with extra layer 
            # Got 29.3% test, 50.5% train
        # Stripped back with extra layer 
            # Got 29% test, 70% train

    # Using hidden size 512, num layers 2 and stripped back fusion MLP, Got 30.9% test, 83.1% train
        # With MAE loss, Got 28.3% test, 53.5% train
        # With Huber loss 0.5, Got 29.5% test, 69.7% train
        # With Huber loss 1.0, Got 31.3% test, 91% train
        # With Huber loss 2, Got 31.6% test, 89.4% train
        # With Huber loss 5, Got 30.0% test, 76.2% train

    # Then using validation set

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_mosi.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)