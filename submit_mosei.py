import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['mosei'],
        'model': ['MoseiFusion'],
        'batch_size': [128],
        'patience': [10],
        'optimizer': ['SGD'], 

        'eval_num_epochs': [500],
        'eval_patience': [20],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [1e-1, 1e-2, 1e-3],
        'temperature': [1], 
        'output_dim': [128, 512, 2048],
        'num_epochs': [100],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'temperature' : [1],
        'output_dim': [128, 512, 2048],
        'num_epochs': [100],
    }
    supervised = {
        'est': ['supervised'],
        'learning_rate': [1e-1, 1e-2],
        'output_dim': [128, 512, 2048],
        'num_epochs': [100],
        'eval_lr': [None, 1e-3, 1e-4],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    # to_search.update(info_critic)
    to_search.update(supervised)

    # info_critic unsupervised doesn's seem to be working
    # eval train looks better than MOSI, but accuracy achieved same for random as for trained
    # Trying with halved regression and laregr bathc size
    # trying even larger batch size and supervised with Dropout in regression
    # now trying without dropout in regression
    # Added batch norm to other projectors (currently only image)

    # Add layer to regression MLP
    # Train accuracy for info_critic concerning
    # Might need to improve augmentations

    # With improved aug:
        # supervised test 7acc 45.6%, train 91.1%
        # info_critic test 7acc 40.89%, train 44.15%
        # SimCLR test 7acc 40.36%, train 46.1%

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_mosei.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)