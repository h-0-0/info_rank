import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['nyu_v2_13'],
        'model':  ['ESANet_18'], 
        'num_epochs': [500],
        'patience': [500],
        'optimizer': ['Adam'],

        'eval_num_epochs': [200],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [1e-2, 1e-3, 1e-4], #1e-4 only one without immediate failure
        'temperature': [1], 
        'output_dim': [2048],
        'batch_size': [98],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'temperature' : [1],
        'output_dim': [2048],
        'batch_size': [98],
    }
    supervised = {
        'est': ['supervised'],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'temperature' : [1],
        'output_dim': [2048],
        'batch_size': [48],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    # to_search.update(info_critic)
    to_search.update(supervised)

    # Trying fix unsup learn again with some more lr's, if it doesn't work then try adam
    # Then the problem of eval_train

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_nyu.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)