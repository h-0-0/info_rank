import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['nyu_v2_13'],
        'model': ['ResNet50', 'ResNet101'],
        'num_epochs': [100],
        'batch_size': [8],
        'patience': [5],
        'optimizer': ['SGD'],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'temperature': [1], 
        'output_dim': [2048],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [1e-1, 1e-2, 1e-3],
        'temperature' : [1],
        'output_dim': [2048],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    to_search.update(info_critic)

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_nyu.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)