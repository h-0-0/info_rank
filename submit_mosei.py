import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['mosei'],
        'model': ['MoseiFusion'],
        'num_epochs': [100],
        'batch_size': [16],
        'patience': [15],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
        'temperature': [10, 1, 0.1], 
        'output_dim': [64, 128, 256],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
        'temperature' : [1],
        'output_dim': [64, 128, 256],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    to_search.update(info_critic)

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_mosei.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)