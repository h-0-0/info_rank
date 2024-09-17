import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['mosi'],
        'model': ['MosiFusion'],
        'num_epochs': [200],
        'batch_size': [32],
        'patience': [15],
        'optimizer': ['SGD'],
        'grad_clip': ['None'],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
        'temperature': [10, 1, 0.1], 
        'output_dim': [64, 128, 256, 512],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
        'temperature' : [1],
        'output_dim': [64, 128, 256, 512],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    to_search.update(info_critic)

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_mosi.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)