import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['written_spoken_digits'],
        'model': ['FusionModel', 'AudioOnly', 'ImageOnly'],
        'num_epochs': [200],
        'batch_size': [128],
        'patience': [10],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [5e-5, 1e-4, 5e-5, 1e-5],
        'temperature': [100, 10, 1],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [5e-1, 2e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'temperature' : [1],
    }
    supervised = {
        'est': ['supervised'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'temperature' : [1],
    }

    # Join dictionary
    # to_search.update(SimCLR)
    # to_search.update(info_critic)
    to_search.update(supervised)

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)