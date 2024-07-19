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
        'learning_rate': [1e-4], #try lr = 2e-4, 5e-5
        'temperature': [10], #Try temp=5
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    info_critic = {
        'est': ['info_critic'],
        'learning_rate': [5e-2],
        'temperature' : [1],
        'output_dim': [64], #, 32, 16, 8, 4, 2],
    }
    info_critic_plus = {
        'est': ['info_critic_plus'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'temperature' : [1],
        'output_dim': [64],
    }
    supervised = {
        'est': ['supervised'],
        'learning_rate': [1e-3],
        'temperature' : [1],
        'output_dim': [64, 32, 16, 8, 4, 2],
    }

    # Join dictionary
    # to_search.update(SimCLR)
    to_search.update(info_critic)
    # to_search.update(info_critic_plus)
    # to_search.update(supervised)

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)