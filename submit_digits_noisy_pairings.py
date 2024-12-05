import slune

if  __name__ == "__main__":    
    base_config = {
        'benchmark': ['written_spoken_digits_noisy_pairing'],
        'sigma': [0.01, 0.02, 0.04, 0.08, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6],
        'num_epochs': [200],
        'batch_size': [128],
        'patience': [15],
        'optimizer': ['SGD'],
    }
    # Expriments
    # ----------------- #
    info_critic = {
        'est': ['info_critic'],
        'model': ['FusionModel'],
        'learning_rate': [5e-2], 
        'temperature' : [1],
        'output_dim': [64],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'model': ['FusionModel'],
        'learning_rate': [2e-4],  
        'temperature': [10], 
        'output_dim': [64],
    }
    supervised = {
        'est': ['supervised'],
        'model': ['FusionModel'],
        'learning_rate': [1e-3],
        'temperature' : [1],
        'output_dim': [64],
    }
    for config in [info_critic, SimCLR, supervised]:
        to_search = base_config.copy()
        to_search.update(config)
        grid_info_rank = slune.searchers.SearcherGrid(to_search)

        script_path = 'run_grid.py'
        template_path = 'compute_spec_digits.sh'
        slune.sbatchit(script_path, template_path, grid_info_rank)