import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': ['written_spoken_digits'],
        'est': ['SimCLR'],  #'prob_loss', 'decomposed_loss', 'info_rank', 'info_rank_plus' #'SimCLR',  'info_critic', 'info_critic_plus'
        'model': [
            'FusionModel', 'AudioOnly', 'ImageOnly',
            # 'StrictFusionModel', 'StrictAudioOnly', 'StrictImageOnly',
            # 'ShallowStrictFusionModel', 'ShallowStrictAudioOnly', 'ShallowStrictImageOnly',
            ],
        'num_epochs': [1, 25, 50, 100],
        'batch_size': [128],
        'patience': [-1],
        'temperature': [1, 0.01, 0.0001],
        # 'learning_rate': [5e-1, 2e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], #info_critic
        'learning_rate': [5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5], #SimCLR
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)