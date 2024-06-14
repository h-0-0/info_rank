import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': ['written_spoken_digits'],
        'model': [
            'FusionModel', 'AudioOnly', 'ImageOnly',
            # 'StrictFusionModel', 'StrictAudioOnly', 'StrictImageOnly',
            # 'ShallowStrictFusionModel', 'ShallowStrictAudioOnly', 'ShallowStrictImageOnly',
            ],
        'num_epochs': [0, 0.1, 0.2, 0.4, 0.8],
        'batch_size': [128],
        'est': ['SimCLR', 'info_critic'],  #'prob_loss', 'decomposed_loss', 'info_rank', 'info_rank_plus' #'SimCLR',  'info_critic', 'info_critic_plus'
        'patience': [-1],
        'temperature': [1],
        'learning_rate': [1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)