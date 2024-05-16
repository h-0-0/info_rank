import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': ['written_spoken_digits'],
        'model': [
            'FusionModel', 'AudioOnly', 'ImageOnly',
            'FusionModelStrict', 'AudioOnlyStrict', 'ImageOnlyStrict',
            'FusionModelStrictShallow', 'AudioOnlyStrictShallow', 'ImageOnlyStrictShallow',
            ],
        'num_epochs': [200],
        'batch_size': [256],
        'est': ['info_rank_plus', 'info_rank', 'SimCLR'],
        'patience': [20],
        'temperature': [1, 0.1, 0.01],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)