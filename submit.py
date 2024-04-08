import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': ['written_spoken_digits'],
        'model': ['FusionModel'],
        'num_epochs': [200],
        'batch_size': [256],
        'est': ['info_rank'],
        'patience': [10],
        'temperature': [0.1],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)