import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': ['written_spoken_digits'],
        'model': ['MNIST_Audio_CNN3'],
        'num_epochs': [200],
        'batch_size': [256],
        'est': ['SimCLR'],
        'patience': [20],
        'temperature': [1, 0.1, 0.01],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)