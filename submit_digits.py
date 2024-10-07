import slune

if  __name__ == "__main__":
    # to_search = {
    #     'benchmark': ['written_spoken_digits'],
    #     'model': ['FusionModel', 'AudioOnly', 'ImageOnly'],
    #     'num_epochs': [200],
    #     'batch_size': [128],
    #     'patience': [15],
    #     'optimizer': ['SGD'],
    # }
    # SimCLR = {
    #     'est': ['SimCLR'],
    #     'learning_rate': [2e-4], 
    #     'temperature': [10], 
    #     'output_dim': [64, 32, 16, 8, 4, 2],
    # }
    # info_critic = {
    #     'est': ['info_critic'],
    #     'learning_rate': [5e-2],
    #     'temperature' : [1],
    #     'output_dim': [64], #, 32, 16, 8, 4, 2],
    # }
    # info_critic_plus = {
    #     'est': ['info_critic_plus'],
    #     'learning_rate': [1e-2],
    #     'temperature' : [1],
    #     'output_dim': [64, 32, 16, 8, 4, 2],
    # }
    # prob_loss = {
    #     'est': ['prob_loss'],
    #     'learning_rate': [5e-2, 1e-2, 5e-3, 1e-3],
    #     'temperature' : [1],
    #     'output_dim': [64],
    # }
    # decomposed_loss = {
    #     'est': ['decomposed_loss'],
    #     'learning_rate': [1e-7],
    #     'temperature' : [1],
    #     'output_dim': [64, 32, 16, 8, 4, 2],
    # }
    # supervised = {
    #     'est': ['supervised'],
    #     'learning_rate': [1e-3],
    #     'temperature' : [1],
    #     'output_dim': [64, 32, 16, 8, 4, 2],
    # }

    # Join dictionary
    # to_search.update(SimCLR) 
    # to_search.update(info_critic) # Make learing rate much smaller for good performance on AudioOnly
    # to_search.update(info_critic_plus) # Make learing rate much smaller for good performance on AudioOnly
    # to_search.update(prob_loss) # Make learing rate much smaller for good performance on AudioOnly (1e-06)
    # to_search.update(decomposed_loss)
    # to_search.update(supervised) 
    
    base_config = {
        'benchmark': ['written_spoken_digits'],
        'num_epochs': [200],
        'batch_size': [128],
        'patience': [15],
        'optimizer': ['SGD'],
    }
    # Expriments
    # ----------------- #
    info_critic_fusion = {
        'est': ['info_critic'],
        'model': ['FusionModel'],
        'learning_rate': [5e-2],
        'temperature' : [1],
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    info_critic_audio = {
        'est': ['info_critic'],
        'model': ['AudioOnly'],
        'learning_rate': [1e-5],
        'temperature' : [1],
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    info_critic_image = {
        'est': ['info_critic'],
        'model': ['ImageOnly'],
        'learning_rate': [5e-2],
        'temperature' : [1],
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'learning_rate': [2e-4], 
        'temperature': [10], 
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    supervised = {
        'est': ['supervised'],
        'model': ['FusionModel', 'AudioOnly', 'ImageOnly'],
        'learning_rate': [1e-3],
        'temperature' : [1],
        'output_dim': [64, 32, 16, 8, 4, 2],
    }
    for config in [info_critic_fusion, info_critic_audio, info_critic_image, SimCLR, supervised]:
        to_search = base_config.copy()
        to_search.update(config)
        grid_info_rank = slune.searchers.SearcherGrid(to_search)

        script_path = 'run_grid.py'
        template_path = 'compute_spec_digits.sh'
        slune.sbatchit(script_path, template_path, grid_info_rank)