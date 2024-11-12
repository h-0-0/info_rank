import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['mosi'],
        'batch_size': [128],

        'optimizer': ['Adam'],
    }
    SimCLR = {
        'est': ['SimCLR'],
        'model': ['MosiFusion', 'MosiFusionAttention'],
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'temperature': [1], 
        'output_dim': [64, 128, 256],
        'num_epochs': [300],
        'patience': [300],
        'eval_num_epochs': [200],
        'eval_patience': [50],
    }
    SimCLR_MISA = {
        'est': ['SimCLR'],
        'model': ['MosiMISA'],
        'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4],
        'temperature': [1], 
        'output_dim': [64, 128, 256],
        'num_epochs': [400],
        'patience': [400],
        'eval_num_epochs': [200],
        'eval_patience': [50],
    }
    info_critic = {
        'est': ['info_critic'],
        'model': ['MosiFusion', 'MosiFusionAttention'],
        'learning_rate': [5e-2, 1e-2, 5e-3],
        'temperature' : [1],
        'output_dim': [64, 128, 256],
        'num_epochs': [300],
        'patience': [300],
        'eval_num_epochs': [200],
        'eval_patience': [50],
    }
    info_critic_MISA = {
        'est': ['info_critic'],
        'model': ['MosiMISA'],
        'learning_rate': [1e-1, 1e-2, 1e-4, 1e-6, 1e-8],
        'temperature' : [1],
        'output_dim': [32, 64, 128, 256],
        'num_epochs': [300],
        'patience': [300],
        'eval_num_epochs': [200],
        'eval_patience': [50],
    }
    supervised = {
        'est': ['supervised'],
        'model': ['MosiFusion'],
        'learning_rate': [5e-3, 1e-3, 5e-4],
        'output_dim': [128], #16, 32, 64, 256
        'num_epochs': [250, 500, 1000],
        'patience': [1000],
        'eval_lr': [0.005],
        'eval_num_epochs': [200],
        'eval_patience': [200],
    }
    supervised_MISA = {
        'est': ['supervised'],
        'model': ['MosiMISA'],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'output_dim': [128], #16, 32, 64, 256
        'num_epochs': [250, 500, 1000],
        'patience': [1000],
        'eval_lr': ['None', 0.005],
        'eval_num_epochs': [200],
        'eval_patience': [200],
        'grad_clip': [1.0, 'None'],
    }
    supervised_Attention = {
        'est': ['supervised'],
        'model': ['MosiFusionAttention'],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'output_dim': [128], #16, 32, 64, 256
        'num_epochs': [250, 500, 1000],
        'patience': [1000],
        'eval_lr': [0.005],
        'eval_num_epochs': [200],
        'eval_patience': [200],
    }
    supervised_MULT = {
        'est': ['supervised'],
        'model': ['MosiMULT'],
        'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        'output_dim': [120, 180, 240],
        'num_epochs': [50, 100, 150],
        'patience': [200],
        'grad_clip': [0.6, 0.8, 1.0, 'None'],
        'scheduler' : ['ReduceLROnPlateau', 'None'],
        # 'eval_lr': ['None', 0.005, 1e-8],
        # 'eval_num_epochs': [200],
        # 'eval_patience': [200],
    }



    supervised_MosiTrans = {
        'est': ['supervised'],
        'model': ['MosiTransformer'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        # 'output_dim': [120, 180, 240],
        'num_epochs': [100],
        'patience': [100],
        'grad_clip': [0.6, 0.8, 1.0, 'None'],
        'scheduler' : ['ReduceLROnPlateau', 'None'],
        # 'eval_lr': ['None', 0.005, 1e-8],
        # 'eval_num_epochs': [200],
        # 'eval_patience': [200],
    }
    info_critic_MosiTrans = {
        'est': ['info_critic'],
        'model': ['MosiTransformer'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        # 'output_dim': [120, 180, 240],
        'num_epochs': [100],
        'patience': [100],
        'grad_clip': [0.6, 0.8, 1.0, 'None'],
        # 'scheduler' : ['ReduceLROnPlateau', 'None'],
        # 'eval_lr': ['None', 0.005, 1e-8],
        # 'eval_num_epochs': [200],
        # 'eval_patience': [200],
    }
    SimCLR_MosiTrans = {
        'est': ['SimCLR'],
        'model': ['MosiTransformer'],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        # 'output_dim': [120, 180, 240],
        'num_epochs': [100],
        'patience': [100],
        'grad_clip': [0.6, 0.8, 1.0, 'None'],
        # 'scheduler' : ['ReduceLROnPlateau', 'None'],
        # 'eval_lr': ['None', 0.005, 1e-8],
        # 'eval_num_epochs': [200],
        # 'eval_patience': [200],
    }

    # Join dictionary
    # to_search.update(SimCLR) 
    # to_search.update(SimCLR_MISA) 
    # to_search.update(info_critic)
    # to_search.update(info_critic_MISA)
    # to_search.update(supervised)
    # to_search.update(supervised_MISA)
    # to_search.update(supervised_Attention)
    # to_search.update(supervised_MULT)
    # to_search.update(supervised_MosiTrans)
    # to_search.update(info_critic_MosiTrans)
    to_search.update(SimCLR_MosiTrans)

    # Trying shallower thinner GRUs, no dropout in fusion layer, smaller output dim and thinned lr
    # Now trying better range of lrs and increased unsupervised num_epochs and batch size, also wide range of output dims
    # Now added layer norm, previous 7acc 19%
    # Changed regression model so that it halves from input dimension, didnt work
    # Now adding dropout to regression model
    # Focusing on boosting supervised performance 
    # MOSEI working well so it may be that backbone bad, going to gradually make it like MOSEIs
    # Started by removing dropout from regression model and using MOSEIs fusion model
    # Performance roughly the same, now added batchnorm to projection, 25.7 test, 47.8 train
    # Trying hidden size 64, num layers 1, Got 26% test, 43.5% train
    # Trying hidden size 128, num layers 1, Got 28% test, 58% train
    # Trying hidden size 128, num layers 2, Got 25% test, 43.9% train
    # Trying hidden size 256, num layers 2, Got 29.5% test, 59.9% train
    # Trying hidden size 512, num layers 2, Got 29.9% test, 62.1% train
    # Trying hidden size 128, num layers 3, Got 27.7 test, 52% train
    # Trying hidden size 256, num layers 3, Got 29.3% test, 58.3% train
    # Trying hidden size 512, num layers 3, Got 29.2% test, 65% train
    

    # Try hidden size 512, num layers 2, 
        # Stripped back fusion MLP 
            # Got 30.9 % test, 83.1% train
        # Verbose fusion MLP with extra layer 
            # Got 29.3% test, 50.5% train
        # Stripped back with extra layer 
            # Got 29% test, 70% train

    # Using hidden size 512, num layers 2 and stripped back fusion MLP, Got 30.9% test, 83.1% train
        # With MAE loss, Got 28.3% test, 53.5% train
        # With Huber loss 0.5, Got 29.5% test, 69.7% train
        # With Huber loss 1.0, Got 31.3% test, 91% train
        # With Huber loss 2, Got 31.6% test, 89.4% train
        # With Huber loss 5, Got 30.0% test, 76.2% train

    # Then using validation set

    # MosiFusion
        # Supervised - getting 7acc test 30.3%, train 84.5%
        # SimCLR - getting 7acc test 21%, train 82%
        # InfoCritic - getting 7acc test 18%, train 26%
    # MosiMISA
        # Supervised - getting 7acc test 28.6%, train 88.6%
        # SimCLR - getting 7acc test 20.1%, train 60.8%
        # InfoCritic - getting 7acc test 17.4%, train 26.6%

    # Now trying bigger patience for hopefully smaller unsup losses
    # sigma =0.1, iid_masks=False
    # Now trying varying hidden size with output dim in MISA
    # Also trying MosiFusionAttention

    # Next
    # Try linear Regression layer
    # Try sigma=0.2

    grid_info_rank = slune.searchers.SearcherGrid(to_search)

    script_path = 'run_grid.py'
    template_path = 'compute_spec_mosi.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)