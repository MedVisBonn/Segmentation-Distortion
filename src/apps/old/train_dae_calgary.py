import os, sys
import collections
import time
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import wandb
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

sys.path.append('..')
from dataset import CalgaryCampinasDataset
from model.unet import UNet2D
# from model.ae import AE
from model.dae import AugResDAE
from trainer.ae_trainerV2 import AETrainerCalgaryV2
from model.wrapper import ModelAdapter
from losses import (
    CalgaryCriterionAE, 
    SampleDice,
    UnetDice
)
from data_utils import (
    volume_collate,
    Transforms,
    MultiImageSingleViewDataLoader
)



def main(args):

    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft()
        if arg in ['--residual']:
            residual = False if int(arguments.popleft()) == 0 else True
            print(residual) 
            
    cfg = {
        'debug': False,
        'log': True,
        'description': f'calgary_AugResDAE{it}_localAug_multiImgSingleView_{"res" if residual else "recon"}_balanced_same_up3',
        'project': 'MICCAI2023-loose_end',

        # Data params
        'n': 0,
        'root': '../../',
        'data_path': 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/',
        'train_site': 6,
        'unet': f'calgary_unet{it}',
        'channel_out': 8,

        # Hyperparams
        'batch_size': 32,
        'augment': False,
        'difference': True,
        'reconstruction': True, 
        'loss': 'huber',
        'target': 'output',
        'disabled_ids': ['shortcut0', 'shortcut1', 'shortcut2'],
#         'disabled_ids': ['up3', 'shortcut1', 'shortcut2'],
#         'disabled_ids': [],
        # outputs
        'plot_dir': '../experiments/unet/calgary/logs/'
    }
    
    if cfg['log']:
        run = wandb.init(reinit=True, 
                         name=cfg['description'],
                         project=cfg['project'], 
                         config=cfg)
        cfg = wandb.config
        
    description = cfg['description']
    if cfg['augment']:
        description += 'augment'

    
    ### data loading 
    root      = cfg['root']
    data_path = root + cfg['data_path']    
    
    # Training data
    train_set = CalgaryCampinasDataset(
        data_path=data_path, 
        site=cfg['train_site'], 
        augment=False, 
        normalize=True, 
        split='train', 
        debug=cfg['debug']
    )
    
    train_loader = MultiImageSingleViewDataLoader(
        train_set,
        batch_size=cfg['batch_size'], 
        return_orig=True
    )
    
    transforms = Transforms()
    train_augmentor = transforms.get_transforms('local_transforms')
    train_gen = MultiThreadedAugmenter(
        data_loader = train_loader, 
        transform = train_augmentor, 
        num_processes = 4, 
        num_cached_per_queue = 2, 
        seeds=None
    )    
    
    # Validation data
    valid_set = CalgaryCampinasDataset(
        data_path=data_path, 
        site=cfg['train_site'], 
        augment=False, 
        normalize=True, 
        split='validation', 
        debug=cfg['debug']
    )
    
    valid_augmentor = transforms.get_transforms('local_val_transforms')
    valid_loader = MultiImageSingleViewDataLoader(
        valid_set,
        batch_size=cfg['batch_size'], 
        return_orig=True
    )
    
    valid_gen = MultiThreadedAugmenter(valid_loader, valid_augmentor, 4, 2, seeds=None)
    
    ### Unet
    unet_path = cfg['unet']
    unet = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=cfg['channel_out']).to(0)
    model_path = f'{root}pre-trained/trained_UNets/{unet_path}_best.pt'
    state_dict = torch.load(model_path)['model_state_dict']
    unet.load_state_dict(state_dict)

    ### AE Params
#     layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']

#### ORIGINAL

#                        #    channel, spatial, latent,  depth, block 
#     dae_map   = {
#          'shortcut0': [         8,     256,    256,     3,      4],
#          'shortcut1': [        16,     128,    256,     3,      4],
#          'shortcut2': [        32,      64,    256,     3,      4],
#          'up3':       [        64,      32,    256,     3,      4]
#     }
    
####
    
                     #    channel, spatial, latent,  depth, block 
    dae_map   = {
         'shortcut0': [         8,     256,    256,     3,      4],
         'shortcut1': [        16,     128,    256,     3,      4],
         'shortcut2': [        32,      64,    256,     3,      4],
         'up3':       [        64,      32,    256,     3,      4]
    }
    
# #                         channel, spatial, latent,  depth, block 
#     dae_map   = {
#          'out_path': [         8,     256,    256,     3,      4],
#     }
    
    
    DAEs = nn.ModuleDict({
        key: AugResDAE(
            in_channels = dae_map[key][0], 
            in_dim      = dae_map[key][1],
            latent_dim  = dae_map[key][2],
            depth       = dae_map[key][3],
            block_size  = dae_map[key][4],
            residual    = residual)
        for key in dae_map
    })


    for layer_id in cfg['disabled_ids']:
         DAEs[layer_id] = nn.Identity()


    model = ModelAdapter(
        unet, 
        DAEs, 
        disabled_ids=cfg['disabled_ids'],
        copy=True
    )
    
    criterion = CalgaryCriterionAE(
        loss=cfg['loss'], 
        recon=cfg['reconstruction'], 
        diff=cfg['difference']
    )
    
    eval_metrics = {
        'Sample Volumetric Dice': SampleDice(data='calgary'),
        'UNet Volumetric Dice': UnetDice(data='calgary')
    }
    
    ae_trainer = AETrainerCalgaryV2(
        model=model, 
        unet=unet, 
        criterion=criterion,
        train_loader=train_gen, 
        valid_loader=valid_gen,
        num_batches_per_epoch=250,
        num_val_batches_per_epoch=50,
        root=root,
        target=cfg['target'],
        description=description,
        lr=1e-4, 
        eval_metrics=eval_metrics, 
        log=cfg['log'],
        n_epochs=250,
        patience=20
    ) 
    ae_trainer.fit()
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
