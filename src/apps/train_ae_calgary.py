import os, sys
import collections
import time
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import wandb

sys.path.append('..')
from dataset import CalgaryCampinasDataset
from model.unet import UNet2D
from model.ae import AE
from trainer.ae_trainer import AETrainerCalgary
from model.wrapper import Frankenstein
from losses import CalgaryCriterionAE, SampleDice, UnetDice


def main(args):

    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft()
            
    cfg = {
        'debug': False,
        'log': False,
        'description': f'calgary_ae{it}_test',
        'project': 'MICCAI2023',

        # Data params
        'n': 0,
        'root': '../../',
        'data_path': 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/',
        'train_site': 6,
        'unet': f'calgary_unet{it}',
        'channel_out': 8,

        # Hyperparams
        'batch_size': 64,
        'augment': False,
        'difference': True,
        'loss': 'huber',
        'target': 'output',
        'identity_layers': ['shortcut0', 'shortcut1', 'shortcut2'],

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
    train_set = CalgaryCampinasDataset(data_path=data_path, 
                                       site=cfg['train_site'], 
                                       augment=cfg['augment'], 
                                       normalize=True, 
                                       split='train', 
                                       debug=cfg['debug'])
    
    valid_set = CalgaryCampinasDataset(data_path=data_path, 
                                       site=cfg['train_site'], 
                                       normalize=True, 
                                       split='validation', 
                                       debug=cfg['debug'])

    train_loader = DataLoader(train_set, 
                              batch_size=cfg['batch_size'], 
                              shuffle=True, 
                              drop_last=False,
                              num_workers=10)
    
    valid_loader = DataLoader(valid_set, 
                              batch_size=cfg['batch_size'], 
                              shuffle=False, 
                              drop_last=False,
                              num_workers=10)
    
    
    ### Unet
    unet_path = cfg['unet']
    seg_model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=cfg['channel_out']).to(0)
    model_path = f'{root}pre-trained-tmp/trained_UNets/{unet_path}_best.pt'
    state_dict = torch.load(model_path)['model_state_dict']
    seg_model.load_state_dict(state_dict)

    
    ### AE Params
    layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']
       
    
                       # channel, spatial, latent, depth
    ae_map   = {'up3': [     64,      32,     64,     2]}
    
    
    AEs = nn.ModuleDict({layer_id: AE(in_channels = ae_map[layer_id][0], 
                                      in_dim      = ae_map[layer_id][1],
                                      latent_dim  = ae_map[layer_id][2],
                                      depth       = ae_map[layer_id][3],
                                      block_size  = 4) 
                              for layer_id in layer_ids if layer_id not in cfg['identity_layers']})
    
        
    for layer_id in cfg['identity_layers']:
        AEs[layer_id] = nn.Identity()
    
    model = Frankenstein(seg_model, 
                         AEs, 
                         disabled_ids=cfg['identity_layers'],
                         copy=True)
    
    criterion = CalgaryCriterionAE(loss=cfg['loss'])
    
    eval_metrics = {'Sample Volumetric Dice': SampleDice(data='calgary'),
                    'UNet Volumetric Dice': UnetDice(data='calgary')}
    
    trainer = AETrainerCalgary(model=model, 
                        unet=seg_model, 
                        criterion=criterion, 
                        train_loader=train_loader, 
                        valid_loader=valid_loader, 
                        root=root,
                        target=cfg['target'],
                        description=description,
                        lr=1e-4, 
                        eval_metrics=eval_metrics, 
                        log=cfg['log'],
                        n_epochs=250,
                        patience=4) #20
    trainer.fit()
    
    
if __name__ == '__main__':
    main(sys.argv[1:])