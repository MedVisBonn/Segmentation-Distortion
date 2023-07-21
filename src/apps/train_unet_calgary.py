import os
import collections
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import sys
import wandb

sys.path.append('..')
from utils import volume_collate
from dataset import CalgaryCampinasDataset
from model.UNet2D20 import UNet2D
from losses import CalgaryCriterion, SampleDice, UnetDice, DiceScoreCalgary, SurfaceDiceCalgary
from unet_trainer import UNetTrainer


def main(args):
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft() 
        
    cfg = {
        'debug': False,
        'log': True,
        'description': f'calgary_unet{it}',
        'project': 'IPMI2023',

        # Data params
        'root': '../../',
        'data_path': 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/',
        'train_site': 6,
        'augment': False,
        'validation': True,
        'channel_out': 8,

        # Training params
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 250,
        'patience': 4
    }
        
    
    if cfg['log']:
        run = wandb.init(reinit=True, 
                         name=cfg['description'],
                         project=cfg['project'], 
                         config=cfg)
        cfg = wandb.config
    
    print(f"Decsription: {description}")
    
    data_path = cfg['root'] + cfg['data_path']
    
    train_set = CalgaryCampinasDataset(data_path=data_path, site=cfg['train_site'], 
                                       augment=cfg['augment'], normalize=True, 
                                       split='train', debug=cfg['debug'])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=10)
    

    valid_set = CalgaryCampinasDataset(data_path=data_path, site=cfg['train_site'], 
                                       normalize=True, volume_wise=True,
                                       split='validation')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=False, 
                          collate_fn=volume_collate)


    # UNET
    seg_model   = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=cfg['channel_out'])
    criterion   = nn.BCEWithLogitsLoss()
    eval_metrics = {
        "Volumetric Dice": DiceScoreCalgary(),
        "Surface Dice": SurfaceDiceCalgary()
    }
    root = cfg['root']
    unet_trainer = UNetTrainer(model=seg_model, 
                               criterion=criterion, 
                               train_loader=train_loader, 
                               valid_loader=valid_loader, 
                               root=root, 
                               eval_metrics=eval_metrics, 
                               lr=cfg['lr'],
                               n_epochs=cfg['epochs'],
                               description=description,
                               patience=cfg['patience'],
                               log=True)

    unet_trainer.fit()
    metrics = unet_trainer.eval_calgary(cfg=cfg)
    np.save(f'{root}experiments/results/unet/metrics_{description}', metrics)

if __name__ == '__main__':    
    main(sys.argv[1:])
