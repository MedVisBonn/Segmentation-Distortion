import os
import collections
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import sys
import wandb
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

sys.path.append('..')
from dataset import CalgaryCampinasDataset
from data_utils import (
    volume_collate,
    Transforms,
    MultiImageSingleViewDataLoader
)
from model.unet import UNet2D
from losses import (
    DiceScoreCalgary, 
    SurfaceDiceCalgary
)
from trainer.unet_trainer import UNetTrainerCalgary



def main(args):
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft() 
        
    cfg = {
        'debug': False,
        'log': True,
        'description': f'calgary_unet{it}_ablation',
        'project': 'MICCAI2023-loose_ends',

        # Data params
        'root': '../../',
        'data_path': 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/',
        'save_loc': 'pre-trained',
        'train_site': 6,
        'augment': True,
        'validation': True,
        'channel_out': 8,

        # Training params
        'batch_size': 32,
        'num_batches_per_epoch': 250,
        'lr': 1e-3,
        'epochs': 250,
        'patience': 4
    }
        
    
    if cfg['log']:
        run = wandb.init(
            reinit=True, 
            name=cfg['description'],
            project=cfg['project'], 
            config=cfg
        )
        cfg = wandb.config
    
    description = cfg['description']
    print(f"Decsription: {description}")
    
    data_path = cfg['root'] + cfg['data_path']
    
    train_set = CalgaryCampinasDataset(
        data_path=data_path, 
        site=cfg['train_site'], 
        augment=False, 
        normalize=True, 
        split='train', 
        debug=cfg['debug']
    )
    
    train_loader = MultiImageSingleViewDataLoader(
        data=train_set, 
        batch_size=cfg['batch_size'], 
        return_orig=False
    )
    
    transforms = Transforms()
    train_augmentor = transforms.get_transforms('all_transforms')
    train_gen = MultiThreadedAugmenter(
        data_loader = train_loader, 
        transform = train_augmentor, 
        num_processes = 4, 
        num_cached_per_queue = 2, 
        seeds=None
    )

    valid_set = CalgaryCampinasDataset(
        data_path=data_path, 
        site=cfg['train_site'], 
        normalize=True, 
        volume_wise=True,
        split='validation'
    )
    
    valid_loader = DataLoader(
        valid_set, 
        batch_size=1,
        shuffle=False, 
        drop_last=False, 
        collate_fn=volume_collate
    )

    # UNET
    unet = UNet2D(
        n_chans_in=1, 
        n_chans_out=1, 
        n_filters_init=cfg['channel_out']
    )
    if cfg['log']:
        wandb.watch(unet) 
    criterion   = nn.BCEWithLogitsLoss()
    eval_metrics = {
        "Volumetric Dice": DiceScoreCalgary(),
        "Surface Dice": SurfaceDiceCalgary()
    }
    
    root = cfg['root']
    unet_trainer = UNetTrainerCalgary(
        model=unet, 
        criterion=criterion, 
        train_generator=train_gen, 
        valid_loader=valid_loader, 
        root=root, 
        eval_metrics=eval_metrics, 
        lr=cfg['lr'],
        num_batches_per_epoch=cfg['num_batches_per_epoch'],
        n_epochs=cfg['epochs'],
        description=description,
        patience=cfg['patience'],
        log=cfg['log'],
        save_loc=cfg['save_loc']
    )

    unet_trainer.fit()
    metrics = unet_trainer.eval_all(cfg=cfg)
    np.save(f'{root}results/unet/metrics_{description}', metrics)

if __name__ == '__main__':    
    main(sys.argv[1:])
