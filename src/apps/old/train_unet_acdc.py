import os, sys
import torch
from torch.utils.data import DataLoader 
import wandb
import collections
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

sys.path.append('../')
from dataset import ACDCDataset
from data_utils import (
    Transforms,
    MultiImageSingleViewDataLoader
)
from model.unet import UNet2D
from losses import DiceScoreMMS, CrossEntropyTargetArgmax
from trainer.unet_trainer import UNetTrainerACDC

        
def main(args):
    
    # Config
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft() 
    
    cfg = {
        'debug': False,
        'log': True,
        'description': f'acdc_unet8_{it}_ablation',
        'project': 'MICCAI2023',

        # Data params
        'n': 0,
        'root': '../../',
        'validation': True,
        'channel_out': 8,

        # Training params
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 250,
        'patience': 8
    }
    
    if cfg['log']:
        run = wandb.init(
            reinit=True, 
            name=cfg['description'],
            project=cfg['project'], 
            config=cfg
        )
        cfg = wandb.config
    
    # Data Loading
    transforms = Transforms()
    train_set = ACDCDataset(
        data="train",
        debug=cfg['debug']
    )
    train_loader = MultiImageSingleViewDataLoader(
        data=train_set, 
        batch_size=cfg['batch_size'], 
        return_orig=False
    )    
    train_augmentor = transforms.get_transforms('all_transforms')
    train_gen = SingleThreadedAugmenter(
        data_loader = train_loader, 
        transform = train_augmentor, 
#         num_processes = 4, 
#         num_cached_per_queue = 2, 
#         seeds=None
    )
    
    val_set = ACDCDataset(
        data="val",
        debug=cfg['debug']
    )
    valid_loader = MultiImageSingleViewDataLoader(
        data=val_set, 
        batch_size=cfg['batch_size'], 
        return_orig=False
    )
    valid_augmentor = transforms.get_transforms('io_transforms')
    valid_gen = SingleThreadedAugmenter(
        data_loader = valid_loader, 
        transform = valid_augmentor, 
#         num_processes = 4, 
#         num_cached_per_queue = 2, 
#         seeds=None
    )
    
    #print(len(val_set))
    
    batch = next(valid_gen)
    #print(batch['data'].shape, batch['target'].shape)
    
    # Model
    unet = UNet2D(
        n_chans_in=1,
        n_chans_out=4, 
        n_filters_init=cfg['channel_out']
    )
    if cfg['log']:
        wandb.watch(unet)
        
    # Trainer
    criterion    = CrossEntropyTargetArgmax()
    eval_metrics = {
        "Volumetric Dice": DiceScoreMMS()
    } 
    root = cfg['root']
    unet_trainer = UNetTrainerACDC(
        model=unet,
        criterion=criterion,
        train_loader=train_gen,
        valid_loader=valid_gen,
        num_batches_per_epoch=250,
        num_val_batches_per_epoch=50,
        root=root,
        eval_metrics=eval_metrics,
        lr=cfg['lr'],
        n_epochs=cfg['epochs'],
        description=cfg["description"],
        patience=cfg['patience'],
        log=cfg['log']
    )
    
    # Run
    unet_trainer.fit()


if __name__ == '__main__':
    main(sys.argv[1:])