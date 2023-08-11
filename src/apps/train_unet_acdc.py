import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Callable
import time
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
import numpy as np
import collections

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
sys.path.append('../')
from utils import EarlyStopping, epoch_average, slice_selection, dataset_from_indices, volume_collate
from model.unet import UNet2D
from losses import SampleDice, UnetDice, DiceScoreMMS, CrossEntropyTargetArgmax
from trainer.unet_trainer import UNetTrainerACDC


        
def main(args):
    
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft() 
                        
    ### Dataloader
    ## Initialize trainer to get data loaders with data augmentations from training
    nnUnet_prefix = '../../../nnUNet/'
    pkl_file          = nnUnet_prefix + 'data/nnUNet_preprocessed/Task500_ACDC/nnUNetPlansv2.1_plans_2D.pkl'
    fold              = 0
    output_folder     = nnUnet_prefix + 'results/nnUnet/nnUNet/2d/Task027_ACDC/nnUNetTrainerV2__nnUNetPlansv2.1/'
    dataset_directory = nnUnet_prefix + 'data/nnUNet_preprocessed/Task500_ACDC'

    trainer = nnUNetTrainerV2(pkl_file, 0, output_folder, dataset_directory)
    trainer.initialize()

    train_loader = trainer.tr_gen
    valid_loader = trainer.val_gen

    
    cfg = {
        'debug': False,
        'log': True,
        'description': f'acdc_unet8_{it}',
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
        run = wandb.init(reinit=True, 
                         name=cfg['description'],
                         project=cfg['project'], 
                         config=cfg)
        cfg = wandb.config


    seg_model    = UNet2D(n_chans_in=1, n_chans_out=4, n_filters_init=cfg['channel_out'])
    criterion    = CrossEntropyTargetArgmax()
    eval_metrics = {
        "Volumetric Dice": DiceScoreMMS()
    } 
    root = cfg['root']
    unet_trainer = UNetTrainerACDC(model=seg_model,
                                   criterion=criterion,
                                   train_loader=train_loader,
                                   valid_loader=valid_loader,
                                   num_batches_per_epoch=trainer.num_batches_per_epoch,
                                   num_val_batches_per_epoch=trainer.num_val_batches_per_epoch,
                                   root=root,
                                   eval_metrics=eval_metrics,
                                   lr=cfg['lr'],
                                   n_epochs=cfg['epochs'],
                                   description=cfg["description"],
                                   patience=cfg['patience'],
                                   log=cfg['log'])

    unet_trainer.fit()


if __name__ == '__main__':    
    main(sys.argv[1:])