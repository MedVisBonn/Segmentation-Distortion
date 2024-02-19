import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
import os, sys
import time
import numpy as np
import collections
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Resize, CenterCrop
from typing import Iterable, Dict, Callable, Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nnunet.training.model_restore import restore_model
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.run.load_pretrained_weights import load_pretrained_weights

sys.path.append('..')
from utils import EarlyStopping, epoch_average, average_metrics
from model.ae import AE
from model.unet import UNet2D
from model.wrapper import Frankenstein
from losses import MNMCriterionAE, SampleDice, UnetDice
from trainer.ae_trainer import AETrainerACDC

nnUnet_prefix = '../../../nnUNet/'
        
    
def main(args):
    
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft()         
        
    cfg = {
        'debug': False,
        'log': False,
        'description': f'acdc_ae{it}_test', #'mms_vae_for_nnUNet_fc3_0_bs50',
        'project': 'MICCAI2023',

        # Data params
        'n': 0,
        'root': '../../',
        'data_path': 'data/mnm/',
        'train_vendor': 'B',
        'unet': f'acdc_unet8_{it}',
        'channel_out': 8,

        # Hyperparams
        'batch_size': 32,
        'augment': False,
        'difference': True,
        'loss': 'huber',  # huber or ce
        'target': 'output', #gt or output
        'disabled_ids': ['shortcut0', 'shortcut1', 'shortcut2']
    }

    description = cfg['description']
    root = cfg['root']

    # Unet
    unet_path = cfg['unet'] # + str(cfg['n'])
    unet = UNet2D(n_chans_in=1, n_chans_out=4, n_filters_init=cfg['channel_out']).to(0)
    model_path = f'{root}pre-trained-tmp/trained_UNets/{unet_path}_best.pt'
    state_dict = torch.load(model_path)['model_state_dict']
    unet.load_state_dict(state_dict)

    ### Dataloader
    ## Initialize trainer to get data loaders with data augmentations from training
    pkl_file          = nnUnet_prefix + 'data/nnUNet_preprocessed/Task500_ACDC/nnUNetPlansv2.1_plans_2D.pkl'
    fold              = 0
    output_folder     = nnUnet_prefix + 'results/nnUnet/nnUNet/2d/Task027_ACDC/nnUNetTrainerV2__nnUNetPlansv2.1/'
    dataset_directory = nnUnet_prefix + 'data/nnUNet_preprocessed/Task500_ACDC'

    trainer = nnUNetTrainerV2(pkl_file, 0, output_folder, dataset_directory)
    trainer.initialize()

    train_loader = trainer.tr_gen
    valid_loader = trainer.val_gen


    ### VAE Params
    layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']

                       #    channel, spatial, latent,  depth, block 
    dae_map   = {'up3': [        64,      32,    128,     2,      4]}

    cfg['dae_map'] = dae_map
    if cfg['log']:
        run = wandb.init(reinit=True, 
                         name=cfg['description'],
                         project=cfg['project'], 
                         config=cfg)
        cfg = wandb.config
    
    
    DAEs = nn.ModuleDict({'up3': AE(in_channels = dae_map['up3'][0], 
                                        in_dim      = dae_map['up3'][1],
                                        latent_dim  = dae_map['up3'][2],
                                        depth       = dae_map['up3'][3],
                                        block_size  = dae_map['up3'][4])})

    for layer_id in cfg['disabled_ids']:
         DAEs[layer_id] = nn.Identity()


    model = Frankenstein(unet, 
                         DAEs, 
                         disabled_ids=cfg['disabled_ids'],
                         copy=True)

    model.cuda()
    print()
    criterion    = MNMCriterionAE(loss=cfg['loss'], diff=cfg['difference'])
    eval_metrics = {'Sample Volumetric Dice': SampleDice(data='MNM'),
                    'UNet Volumetric Dice': UnetDice(data='MNM')}

    vae_trainer = AETrainerACDC(model=model, 
                             unet=unet, 
                             criterion=criterion, 
                             train_loader=train_loader, 
                             valid_loader=valid_loader, 
                             num_batches_per_epoch=trainer.num_batches_per_epoch,
                             num_val_batches_per_epoch=trainer.num_val_batches_per_epoch,
                             root=root,
                             target=cfg['target'],
                             description=description,
                             lr=1e-4, 
                             eval_metrics=eval_metrics, 
                             log=cfg['log'],
                             n_epochs=250, 
                             patience=8)


    vae_trainer.fit()
    
    
if __name__ == '__main__':    
    main(sys.argv[1:])