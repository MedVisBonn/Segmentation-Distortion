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
from random import randrange

from nnunet.training.model_restore import restore_model
import batchgenerators
from batchgenerators.transforms.local_transforms import *
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.run.load_pretrained_weights import load_pretrained_weights

sys.path.append('..')
from dataset import CalgaryCampinasDataset, ACDCDataset, MNMDataset
from augment import SingleImageMultiViewDataLoader, MultiImageSingleViewDataLoader
from utils import EarlyStopping, epoch_average, average_metrics
from model.dae import AugResDAE
from model.unet import UNet2D
from model.wrapper import FrankensteinV2
from losses import MNMCriterionAE, SampleDice, UnetDice
from trainer.ae_trainerV2 import AETrainerACDCV2




def main(args):    
    
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft()
        if arg in ['--residual']:
            residual = True if int(arguments.popleft()) == 1 else False
            print(residual) 
    cfg = {
        'debug': False,
        'log': True,
        'description': f'acdc_AugResDAE{it}_localAug_multiImgSingleView_{"res" if residual else "recon"}_balanced_same', #'mms_vae_for_nnUNet_fc3_0_bs50',
        'project': 'MICCAI2023-loose_ends',

        # Data params
        'n': 0,
        'root': '../../',
        'data_path': 'data/mnm/',
        'unet': f'acdc_unet8_{it}',
        'channel_out': 8,

        # Hyperparams
        'batch_size': 32,
        'lr': 1e-4,
        'augment': False,
        'difference': True,
        'loss': 'huber',  # huber or ce
        'target': 'output', #gt or output
        'reconstruction': True,
        'augmentations': 'local',
        'disabled_ids': [
            'shortcut0', 
            'shortcut1', 
            'shortcut2'
        ], #['shortcut0', 'shortcut1', 'shortcut2']
    }
    
    nnUnet_prefix = '../../../nnUNet/'
    
    description = cfg['description']
    root = cfg['root']

    # Unet
    unet_path = cfg['unet'] # + str(cfg['n'])
    unet = UNet2D(n_chans_in=1, n_chans_out=4, n_filters_init=cfg['channel_out']).cuda()
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

    if cfg['augmentations'] == 'all':
        train_transforms = [t for t in train_loader.transform.transforms]
        valid_transforms = [t for t in valid_loader.transform.transforms]
        
    elif cfg['augmentations'] == 'output_invariant':
        data_only_transforms = (
            batchgenerators.transforms.resample_transforms.SimulateLowResolutionTransform,
            batchgenerators.transforms.noise_transforms.GaussianNoiseTransform,
            batchgenerators.transforms.noise_transforms.GaussianBlurTransform,
            batchgenerators.transforms.color_transforms.BrightnessMultiplicativeTransform,
            batchgenerators.transforms.color_transforms.ContrastAugmentationTransform,
            batchgenerators.transforms.color_transforms.GammaTransform,
            batchgenerators.transforms.utility_transforms.RemoveLabelTransform,
            batchgenerators.transforms.utility_transforms.RenameTransform,
            batchgenerators.transforms.utility_transforms.NumpyToTensor
        )

        train_transforms = [t for t in train_loader.transform.transforms if isinstance(t, data_only_transforms)]
        valid_transforms = [t for t in valid_loader.transform.transforms if isinstance(t, data_only_transforms)]

    elif cfg['augmentations'] == 'local':
        original_transforms = (
            batchgenerators.transforms.resample_transforms.SimulateLowResolutionTransform,
            batchgenerators.transforms.noise_transforms.GaussianNoiseTransform,
            batchgenerators.transforms.utility_transforms.RemoveLabelTransform,
            batchgenerators.transforms.utility_transforms.RenameTransform,
            batchgenerators.transforms.utility_transforms.NumpyToTensor
        )

        scale = 200.
        local_transforms = [
            BrightnessGradientAdditiveTransform(scale=scale, max_strength=4, p_per_sample=0.2, p_per_channel=1),
            LocalGammaTransform(scale=scale, gamma=(2, 5), p_per_sample=0.2, p_per_channel=1),
            LocalSmoothingTransform(scale=scale, smoothing_strength=(0.5, 1), p_per_sample=0.2, p_per_channel=1),
            LocalContrastTransform(scale=scale, new_contrast=(1, 3), p_per_sample=0.2, p_per_channel=1),
        ]

        train_transforms = local_transforms + [t for t in train_loader.transform.transforms if isinstance(t, original_transforms)]
        valid_transforms = local_transforms + [t for t in valid_loader.transform.transforms if isinstance(t, original_transforms)]

    train_augmentor = batchgenerators.transforms.abstract_transforms.Compose(train_transforms)
    valid_augmentor = batchgenerators.transforms.abstract_transforms.Compose(valid_transforms)
    ### - Load dataset and init batch generator
    train_data = ACDCDataset(data='train', debug=False)
    valid_data = ACDCDataset(data='val', debug=False)

    train_gen = MultiImageSingleViewDataLoader(train_data, batch_size=cfg['batch_size'], return_orig=True)
    #train_gen = SingleThreadedAugmenter(train_gen, train_augmentor)
    train_gen = MultiThreadedAugmenter(train_gen, train_augmentor, 4, 2, seeds=None)
    valid_gen = MultiImageSingleViewDataLoader(valid_data, batch_size=cfg['batch_size'], return_orig=True)
    #valid_gen = SingleThreadedAugmenter(valid_gen, valid_augmentor)
    valid_gen = MultiThreadedAugmenter(valid_gen, valid_augmentor, 4, 2, seeds=None)


    ### VAE Params
    layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']

                       #    channel, spatial, latent,  depth, block 
    dae_map   = {
         #'shortcut0': [         8,     256,    128,     6,      1],
         #'shortcut1': [        16,     128,    128,     5,      1],
         #'shortcut2': [        32,      64,    128,     4,      1],
         'up3':       [        64,      32,    256,     3,      4]}

    cfg['dae_map'] = dae_map
    if cfg['log']:
        run = wandb.init(reinit=True, 
                         name=cfg['description'],
                         project=cfg['project'], 
                         config=cfg)
        cfg = wandb.config


#     DAEs = nn.ModuleDict({key: AugResDAE(in_channels = dae_map[key][0], 
#                                          in_dim      = dae_map[key][1],
#                                          latent_dim  = dae_map[key][2],
#                                          depth       = dae_map[key][3],
#                                          block_size  = dae_map[key][4]) for key in dae_map})


    DAEs = nn.ModuleDict({key: AugResDAE(in_channels = dae_map[key][0], 
                                         in_dim      = dae_map[key][1],
                                         latent_dim  = dae_map[key][2],
                                         depth       = dae_map[key][3],
                                         block_size  = dae_map[key][4],
                                         residual    = residual) for key in dae_map})


    for layer_id in cfg['disabled_ids']:
         DAEs[layer_id] = nn.Identity()


    model = FrankensteinV2(unet, 
                           DAEs, 
                           disabled_ids=cfg['disabled_ids'],
                           copy=True)

    print()
    if cfg['log']:
        wandb.watch(model)

    criterion = MNMCriterionAE(
        loss=cfg['loss'], 
        recon=cfg['reconstruction'], 
        diff=cfg['difference']
    )

    eval_metrics = {
        'Sample Volumetric Dice': SampleDice(data='MNM'),
        'UNet Volumetric Dice': UnetDice(data='MNM')
    }


    ae_trainer = AETrainerACDCV2(
        model=model, 
        unet=unet, 
        criterion=criterion, 
        train_loader=train_gen, 
        valid_loader=valid_gen,
        num_batches_per_epoch=trainer.num_batches_per_epoch,
        num_val_batches_per_epoch=trainer.num_val_batches_per_epoch,
        root=root,
        target=cfg['target'],
        description=description,
        lr=cfg['lr'], 
        eval_metrics=eval_metrics, 
        log=cfg['log'],
        n_epochs=250, 
        patience=8,
        device=torch.device('cuda')
    )

    ae_trainer.fit()
    
    
if __name__ == '__main__':    
    main(sys.argv[1:])
