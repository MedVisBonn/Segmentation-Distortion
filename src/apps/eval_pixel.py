import collections
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from random import sample, seed
from tqdm.auto import tqdm
import sys

sys.path.append('..')
from dataset import (
    CalgaryCampinasDataset,
    ACDCDataset, 
    MNMDataset
)
from data_utils import get_subset
from model.unet import UNet2D, UNetEnsemble
from model.ae import AE
from model.dae import resDAE, AugResDAE
from model.wrapper import FrankensteinV2
from losses import (
    DiceScoreCalgary, 
    DiceScoreMMS, 
    SurfaceDiceCalgary
)
from eval.pixel_wise import (
    get_metrics_from_aes,
    get_metrics_from_probs
)



def main(args):
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['--n_unets']:
            n_unets = int(arguments.popleft())
        if arg in ['--net_out']:
            net_out = arguments.popleft()
        if arg in ['--method']:
            method = arguments.popleft()
        if arg in ['--scanner']:
            scanner = arguments.popleft()
            if net_out == 'calgary':
                scanner = int(scanner)
        if arg in ['--debug']:
            debug = True if arguments.popleft() == 'yes' else False
        if arg in ['--post']:
            post = arguments.popleft()
        if arg in ['--save_id']:
            save_id = arguments.popleft()
    
    # Globals
    ROOT = '../../'
    SEED = 42
    print(net_out, method, scanner, debug)
    # Dataset
    if net_out == 'calgary':
        data      = 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/'
        data_path = ROOT + data
        dataset   = CalgaryCampinasDataset(
            data_path=data_path, 
            site=scanner,
            augment=False, 
            normalize=True,
            split='validation' if scanner == 6 else 'all',
            debug=debug
        )
        
    elif net_out == 'mms':
        data      = 'data/mnm/'
        data_path = ROOT + data
        if scanner != 'val':
            dataset = MNMDataset(vendor=scanner, 
                                 debug=debug)
        else:
            dataset = ACDCDataset(data=scanner, 
                                  debug=debug)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            drop_last=False, 
        )
    
    # U-Nets
    middle = 'unet' if net_out == 'calgary' else 'unet8_'
    pre = 'calgary' if net_out == 'calgary' else 'acdc'
    unet_names = [f'{pre}_{middle}{i}' for i in range(n_unets)]
    unets = []
    
    if net_out == 'calgary':
        datasets = []
    
    for name in unet_names:
        
        model_path = f'{ROOT}pre-trained/trained_UNets/{name}_best.pt'
        state_dict = torch.load(model_path)['model_state_dict']
        n_chans_out = 1 if net_out == 'calgary' else 4
        unet = UNet2D(
            n_chans_in=1, 
            n_chans_out=n_chans_out, 
            n_filters_init=8, 
            dropout=False
        )
        unet.load_state_dict(state_dict)
        unets.append(unet)
        
        # if task is on calgary, we build test sets based on
        # the respective U-Nets by selecting mostly poorly
        # segmented slices
        if net_out == 'calgary':
            datasets.append(get_subset(
                    dataset=dataset, 
                    model=unet.cuda(),
                    criterion=nn.BCEWithLogitsLoss(reduction='none'),
                    n_cases=256,
                    fraction=0.1,
                    batch_size=32,
                    verbose=True
                )
            )
    
    metrics = []
    
    if method == 'base':
        for i, unet in enumerate(tqdm(unets)):
            if net_out == 'calgary':
                dataloader = DataLoader(
                    datasets[i],
                    batch_size=1, 
                    shuffle=False, 
                    drop_last=False, 
                )
                
            metrics.append(
                get_metrics_from_probs(
                    model=unet,
                    dataloader=dataloader,
                    net_out=net_out,
                    n_taus=250
                )
            )
            
    if method == 'ensemble':
        n_unets = len(unets)
        for i in tqdm(range(10)):
            if net_out == 'calgary':
                dataloader = DataLoader(
                    datasets[i],
                    batch_size=1, 
                    shuffle=False, 
                    drop_last=False, 
                )
            
            seed(SEED+i)
            ensemble = UNetEnsemble(
                sample(unets, 5), 
                reduce='none'
            )
            metrics.append(
                get_metrics_from_probs(
                    model=ensemble,
                    dataloader=dataloader,
                    net_out=net_out,
                    n_taus=250
                )
            )
        
    if method == 'ae':
        disabled_ids = ['shortcut0', 'shortcut1', 'shortcut2']
        AEs = nn.ModuleDict(
            {'up3': AE(
                in_channels = 64, 
                in_dim      = 32,
                latent_dim  = 128 if net_out=='mms' else 64,
                depth       = 2,
                block_size  = 4)
            }
        )
        
        for layer_id in disabled_ids:
            AEs[layer_id] = nn.Identity()
        
        #for i, unet in enumerate(tqdm(unets)):
        for i, unet in enumerate(tqdm(unets)):
            if net_out == 'calgary':
                dataloader = DataLoader(
                    datasets[i],
                    batch_size=1, 
                    shuffle=False, 
                    drop_last=False, 
                )
            
            
            model = FrankensteinV2(
                seg_model=unet,
                transformations=AEs,
                disabled_ids=disabled_ids,
                copy=True
            )
            
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_ae{i}_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append(
                get_metrics_from_aes(
                    model=model, 
                    dataloader=dataloader,
                    net_out=net_out,
                    n_samples=1,
                    n_taus=250,
                    method='ae'
                )
            )
            
            
    if method == 'single':
        disabled_ids = ['shortcut0', 'shortcut1', 'shortcut2']
        DAEs = nn.ModuleDict(
            {'up3': AugResDAE(
                in_channels = 64, 
                in_dim      = 32,
                latent_dim  = 256,
                depth       = 3,
                block_size  = 4)
            }
        )

        for layer_id in disabled_ids:
            DAEs[layer_id] = nn.Identity()
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out}")
            if net_out == 'calgary':
                dataloader = DataLoader(
                    datasets[i],
                    batch_size=1, 
                    shuffle=False, 
                    drop_last=False, 
                )
            
            model = FrankensteinV2(
                seg_model=unet,
                transformations=DAEs,
                disabled_ids=disabled_ids,
                copy=True
            )
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_AugResDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_AugResDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_resDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_epinet_CE-only_prior-1_best.pt'localAug_multiImgSingleView_res
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_resDAE0_venus_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)

            metrics.append(
                get_metrics_from_aes(
                    model=model, 
                    dataloader=dataloader,
                    net_out=net_out,
                    n_samples=1,
                    n_taus=250,
                    method='cross_entropy'
                )
            )

    if method == 'multi':
        layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']
                           #    channel, spatial, latent,  depth, block 
        dae_map   = {
            'shortcut0': [         8,     256,    128,     6,      1],
            'shortcut1': [        16,     128,    128,     5,      1],
            'shortcut2': [        32,      64,    128,     4,      1],
            'up3':       [        64,      32,    128,     3,      1]}

        DAEs = nn.ModuleDict(
            {key: resDAE(
                in_channels = dae_map[key][0], 
                in_dim      = dae_map[key][1],
                latent_dim  = dae_map[key][2],
                depth       = dae_map[key][3],
                block_size  = dae_map[key][4]) for key in dae_map
            }
        )
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out}")
            
            if net_out == 'calgary':
                dataloader = DataLoader(
                    datasets[i],
                    batch_size=1, 
                    shuffle=False, 
                    drop_last=False, 
                )
            
            model = FrankensteinV2(
                seg_model=unet,
                transformations=DAEs,
                disabled_ids=[],
                copy=True
            )
            
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_resDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_epinet_CE-only_prior-1_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_resDAE0_venus_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append(
                get_metrics_from_aes(
                    model=model, 
                    dataloader=dataloader,
                    net_out=net_out,
                    n_samples=1,
                    n_taus=250,
                    method='cross_entropy'
                )
            )
        
    for i, matric in enumerate(metrics):
        save_path = f'{ROOT}results/ae/eval/'
        save_name = f'{net_out}-{save_id}-{scanner}-{i}'
        torch.save(matric, save_path + save_name)

        
if __name__ == '__main__':    
    main(sys.argv[1:])