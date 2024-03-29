import warnings
import sys
import collections
from random import sample, seed

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
sys.path.append('..')
from dataset import CalgaryCampinasDataset, ACDCDataset, MNMDataset
from model.unet import UNet2D, UNetEnsemble
from model.ae import AE
from model.dae import resDAE, AugResDAE
from model.wrapper import Frankenstein
from losses import (
    DiceScoreCalgary, 
    DiceScoreMMS, 
    SurfaceDiceCalgary, 
    AccMMS
)
from utils import volume_collate
from eval.slice_wise import (
    PoolingMahalabonisDetector, 
    AEMahalabonisDetector, 
    MeanDistSamplesDetector, 
    EntropyDetector, 
    EnsembleEntropyDetector
)

def main(args):
    ## default parameters
    selection = 'all_cases'
    
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['--n_unets']:
            n_unets = int(arguments.popleft())
        if arg in ['--net_out']:
            net_out = arguments.popleft()
        if arg in ['--method']:
            method = arguments.popleft()
        if arg in ['--task']:
            task = arguments.popleft()
        if arg in ['--scanner']:
            scanner = arguments.popleft()
            if net_out == 'calgary':
                scanner = int(scanner)
        if arg in ['--selection']:
            selection = arguments.popleft()
        if arg in ['--debug']:
            debug = True if arguments.popleft() == 'yes' else False
        if arg in ['--post']:
            post = arguments.popleft()
        if arg in ['--save_id']:
            save_id = arguments.popleft()
            
    print(f'Configuration: {net_out}-{method}-{task}-{scanner}-{selection}')
    
    
    criterion = DiceScoreCalgary() if net_out=='calgary' else AccMMS()
            
    # Globals
    ROOT = '../../'
    SEED = 42
    
    # Dataset
    if net_out == 'calgary':
        data       = 'data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/'
        data_path  = ROOT + data
        
        if method != 'ae':
            train_set = CalgaryCampinasDataset(data_path=data_path, 
                                               site=6,
                                               augment=False, 
                                               normalize=True,
                                               volume_wise=False,
                                               split='train',
                                               debug=debug)
                
            train_loader = DataLoader(train_set, 
                                      batch_size=32, 
                                      shuffle=False, 
                                      drop_last=False,
                                      num_workers=10)
            
        valid_set = CalgaryCampinasDataset(data_path=data_path, 
                                           site=6,
                                           augment=False, 
                                           normalize=True,
                                           volume_wise=True,
                                           split='validation',
                                           debug=debug)
                
        valid_loader = DataLoader(valid_set, 
                                  batch_size=1, 
                                  shuffle=False, 
                                  drop_last=False, 
                                  collate_fn=volume_collate)
        
        test_set  = CalgaryCampinasDataset(data_path=data_path, 
                                           site=scanner,
                                           augment=False, 
                                           normalize=True,
                                           volume_wise=True,
                                           split='validation' if scanner == 6 else 'all',
                                           debug=debug)
        
        test_loader = DataLoader(test_set,
                                 batch_size=1, 
                                 shuffle=False, 
                                 drop_last=False, 
                                 collate_fn=volume_collate)
        
    elif net_out == 'mms':
        data = 'data/mnm/'
        data_path = ROOT + data
        if method != 'ae':
            train_set = ACDCDataset(data='train', 
                                    debug=debug)
            
            train_loader = DataLoader(train_set, 
                                      batch_size=32, 
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=10)
            
        valid_set = ACDCDataset(data='val', 
                                debug=debug)
        
        valid_loader = DataLoader(valid_set, 
                                  batch_size=1, 
                                  shuffle=False, 
                                  drop_last=False, 
                                  num_workers=10)
        
        if scanner != 'val':
            test_set = MNMDataset(vendor=scanner,
                                  selection=selection,
                                  debug=debug)
        else:
            test_set = ACDCDataset(data=scanner,
                                   debug=debug)
        
        test_loader = DataLoader(test_set, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 drop_last=False,
                                 num_workers=10)
    
    # U-Nets
    middle = 'unet' if net_out == 'calgary' else 'unet8_'
    pre = 'calgary' if net_out == 'calgary' else 'acdc'
    unet_names = [f'{pre}_{middle}{i}' for i in range(n_unets)]
    unets = []
    for name in unet_names:
        model_path = f'{ROOT}pre-trained-tmp/trained_UNets/{name}_best.pt'
        state_dict = torch.load(model_path)['model_state_dict']
        n_chans_out = 1 if net_out == 'calgary' else 4
        unet = UNet2D(n_chans_in=1, 
                      n_chans_out=n_chans_out, 
                      n_filters_init=8, 
                      dropout=False)
        unet.load_state_dict(state_dict)
        unets.append(unet)
    
    metrics = []
    
    if method == 'gonzales':
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            
            metrics.append({})
            detector = PoolingMahalabonisDetector(model=unet,
                                                  layer_ids=['up3'],
                                                  train_loader=train_loader,
                                                  valid_loader=valid_loader,
                                                  net_out=net_out,
                                                  criterion=criterion)
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader)['up3'].item()
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)['up3']
    
        
    if method == 'latents':
        
        disabled_ids = ['shortcut0', 'shortcut1', 'shortcut2']
        AEs = nn.ModuleDict({'up3': AE(in_channels = 64, 
                                       in_dim      = 32,
                                       latent_dim  = 128 if net_out=='mms' else 64,
                                       depth       = 2,
                                       block_size  = 4)})
        
        for layer_id in disabled_ids:
            AEs[layer_id] = nn.Identity()
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            
            model = Frankenstein(seg_model=unet,
                                 transformations=AEs,
                                 disabled_ids=disabled_ids,
                                 copy=True)
            
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_ae{i}_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append({})
            detector = AEMahalabonisDetector(model=model,
                                              layer_ids=['up3'],
                                              train_loader=train_loader,
                                              valid_loader=valid_loader,
                                              net_out=net_out.lower(),
                                              criterion=criterion)
            
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader)['up3'].item()
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)['up3']
                
                
    if method == 'entropy':
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            metrics.append({})
            
            detector = EntropyDetector(model=unet,
                                       valid_loader=valid_loader,
                                       net_out=net_out,
                                       criterion=criterion)
            
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader).item()
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)   
                
    if method == 'ae':
        disabled_ids = ['shortcut0', 'shortcut1', 'shortcut2']
        AEs = nn.ModuleDict({'up3': AE(in_channels = 64, 
                                       in_dim      = 32,
                                       latent_dim  = 128 if net_out=='mms' else 64,
                                       depth       = 2,
                                       block_size  = 4)})
        
        
        for layer_id in disabled_ids:
            AEs[layer_id] = nn.Identity()
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            
            model = Frankenstein(seg_model=unet,
                                 transformations=AEs,
                                 disabled_ids=disabled_ids,
                                 copy=True)
            
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_ae{i}_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append({})
            detector = MeanDistSamplesDetector(
                model          = model,
                valid_loader   = valid_loader,
                net_out        = net_out,
                criterion      = criterion,
                umap_method    = 'mse',
                umap_reduction = 'norm'
            )
            
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader)
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)
                
                
    if method == 'single':
        disabled_ids = ['shortcut0', 'shortcut1', 'shortcut2']
        
        if '_res_' in post:
            residual = True
        else:
            residual = False
            
        print(f'Residual: {residual}')
        
        
        DAEs = nn.ModuleDict({
            'up3': AugResDAE(
                in_channels = 64, 
                in_dim      = 32,
                latent_dim  = 256,
                depth       = 3,
                block_size  = 4,
                residual    = residual
            )
        })
        
        
        for layer_id in disabled_ids:
            DAEs[layer_id] = nn.Identity()
        
        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            
            model = Frankenstein(
                seg_model=unet,
                transformations=DAEs,
                disabled_ids=disabled_ids,
                copy=True
            )
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_AugResDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_resDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_epinet_CE-only_prior-1_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_resDAE0_venus_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append({})
            detector = MeanDistSamplesDetector(
                model          = model,
                valid_loader   = valid_loader,
                net_out        = net_out,
                criterion      = criterion,
                umap_method    = 'cross_entropy',
                umap_reduction = 'mean'
            )
            
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader)
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)
                
    
    if method == 'multi':
        layer_ids = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']
                           #    channel, spatial, latent,  depth, block 
        dae_map   = {
            'shortcut0': [         8,     256,    128,     6,      1],
            'shortcut1': [        16,     128,    128,     5,      1],
            'shortcut2': [        32,      64,    128,     4,      1],
            'up3':       [        64,      32,    128,     3,      1]}

        DAEs = nn.ModuleDict({key: resDAE(in_channels = dae_map[key][0], 
                                          in_dim      = dae_map[key][1],
                                          latent_dim  = dae_map[key][2],
                                          depth       = dae_map[key][3],
                                          block_size  = dae_map[key][4],
                                          w_prior     = None) for key in dae_map})

        for i, unet in enumerate(tqdm(unets)):
            print(f"Method {method}, Unet {i} - {net_out} {scanner}")
            
            model = Frankenstein(unet, 
                                 DAEs, 
                                 disabled_ids=[],
                                 copy=True)
            
            model_path = f'{ROOT}pre-trained-tmp/trained_AEs/{pre}_resDAE{i}_{post}_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_epinet_CE-only_prior-1_best.pt'
            #model_path = f'{ROOT}pre-trained-tmp/trained_AEs/acdc_resDAE0_venus_best.pt'
            state_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(state_dict)
            
            metrics.append({})
            detector = MeanDistSamplesDetector(
                model          = model,
                valid_loader   = valid_loader,
                net_out        = net_out,
                criterion      = criterion,
                umap_method    = 'cross_entropy',
                umap_reduction = 'norm'
            )
            
            if task == 'ood' or task == 'both':
                metrics[i]['ood'] = detector.testset_ood_detection(test_loader)
            if task == 'corr' or task == 'both':
                metrics[i]['corr'] = detector.testset_correlation(test_loader)

    if method == 'ensemble':
        metrics = {}
        ensemble = UNetEnsemble(unets)
        detector = EnsembleEntropyDetector(model=ensemble, 
                                           net_out=net_out, 
                                           valid_loader=valid_loader, 
                                           criterion=criterion)
        
        if task == 'ood' or task == 'both':
            metrics['ood'] = detector.testset_ood_detection(test_loader)
        if task == 'corr' or task == 'both':
            metrics['corr'] = detector.testset_correlation(test_loader)  
    
        for task in metrics:
            for i, result in enumerate(metrics[task]):
                save_path = f'{ROOT}results-tmp/results/eval/{net_out.lower()}/image/'
                save_name = f'{net_out}-{method}_tmp-{task}-{scanner}-{i}-{selection}_auroc'
                if task == 'ood':
                    result = result.item()
                out = {task: result}
                torch.save(out, save_path + save_name)
                
    if method != 'ensemble':
        for i, matric in enumerate(metrics):
            save_path = f'{ROOT}results-tmp/results/eval/{net_out.lower()}/image/'
            save_name = f'{net_out}-{save_id}-{task}-{scanner}-{i}-{selection}'
            torch.save(matric, save_path + save_name)


        
if __name__ == '__main__':    
    main(sys.argv[1:])