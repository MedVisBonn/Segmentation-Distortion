import os, sys
from typing import Iterable, Dict, List, Callable, Tuple, Union, List

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../')
from dataset import CalgaryCampinasDataset
from model.unet import UNet2D
from losses import DiceScoreCalgary, SurfaceDiceCalgary
from utils import  epoch_average, volume_collate



@torch.no_grad()
def test_set(model: nn.Module, dataloader: DataLoader, eval_metrics: Dict) -> Dict:
    model.eval()
    batch_sizes = []
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    for batch in dataloader:
        input_ = batch['input']
        target = batch['target']
        batch_sizes.append(input_.shape[0])

        input_chunks  = torch.split(input_, 32, dim=0)
        # target_chunks = torch.split(target, 32, dim=0)
        net_out = []
        for input_chunk in input_chunks:
            net_out_chunk = model(input_chunk.cuda())
            net_out.append(net_out_chunk.detach().cpu())

        net_out = torch.cat(net_out, dim=0)
        for key, metric in eval_metrics.items():
            epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())

    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)

    return epoch_metrics



def main():
    ### - datasets
    debug = True
    loader = {}
    data_path = '../../data/conp-dataset/projects/calgary-campinas/CC359/Reconstructed/'
    
    # - CC359 train (6)
    calgary_train = CalgaryCampinasDataset(data_path=data_path, site=6, normalize=True, 
                                           volume_wise=True, debug=debug, split='train')
    calgary_train_loader = DataLoader(calgary_train, batch_size=1, shuffle=False, 
                                      drop_last=False, collate_fn=volume_collate)
    loader['cc_train'] = calgary_train_loader
    # - CC359 val (6)
    calgary_val = CalgaryCampinasDataset(data_path=data_path, site=6, normalize=True, 
                                         volume_wise=True, debug=debug, split='validation')
    calgary_val_loader = DataLoader(calgary_val, batch_size=1, shuffle=False, 
                                    drop_last=False, collate_fn=volume_collate)
    loader['cc_val'] = calgary_val_loader
    # - CC359 1
    cc_site1 = CalgaryCampinasDataset(data_path=data_path, site=1, normalize=True, 
                                      volume_wise=True, debug=debug, split='all')
    cc_site1_loader = DataLoader(cc_site1, batch_size=1, shuffle=False, 
                                 drop_last=False, collate_fn=volume_collate)
    loader[f"cc_site1"] = cc_site1_loader
    # - CC359 2
    cc_site2 = CalgaryCampinasDataset(data_path=data_path, site=2, normalize=True, 
                                      volume_wise=True, debug=debug, split='all')
    cc_site2_loader = DataLoader(cc_site2, batch_size=1, shuffle=False, 
                                 drop_last=False, collate_fn=volume_collate)
    loader[f"cc_site2"] = cc_site2_loader
    # - CC359 3
    cc_site3 = CalgaryCampinasDataset(data_path=data_path, site=3, normalize=True, 
                                      volume_wise=True, debug=debug, split='all')
    cc_site3_loader = DataLoader(cc_site3, batch_size=1, shuffle=False, 
                                 drop_last=False, collate_fn=volume_collate)
    loader[f"cc_site3"] = cc_site3_loader
    # - CC359 4
    cc_site4 = CalgaryCampinasDataset(data_path=data_path, site=4, normalize=True, 
                                      volume_wise=True, debug=debug, split='all')
    cc_site4_loader = DataLoader(cc_site4, batch_size=1, shuffle=False, 
                                 drop_last=False, collate_fn=volume_collate)
    loader[f"cc_site4"] = cc_site4_loader
    # - CC359 5
    cc_site5 = CalgaryCampinasDataset(data_path=data_path, site=5, normalize=True, 
                                      volume_wise=True, debug=debug, split='all')
    cc_site5_loader = DataLoader(cc_site5, batch_size=1, shuffle=False, 
                                 drop_last=False, collate_fn=volume_collate)
    loader[f"cc_site5"] = cc_site5_loader
    
    # - instantiate U-Net
    unet = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=8).to(0)
    # - evaluation metrics
    eval_metrics = {
        "Volumetric Dice": DiceScoreCalgary(),
        "Surface Dice": SurfaceDiceCalgary()
    }
    # - results
    results = {key: [] for key in loader}
    
    print("Starting evaluation ...")
    # loop over all U-Nets
    for i in range(10):
        # - path
        root = '../../'
        unet_path = f"calgary_unet{i}"
        print(f"U-Net: {i} - path {unet_path}")
        # - load params
        model_path = f'{root}pre-trained-tmp/trained_UNets/{unet_path}_best.pt'
        state_dict = torch.load(model_path)['model_state_dict']
        unet.load_state_dict(state_dict)
        # loop over all sets
        for key in loader:
            print(f"    Case: {key}")
            dice_scores = test_set(unet, loader[key], eval_metrics)
            results[key].append(dice_scores)
            
        np.save('../../results-tmp/brain_unet_results.npy', results)
            
            
if __name__ == '__main__':    
    main()