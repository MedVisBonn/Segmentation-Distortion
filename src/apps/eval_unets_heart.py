import os, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import Tensor

from typing import Iterable, Dict, List, Callable, Tuple, Union, List
import matplotlib.pyplot as plt

sys.path.append('../')
from dataset import ACDCDataset, MNMDataset
from model.unet import UNet2D
from losses import DiceScoreMMS
from utils import  epoch_average, Metrics, Thresholder


def test_set(model, dataloader, eval_metrics):
    model.eval()
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    # saves batch sizes for each batch for averaging
    batch_sizes = []
    for batch in dataloader:
        input_ = batch['input']
        target = batch['target'].cuda()
        # convert -1 labels to background
        target[target == -1] = 0
        # convert to one-hot encoding
        target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
        # get model output
        net_out = model(input_.cuda())
        
        batch_sizes.append(input_.shape[0])
        for key, metric in eval_metrics.items():
            epoch_metrics[key].append(metric(net_out, target).detach().mean().cpu())
            
    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
        
    return epoch_metrics



def main():
    ### - datasets
    debug = False
    loader = {}
    # - ACDC train
    acdc_train = ACDCDataset(data='train', debug=debug)
    acdc_train_loader = DataLoader(acdc_train, batch_size=32, shuffle=False, drop_last=False)
    loader['acdc_train'] = acdc_train_loader
    # - ACDC val
    acdc_val = ACDCDataset(data='val', debug=debug)
    acdc_val_loader = DataLoader(acdc_val, batch_size=32, shuffle=False, drop_last=False)
    loader['acdc_val'] = acdc_val_loader
    # - M&M A
    mnm_a = MNMDataset(vendor='A', debug=debug)
    mnm_a_loader = DataLoader(mnm_a, batch_size=32, shuffle=False, drop_last=False)
    loader['mnm_siemens'] = mnm_a_loader
    # - M&M B
    mnm_b = MNMDataset(vendor='B', debug=debug)
    mnm_b_loader = DataLoader(mnm_b, batch_size=32, shuffle=False, drop_last=False)
    loader['mnm_philips'] = mnm_b_loader
    # - M&M C
    mnm_c = MNMDataset(vendor='C', debug=debug)
    mnm_c_loader = DataLoader(mnm_c, batch_size=32, shuffle=False, drop_last=False)
    loader['mnm_ge'] = mnm_c_loader
    # - M&M D
    mnm_d = MNMDataset(vendor='D', debug=debug)
    mnm_d_loader = DataLoader(mnm_d, batch_size=32, shuffle=False, drop_last=False)
    loader['mnm_canon'] = mnm_d_loader
    
    
    # - instantiate U-Net
    unet = UNet2D(n_chans_in=1, n_chans_out=4, n_filters_init=8).to(0)
    # - evaluation metrics
    eval_metrics = {
            "Volumetric Dice": DiceScoreMMS()
        } 
    # - results
    results = {key: [] for key in loader}
    
    print("Starting evaluation ...")
    # loop over all U-Nets
    for i in range(10):
        # - path
        root = '../../'
        unet_path = f"acdc_unet8_{i}"
        print(f"U-Net: {i} - path {unet_path}")
        # - load params
        model_path = f'{root}pre-trained-tmp/trained_UNets/{unet_path}_best.pt'
        state_dict = torch.load(model_path)['model_state_dict']
        unet.load_state_dict(state_dict)
        # loop over all sets
        for key in loader:
            print(f"    Vendor: {key}")
            dice = test_set(unet, loader[key], eval_metrics)
            results[key].append(dice['Volumetric Dice'].item())
            
        np.save('../../results-tmp/heart_unet_results.npy', results)
            
            
if __name__ == '__main__':    
    main()