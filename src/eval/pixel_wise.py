from typing import Iterable, Dict, Callable, Tuple, Union, List
import wandb
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import _activate_dropout, UMapGenerator, Thresholder, Metrics

            
@torch.no_grad()
def get_metrics_from_vaes(model: nn.Module, dataloader: DataLoader, 
                          net_out: str, n_samples: int = 10, n_taus: int = 1000,
                          method='vae', device='cuda:0') -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for VAE-Based uncertainty estimation.
    
    From a pre-trained model with hooked VAEs and a test loader for either calgary or M&M
    data, generate umaps at different thresholds and aggregate them in a Metrics object for
    post-processing and logging.
    """
    # Remove trainiung hooks, add evaluation hooks
    model.remove_all_hooks()        
    model.hook_transformations(model.transformations,
                               n_samples=n_samples)
    # Put model in evaluation state
    model.to(device)
    model.eval()
    model.freeze_seg_model()

    len_        = dataloader.dataset.__len__()
    bs          = dataloader.batch_size        
    gen_umap    = UMapGenerator(method=method, net_out=net_out).to(device)
    metrics     = Metrics(n_taus)
    umaps       = []
    errmaps     = []
    
    
    for _, batch in enumerate(dataloader):

        input_ = batch['input'].to(device)
        gt     = batch['target'].to(device)
        gt[gt == -1] = 0
        output = model(input_)

        if net_out == 'calgary':
            segmap = (torch.sigmoid(output[:1]) > 0.5) * 1
            errmap = (gt != segmap).float()

        elif net_out == 'mms':
            segmap = torch.argmax(output[:1], dim=1, keepdims=True)
            errmap = (torch.argmax(gt, dim=1, keepdims=True) != segmap).float()
        
        umaps.append(gen_umap(output).cpu())
        errmaps.append(errmap.cpu())
        
    umap_values = torch.cat(umaps).flatten()
    taus        = np.quantile(umap_values, torch.linspace(0, 1, n_taus)**(1/8))
    thresholder = Thresholder(torch.from_numpy(taus).view(1, n_taus, 1, 1))
        
    for umap, errmap in zip(umaps, errmaps):
        binary_umaps = thresholder(umap)
        metrics.update(binary_umaps, 
                       errmap)

    metrics.scale(len_)
    metrics.summary_stats()

    return metrics


def get_metrics_from_dropout(model: nn.Module, dataloader: DataLoader, 
                             name: str, n_samples: int = 10, n_taus: int = 1000,
                             net_out: str = 'calgary', log=True, device='cuda:0') -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for Entropy-Based uncertainty estimation.
    
    From a pre-trained model and a test loader for the calgary or M&M data, generate umaps by 
    calculating the entropy from dropout samples, threshold them at different evenly spaced 
    values and aggregate in a Metrics object for post-processing and logging.
    """
    
    model.to(device)
    
    if log:
        run = wandb.init(reinit=True, name=name, project='Thesis-VAE')
        
    len_        = dataloader.dataset.__len__()
    gen_umap    = UMapGenerator(method='entropy', net_out=net_out).to(device)
    thresholder = Thresholder(n_taus=n_taus, max_value=10)
    metrics     = Metrics(n_taus)

    for _, batch in enumerate(dataloader):

        input_ = batch['input'].to(device)
        gt     = batch['target'].to(device)
        
        # put model in eval mode to get reference segmentation w/o dropout
        model.eval()
        output = model(input_)

        if net_out == 'calgary':
            segmap = (torch.sigmoid(output) > 0.5) * 1
            errmap = (gt != segmap).float()

        elif net_out == 'mms':
            segmap = torch.argmax(output, dim=1, keepdims=True)
            errmap = (torch.argmax(gt, dim=1, keepdims=True) != segmap).float()
        
        # activate only dropout while keeping norms etc in eval mode
        model.apply(_activate_dropout)            
        B, C, H, W = output.size()
        samples    = torch.zeros((n_samples, C, H, W), device=device)
        
        for i in range(n_samples):
            samples[i] = model(input_)
            
            #if net_out == 'calgary':
            #    samples[i] = torch.sigmoid(output)
            #elif net_out == 'mms':
            #    samples[i] = torch.argmax(output, dim=1, keepdims=True)
        #print("allclose", torch.allclose(samples[0], samples[1]))
        umap         = gen_umap(samples).cpu()
        binary_umaps = thresholder(umap)

        metrics.update(binary_umaps, 
                       errmap.cpu())

    metrics.scale(len_)
    metrics.summary_stats()

    if log:
        metrics.log()

    return metrics


@torch.no_grad()
def get_metrics_from_probs(model: nn.Module, dataloader: DataLoader, 
                           net_out: str, n_taus: int = 1000,
                           device='cuda:0') -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for Entropy-Based uncertainty estimation.
    
    From a pre-trained model and a test loader for the calgary data, generate umaps by calculating the
    entropy directly from the model output, threshold them at different evenly spaced values and aggregate
    in a Metrics object for post-processing and logging.
    """
    
    model.to(device)
    model.eval()
        
    len_        = dataloader.dataset.__len__()
    gen_umap    = UMapGenerator(method='probs', net_out=net_out).to(device)
    metrics     = Metrics(n_taus)
    
    umaps       = []
    errmaps     = []
    
    
    for _, batch in enumerate(dataloader):

        input_ = batch['input'].to(device)
        gt     = batch['target'].to(device)
        gt[gt == -1] = 0
        # put model in eval mode to get reference segmentation w/o dropout
        model.eval()
        output = model(input_)

        if net_out == 'calgary':
            segmap = (torch.sigmoid(output) > 0.5) * 1
            errmap = (gt != segmap).float()
            
        if net_out == 'mms':
            segmap = torch.argmax(output, dim=1, keepdims=True)
            errmap = (torch.argmax(gt, dim=1, keepdims=True) != segmap).float()
        
        umap = gen_umap(output).cpu()
        umaps.append(umap)
        errmaps.append(errmap.cpu())
        
    umap_values = torch.cat(umaps).flatten()
    taus        = np.quantile(umap_values, torch.linspace(0, 1, n_taus)**(1/8))
    thresholder = Thresholder(torch.from_numpy(taus).view(1, n_taus, 1, 1))   
    
    for umap, errmap in zip(umaps, errmaps):
        binary_umaps = thresholder(umap)
        metrics.update(binary_umaps, 
                       errmap)    

    metrics.scale(len_)
    metrics.summary_stats()

    return metrics