from typing import (
    Iterable,
    Dict, 
    Callable, 
    Tuple, 
    Union, 
    List
)
from omegaconf import OmegaConf 
import wandb
import numpy as np
from sklearn.metrics import auc
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import BinaryPrecisionRecallCurve
from utils import (
    _activate_dropout, 
    UMapGenerator, 
    Thresholder,
    Metrics
)



@torch.no_grad()
def get_precision_recall(
    model: nn.Module, 
    dataset: Dataset, 
    net_out: str, 
    dae: bool,
    umap: str,
    n_taus: Union[str, int] = 'auto',
    device=['cuda:0', 'cpu']
):
    if dae == True:
        # Remove trainiung hooks, add evaluation hooks
        model.remove_all_hooks()        
        model.hook_inference_transformations(model.transformations, n_samples=1)
        # Put model in evaluation state
        model.to(device[0])
        model.eval()
        model.freeze_seg_model()

    batch_size = 32

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True
    )

    umap_generator = UMapGenerator(
        method=umap,
        net_out=net_out,
    ).to(device[0])

    umaps   = []
    errmaps = []

    for _, batch in enumerate(dataloader):

        input_ = batch['input'].to(device[0])
        gt = batch['target'].to(device[0])
        gt[gt == -1] = 0
        output = model(input_)

        if net_out == 'brain':
            segmap = (torch.sigmoid(output[:batch_size]) > 0.5) * 1
            errmap = (gt != segmap).float()
        elif net_out == 'heart':
            segmap = torch.argmax(output[:batch_size], dim=1, keepdims=True)
            errmap = (gt != segmap).float()
        umaps.append(umap_generator(output, batch_size=batch_size).cpu())
        errmaps.append(errmap.cpu())

    umaps = torch.cat(umaps, dim=0).flatten().half()
    umaps = (umaps - umaps.min()) / (umaps.max() - umaps.min())
    errmaps = torch.cat(errmaps, dim=0).flatten().to(torch.uint8)

    # in case of manual threshold selection
    if n_taus != 'auto':
        # taus = np.quantile(umaps, torch.linspace(0, 1, n_taus)**(1/16)).tolist()
        taus = np.quantile(umaps, torch.linspace(0, 1, n_taus)).tolist()
    elif n_taus == 'auto':
        taus = None

    # TODO: Change to torcheval once its stable :)
    # bprc = torcheval.metrics.BinaryPrecisionRecallCurve()
    bprc = BinaryPrecisionRecallCurve(thresholds = taus).to(device[1])
    pr = bprc(umaps.to(device[1]), errmaps.to(device[1]))
    if device[1] != 'cpu':
        pr = tuple(map(lambda x: x.cpu(), pr))

    pr_auc = auc(pr[1], pr[0])

    # subset precision/recall to 100 points for plotting
    # we find indices along the x axis (recall) such that they
    # have roughly equal distance to each other and select the 
    # corresponding y values (precision)
    p, r, _= pr  
    # find indices for sorting in recall
    indices = np.argsort(r)
    # sort precision and recall similarly
    r_sorted, p_sorted = (r[indices], p[indices])
    # select target values for recall
    target_values = np.linspace(r_sorted[0], r_sorted[-1], 100)
    # find best matches in sorted recall, that are smaller or equal to the target values
    subset_indices = np.abs(r_sorted[None, :] - target_values[:, None]).argmin(axis=1)
    # select precision and recall at the best matches
    r_sampled, p_sampled = (
        r_sorted[subset_indices], 
        p_sorted[subset_indices]
    )

    return p_sampled, r_sampled, pr_auc



@torch.no_grad()
def get_metrics_from_aes(
    model: nn.Module, 
    dataloader: DataLoader, 
    net_out: str, 
    n_samples: int = 10, 
    n_taus: int = 1000,
    method='cross_entropy', 
    device='cuda:0'
) -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for VAE-Based uncertainty estimation.
    
    From a pre-trained model with hooked VAEs and a test loader for either calgary or M&M
    data, generate umaps at different thresholds and aggregate them in a Metrics object for
    post-processing and logging.
    """
    # Remove trainiung hooks, add evaluation hooks
    model.remove_all_hooks()        
    model.hook_inference_transformations(model.transformations,
                               n_samples=n_samples)
    # Put model in evaluation state
    model.to(device)
    model.eval()
    model.freeze_seg_model()

    len_        = len(dataloader.dataset)
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
            errmap = (gt != segmap).float()
            #errmap = (torch.argmax(gt, dim=1, keepdims=True) != segmap).float()
        umaps.append(gen_umap(output).cpu())
        errmaps.append(errmap.cpu())
        
    umap_values = torch.cat(umaps).flatten()
    taus        = np.quantile(umap_values, torch.linspace(0, 1, n_taus)**(1/8))
    thresholder = Thresholder(torch.from_numpy(taus).view(1, n_taus, 1, 1))
        
    for umap, errmap in zip(umaps, errmaps):
        binary_umaps = thresholder(umap)
        metrics.update(binary_umaps, 
                       errmap)
        #print(binary_umaps.shape)

    metrics.scale(len_)
    metrics.summary_stats()

    return metrics



@torch.no_grad()
def get_metrics_from_probs(
    model: nn.Module, 
    dataloader: DataLoader, 
    net_out: str,
    n_taus: int = 1000,
    device='cuda:0'
) -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for Entropy-Based uncertainty estimation.
    
    From a pre-trained model and a test loader for the calgary data, generate umaps by calculating the
    entropy directly from the model output, threshold them at different evenly spaced values and aggregate
    in a Metrics object for post-processing and logging.
    """
    
    model.to(device)
    model.eval()
        
    len_        = len(dataloader.dataset)
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
            segmap = torch.argmax(output.mean(0, keepdim=True), dim=1, keepdims=True)
            #errmap = (torch.argmax(gt, dim=1, keepdims=True) != segmap).float()
            errmap = (gt != segmap).float()
            
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


def get_metrics_from_dropout(
    model: nn.Module, 
    dataloader: DataLoader, 
    name: str, 
    n_samples: int = 10,
    n_taus: int = 1000,
    net_out: str = 'calgary', 
    log=True, device='cuda:0'
) -> Metrics:
    """
    Calculate, collect and post-process evaluation metrics for Entropy-Based uncertainty estimation.
    
    From a pre-trained model and a test loader for the calgary or M&M data, generate umaps by 
    calculating the entropy from dropout samples, threshold them at different evenly spaced 
    values and aggregate in a Metrics object for post-processing and logging.
    """
    
    model.to(device)
    
    if log:
        run = wandb.init(reinit=True, name=name, project='Thesis-VAE')
        
    len_        = len(dataloader.dataset)
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
        umap         = gen_umap(samples).cpu()
        binary_umaps = thresholder(umap)

        metrics.update(binary_umaps, 
                       errmap.cpu())

    metrics.scale(len_)
    metrics.summary_stats()

    if log:
        metrics.log()

    return metrics


