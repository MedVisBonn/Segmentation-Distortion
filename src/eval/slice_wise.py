from typing import Iterable, Dict, List, Callable, Tuple, Union, List

import torch #
from torch import Tensor, nn #
from torch.utils.data import Dataset, DataLoader, default_collate
import torch.nn.functional as F
from sklearn.cluster import KMeans #
from sklearn.metrics import pairwise_distances_argmin_min #
from sklearn.covariance import LedoitWolf #
from scipy.stats import binned_statistic #
from tqdm.auto import tqdm #
from torchmetrics import (
    SpearmanCorrCoef, 
    AUROC)
from losses import DiceScoreCalgary, DiceScoreMMS #
from utils import _activate_dropout, UMapGenerator, UMapScorePerSlice

# CalgaryCampinasDataset
def ood_eval(detector, dataloader, per_layer=False):
    
    accuracy = detector.testset_ood_detection(dataloader)
    if per_layer:
        best_layer = max(accuracy)
        accuracy   = accuracy[best_layer]
    else:
        best_layer = 'None'
    
    return accuracy, best_layer


def cor_eval(detector, dataloader, per_layer=False):
    cache = detector.testset_correlation(dataloader)
    if per_layer:
        corr_coeffs = {}
        for layer in cache:
            corr_coeffs[layer] = cache[layer].compute()
        best_layer         = max(corr_coeffs, key=corr_coeffs.get)
        corr_coeffs        = corr_coeffs[best_layer]
    else:
        corr_coeffs = cache.compute()
        best_layer  = 'None'
        
    return corr_coeffs, best_layer


def detector_eval(detector, dataloader, per_layer=False):
    acc, layer_ood = ood_eval(detector, dataloader, per_layer=per_layer)
    cor, layer_err = cor_eval(detector, dataloader, per_layer=per_layer)
    metrics              = {}
    metrics['acc']       = acc
    metrics['layer_ood'] = layer_ood
    metrics['cor']       = cor
    metrics['layer_err'] = layer_err
    
    return metrics



class PoolingMahalabonisDetector(nn.Module):
    """
    Evaluation class for OOD and ESCE tasks based on https://arxiv.org/abs/2107.05975.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        layer_ids: List[str], 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        net_out: str,
        criterion: nn.Module = DiceScoreCalgary(),
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.device       = device
        self.model        = model.to(device)
        self.model.eval()
        self.layer_ids    = layer_ids
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.net_out      = net_out
        self.criterion    = criterion
        self.pool         = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.auroc        = AUROC(task = 'binary')
        
        # Init score dict for each layer:
        self.latents   = {layer_id: [] for layer_id in self.layer_ids}
        self.mu        = {layer_id: None for layer_id in self.layer_ids}
        self.sigma_inv = {layer_id: None for layer_id in self.layer_ids}
        self.dist      = {layer_id : 0 for layer_id in self.layer_ids}
        
        self._get_latents()
        self._fit_gaussian_to_latents()
        
        
    @torch.no_grad()
    def _get_hook_fn(self, layer_id: str, mode: str = 'collect') -> Callable:
        
        def hook_fn(module: nn.Module, x: Tuple[Tensor]):
            x = x[0]
            while torch.prod(torch.tensor(x.shape[1:])) > 1e4:
                x = self.pool(x)
            x = self.pool(x)
            batch_size = x.shape[0]

            if mode == 'collect':
                self.latents[layer_id].append(x.view(batch_size, -1).detach().cpu())
            elif mode == 'single':
                self.dist[layer_id] = x.view(batch_size, -1).to(self.device)
                
        return hook_fn
    
    
    @torch.no_grad()        
    def _get_latents(self) -> None:
        handles = {}
        for layer_id in self.layer_ids:
            layer = self.model.get_submodule(layer_id)
            hook  = self._get_hook_fn(layer_id, mode='collect')
            handles[layer_id] = layer.register_forward_pre_hook(hook)

        for batch in self.train_loader:
            input_ = batch['input'].to(self.device)
            _ = self.model(input_)
                
        for layer_id in handles:
            self.latents[layer_id] = torch.cat(self.latents[layer_id], dim=0)
            handles[layer_id].remove()
        
        
    @torch.no_grad()         
    def _fit_gaussian_to_latents(self) -> None:
        for layer_id in self.layer_ids:
            self.mu[layer_id] = self.latents[layer_id].mean(0, keepdims=True).to(self.device)
            latents_centered = (self.latents[layer_id] - self.mu[layer_id].cpu()).detach().numpy()
            sigma = torch.from_numpy(LedoitWolf().fit(latents_centered).covariance_)
            self.sigma_inv[layer_id] = torch.linalg.inv(sigma).unsqueeze(0).to(self.device)
            
            
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        
        self.pred = {}
        self.target = {}
        
        valid_dists = {layer_id : [] for layer_id in self.layer_ids}
        for batch in self.valid_loader:
            input_ = batch['input']
            #print(input_.shape)
            if self.net_out == 'calgary':
                dist_volume = []
                for input_chunk in input_:
                    dist, _ = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    dist_volume.append(dist.copy())
                dist = default_collate(dist_volume)
            elif self.net_out == 'mms': 
                dist, _ = self.forward(input_.to(self.device))
            for layer_id in self.layer_ids:
                if self.net_out == 'calgary':
                    valid_dists[layer_id].append(dist[layer_id].mean())
                elif self.net_out == 'mms':
                    valid_dists[layer_id].append(dist[layer_id])
        self.valid_dists = valid_dists
        self.valid_labels = {layer_id: torch.zeros(len(self.valid_dists[layer_id]), dtype=torch.uint8) 
                             for layer_id in self.layer_ids}
        #print(len(self.valid_dists['up3']), len(self.valid_labels['up3']))
            
#             self.thresholds = {layer_id : 0 for layer_id in self.layer_ids}
#             for layer_id in self.layer_ids:
#                 if self.net_out == 'calgary':
#                     valid_dists[layer_id] = torch.tensor(valid_dists[layer_id]).cpu()
#                 elif self.net_out == 'mms':
#                     valid_dists[layer_id] = torch.cat(valid_dists[layer_id], dim=0).cpu()
#                 self.thresholds[layer_id] = torch.sort(valid_dists[layer_id])[0][len(valid_dists[layer_id]) - (len(valid_dists[layer_id]) // 20) - 1]
                
                    
        test_dists = {layer_id : [] for layer_id in self.layer_ids}
        for batch in test_loader:
            input_ = batch['input']
            if self.net_out == 'calgary':
                dist_volume = []
                for input_chunk in input_:
                    dist, _ = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    dist_volume.append(dist.copy())
                dist = default_collate(dist_volume)
            elif self.net_out == 'mms': 
                dist, _ = self.forward(input_.to(self.device))
            for layer_id in self.layer_ids:
                if self.net_out == 'calgary':
                    test_dists[layer_id].append(dist[layer_id].mean())
                elif self.net_out == 'mms':    
                    test_dists[layer_id].append(dist[layer_id])
        
        self.test_dists = test_dists
        self.test_labels = {layer_id: torch.ones(len(self.test_dists[layer_id]), dtype=torch.uint8) 
                             for layer_id in self.layer_ids}
            
            
        AUROC = {layer_id : 0 for layer_id in self.layer_ids}
        for layer_id in self.layer_ids:
            if self.net_out == 'calgary':
                self.valid_dists[layer_id] = torch.tensor(self.valid_dists[layer_id]).cpu()
                self.test_dists[layer_id]  = torch.tensor(self.test_dists[layer_id]).cpu()
            elif self.net_out == 'mms':
                self.valid_dists[layer_id] = torch.cat(self.valid_dists[layer_id], dim=0).cpu()
                self.test_dists[layer_id]  = torch.cat(self.test_dists[layer_id], dim=0).cpu()
            self.pred[layer_id]   = torch.cat([self.valid_dists[layer_id], self.test_dists[layer_id]]).squeeze()
            self.target[layer_id] = torch.cat([self.valid_labels[layer_id], self.test_labels[layer_id]]).squeeze()
            
            print(self.pred[layer_id].shape, self.target[layer_id].shape)
            
            AUROC[layer_id] = self.auroc(self.pred[layer_id], self.target[layer_id])
            #accuracy[layer_id] = ((test_dists[layer_id] > self.thresholds[layer_id]).sum() / len(test_dists[layer_id]))
                
        return AUROC
    
    
    
    @torch.no_grad()        
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeffs = {layer_id: SpearmanCorrCoef() for layer_id in self.layer_ids}
        for batch in test_loader:
            input_ = batch['input']
            target = batch['target']
            if self.net_out == 'calgary':
                dist_volume = []
                net_out_volume = []
                for input_chunk in input_:
                    dist, net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    dist_volume.append(dist.copy())
                    net_out_volume.append(net_out.cpu())
                dist = default_collate(dist_volume)            
                net_out = torch.cat(net_out_volume, dim=0)
            
            if self.net_out == 'mms':
                target[target == -1] = 0
                # convert to one-hot encoding
                target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
                dist, net_out = self.forward(input_.to(self.device))            
            loss = self.criterion(net_out.cpu(), target)

            loss = loss.mean().float().cpu()
            for layer_id in self.layer_ids:
                corr_coeffs[layer_id].update(dist[layer_id].cpu().mean().view(1), 1-loss.view(1))

        return corr_coeffs


    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        
        handles = {}
        for layer_id in self.layer_ids:
            layer = self.model.get_submodule(layer_id)
            hook  = self._get_hook_fn(layer_id, mode='single')
            handles[layer_id] = layer.register_forward_pre_hook(hook)
        
        net_out = self.model(input_)
        
        for layer_id in self.layer_ids:
            latent_centered = self.dist[layer_id].view(self.dist[layer_id].shape[0], 1, -1) - \
                self.mu[layer_id].unsqueeze(0)
            self.dist[layer_id] = latent_centered @ self.sigma_inv[layer_id] @ \
                latent_centered.permute(0,2,1)
            handles[layer_id].remove()
            
        return self.dist, net_out
    
    
    
class AEMahalabonisDetector(nn.Module):
    """
    Evaluation class for OOD and ESCE tasks based on AEs.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        layer_ids: dict, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        net_out: str,
        criterion: nn.Module = DiceScoreCalgary(),
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.device       = device
        self.model        = model.to(device)
        self.layer_ids    = layer_ids
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.net_out      = net_out
        self.criterion    = criterion
        self.auroc        = AUROC(task = 'binary')
        
        # Remove training hooks if necessary
        self.model.remove_all_hooks()
    
        # Put model in evaluation state
        self.model.to(device)
        self.model.eval()
        self.model.freeze_seg_model()
        
        # Init score dict for each layer:
        self.latents = {layer_id: [] for layer_id in self.layer_ids}
        self.mu = {layer_id: None for layer_id in self.layer_ids}
        self.sigma_inv = {layer_id: None for layer_id in self.layer_ids}
        self.dist = {layer_id : 0 for layer_id in self.layer_ids}
        
        self._get_latents()
        self._fit_gaussian_to_latents()

    
    @torch.no_grad()
    def _get_hook_fn(self, layer_id: str, mode: str = 'collect') -> Callable:
        
        def hook_fn(module: nn.Module, x: Tuple[Tensor]):
            #print(x[0].shape)
            #latent = self.model.transformations[layer_id](x[0])
            latent = self.model.transformations[layer_id].get_latent(x[0]).detach().cpu()
            #latent = torch.cat([mu, log_var], dim=-1).detach().cpu()
            if mode == 'collect':
                self.latents[layer_id].append(latent)
            elif mode == 'single':
                self.dist[layer_id] = latent.to(self.device)

        return hook_fn
        
        
    @torch.no_grad()        
    def _get_latents(self) -> None:
        handles = {}
        for layer_id in self.layer_ids:
            layer = self.model.seg_model.get_submodule(layer_id)
            hook  = self._get_hook_fn(layer_id, mode='collect')
            handles[layer_id] = layer.register_forward_pre_hook(hook)

        for batch in self.train_loader:
            input_ = batch['input'].to(self.device)
            _ = self.model.seg_model(input_)
                
        for layer_id in handles:
            self.latents[layer_id] = torch.cat(self.latents[layer_id], dim=0)
            handles[layer_id].remove()
        
        
    @torch.no_grad()         
    def _fit_gaussian_to_latents(self) -> None:
        for layer_id in self.layer_ids:
            self.mu[layer_id] = self.latents[layer_id].mean(0, keepdims=True).to(self.device)
            latents_centered = (self.latents[layer_id] - self.mu[layer_id].cpu()).detach().numpy()
            sigma = torch.from_numpy(LedoitWolf().fit(latents_centered).covariance_)
            self.sigma_inv[layer_id] = torch.linalg.inv(sigma).unsqueeze(0).to(self.device)
            
            
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        
        self.pred = {}
        self.target = {}
        if not hasattr(self, 'valid_dists'):
            valid_dists = {layer_id : [] for layer_id in self.layer_ids}
            for batch in self.valid_loader:
                input_ = batch['input']
                #print(input_.shape)
                if self.net_out == 'calgary':
                    dist_volume = []
                    for input_chunk in input_:
                        dist, _ = self.forward(input_chunk.unsqueeze(0).to(self.device))
                        dist_volume.append(dist.copy())
                    dist = default_collate(dist_volume)
                elif self.net_out == 'mms': 
                    dist, _ = self.forward(input_.to(self.device))
                for layer_id in self.layer_ids:
                    if self.net_out == 'calgary':
                        valid_dists[layer_id].append(dist[layer_id].mean())
                    elif self.net_out == 'mms':
                        valid_dists[layer_id].append(dist[layer_id])
            self.valid_dists = valid_dists
            self.valid_labels = {layer_id: torch.zeros(len(self.valid_dists[layer_id]), dtype=torch.uint8) 
                                 for layer_id in self.layer_ids}
        #print(len(self.valid_dists['up3']), len(self.valid_labels['up3']))
            
#             self.thresholds = {layer_id : 0 for layer_id in self.layer_ids}
#             for layer_id in self.layer_ids:
#                 if self.net_out == 'calgary':
#                     valid_dists[layer_id] = torch.tensor(valid_dists[layer_id]).cpu()
#                 elif self.net_out == 'mms':
#                     valid_dists[layer_id] = torch.cat(valid_dists[layer_id], dim=0).cpu()
#                 self.thresholds[layer_id] = torch.sort(valid_dists[layer_id])[0][len(valid_dists[layer_id]) - (len(valid_dists[layer_id]) // 20) - 1]
                
                    
        test_dists = {layer_id : [] for layer_id in self.layer_ids}
        for batch in test_loader:
            input_ = batch['input']
            if self.net_out == 'calgary':
                dist_volume = []
                for input_chunk in input_:
                    dist, _ = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    dist_volume.append(dist.copy())
                dist = default_collate(dist_volume)
            elif self.net_out == 'mms': 
                dist, _ = self.forward(input_.to(self.device))
            for layer_id in self.layer_ids:
                if self.net_out == 'calgary':
                    test_dists[layer_id].append(dist[layer_id].mean())
                elif self.net_out == 'mms':    
                    test_dists[layer_id].append(dist[layer_id])
        
        self.test_dists = test_dists
        self.test_labels = {layer_id: torch.ones(len(self.test_dists[layer_id]), dtype=torch.uint8) 
                             for layer_id in self.layer_ids}
            
            
        AUROC = {layer_id : 0 for layer_id in self.layer_ids}
        for layer_id in self.layer_ids:
            if self.net_out == 'calgary':
                self.valid_dists[layer_id] = torch.tensor(self.valid_dists[layer_id]).cpu()
                self.test_dists[layer_id]  = torch.tensor(self.test_dists[layer_id]).cpu()
            elif self.net_out == 'mms':
                self.valid_dists[layer_id] = torch.cat(self.valid_dists[layer_id], dim=0).cpu()
                self.test_dists[layer_id]  = torch.cat(self.test_dists[layer_id], dim=0).cpu()
            self.pred[layer_id]   = torch.cat([self.valid_dists[layer_id], self.test_dists[layer_id]]).squeeze()
            self.target[layer_id] = torch.cat([self.valid_labels[layer_id], self.test_labels[layer_id]]).squeeze()
            #print(self.pred[layer_id].shape, self.target[layer_id].shape)
            AUROC[layer_id] = self.auroc(self.pred[layer_id], self.target[layer_id])
            #accuracy[layer_id] = ((test_dists[layer_id] > self.thresholds[layer_id]).sum() / len(test_dists[layer_id]))
                
        return AUROC
    
    
    @torch.no_grad()        
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeffs = {layer_id: SpearmanCorrCoef() for layer_id in self.layer_ids}
        for batch in test_loader:
            input_ = batch['input']
            target = batch['target']
            if self.net_out == 'calgary':
                dist_volume = []
                net_out_volume = []
                for input_chunk in input_:
                    dist, net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    dist_volume.append(dist.copy())
                    net_out_volume.append(net_out.cpu())
                dist = default_collate(dist_volume)            
                net_out = torch.cat(net_out_volume, dim=0)
            
            elif self.net_out == 'mms': 
                target[target == -1] = 0
                # convert to one-hot encoding
                target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
                dist, net_out = self.forward(input_.to(self.device))            
            loss = self.criterion(net_out.cpu(), target)
            loss = loss.mean().float().cpu()
            for layer_id in self.layer_ids:
                corr_coeffs[layer_id].update(dist[layer_id].cpu().mean(), 1-loss)

        return corr_coeffs


    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> Union[dict, torch.Tensor]:
        
        handles = {}
        for layer_id in self.layer_ids:
            layer = self.model.seg_model.get_submodule(layer_id)
            hook  = self._get_hook_fn(layer_id, mode='single')
            handles[layer_id] = layer.register_forward_pre_hook(hook)
            
        net_out = self.model.seg_model(input_)
        
        for layer_id in self.layer_ids:
            latent_centered = self.dist[layer_id].view(self.dist[layer_id].shape[0], 1, -1) - self.mu[layer_id].unsqueeze(0)
            self.dist[layer_id] = latent_centered @ self.sigma_inv[layer_id] @ latent_centered.permute(0,2,1)
            handles[layer_id].remove()
            
        return self.dist, net_out
    
    
    
class MeanDistSamplesDetector(nn.Module):
    """
    Implements an out-of-distribution (OOD) detection and error correlation evaluation tool for 
    autoencoder-based (AE) feature re-sampling systems. This class computes the mean distance between samples
    and original segmentation maps to quantify uncertainty. Different distance functions and aggregation method
    are available as per their respective class implementation, i.e. UMapGenerator and UMapScorePerSlice.

    Parameters:
    - model: Frankenstein model, i.e. segmentation network with attached AEs
    - net_out: Output type of the network ('calgary', 'mms'; specific to dataset.).
    - valid_loader: DataLoader for validation data.
    - criterion: Criterion for computing the evaluation metric (e.g., DiceScoreCalgary).
    - device: The device to run the model on.
    - umap_method: Method for uncertainty mapping (e.g. cross_entropy)
    - umap_reduction: Reduction method for uncertainty maps (e.g. mean, norm or nflips)
    """
    
    def __init__(
        self,
        model: nn.Module, 
        net_out: str,
        valid_loader: DataLoader,
        criterion: nn.Module,
        umap_method: str,
        umap_reduction: str,
        device: str = 'cuda:0',
    ):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.net_out  = net_out
        self.umap_method = umap_method
        self.umap_reduction = umap_reduction
        
        # Remove trainiung hooks, add evaluation hooks
        self.model.remove_all_hooks()        
        self.model.hook_transformations(self.model.transformations,
                                        n_samples=1)
        
        self.model.eval()
        self.model.freeze_seg_model()
        
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.auroc = AUROC(task = 'binary')
        self.umap_generator = UMapGenerator(method=umap_method,
                                            net_out=net_out)
        self.score_fn = UMapScorePerSlice(reduction=umap_reduction)
        
        
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'threshold'):
            valid_dists = []
            for batch in self.valid_loader:
                input_ = batch['input'].to(0)
                
                if self.net_out == 'calgary':
                    model_out_volume = []
                    umap_volume  = []

                    for input_chunk in input_:
                        umap, model_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                        model_out_volume.append(model_out[:1].detach().cpu())
                        umap_volume.append(umap)

                    model_out = torch.cat(model_out_volume, dim=0)
                    umap = torch.cat(umap_volume, dim=0)
                    
                if self.net_out == 'mms':
                    umap, model_out = self.forward(input_.to(self.device))
                score = torch.norm(umap).cpu()
                valid_dists.append(score)
                
            self.valid_dists = torch.tensor(valid_dists)
            self.valid_labels = torch.zeros(len(self.valid_dists), dtype=torch.uint8)
        
        test_dists = []
        for batch in test_loader:
            input_ = batch['input']

            if self.net_out == 'calgary':
                model_out_volume = []
                umap_volume  = []

                for input_chunk in input_:
                    umap, model_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    model_out_volume.append(model_out[:1].detach().cpu())
                    umap_volume.append(umap)

                model_out = torch.cat(model_out_volume, dim=0)
                umap = torch.cat(umap_volume, dim=0)

            if self.net_out == 'mms':
                umap, model_out = self.forward(input_.to(self.device))

            score = torch.norm(umap).cpu()
            test_dists.append(score)
        self.test_dists = torch.tensor(test_dists).cpu()
        self.test_labels = torch.ones(len(self.test_dists), dtype=torch.uint8)
        
        self.pred =  torch.cat([self.valid_dists, self.test_dists]).squeeze()
        self.target = torch.cat([self.valid_labels, self.test_labels]).squeeze()
        #print(self.pred.shape, self.target.shape)
        AUROC = self.auroc(self.pred, self.target)
        
        return AUROC    
        
    
    @torch.no_grad()
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeff = SpearmanCorrCoef()
        #losses = []
        for batch in test_loader:
            input_ = batch['input']
            target = batch['target']
            
            if self.net_out == 'calgary':
                model_out_volume = []
                umap_volume  = []

                for input_chunk in input_:
                    
                    umap, model_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    model_out_volume.append(model_out[:1].detach().cpu())
                    umap_volume.append(umap)
                    
                model_out = torch.cat(model_out_volume, dim=0)
                umap = torch.cat(umap_volume, dim=0)
#                 print(umap.shape, model_out.shape)
            
            if self.net_out == 'mms':
                target[target == -1] = 0
                # convert to one-hot encoding
                target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
                umap, model_out = self.forward(input_.to(self.device))
            
            
            #print(model_out.shape, target.shape)
            loss = self.criterion(model_out, target)
            loss = loss.mean().float()

#             assert model_out.shape[0] == 2, "Model produces too large outputs"
            score = self.score_fn(
                umap=umap, 
#                 pred=model_out[:1], 
#                 pred_r=model_out[1:]
            )
            
            corr_coeff.update(score.cpu().view(1,), 1-loss.view(1,))
            
        return corr_coeff

    
    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        model_out = self.model(input_).cpu()
        umap    = self.umap_generator(model_out).cpu()
        return umap, model_out   #[:1]
    
    
    
class EntropyDetector(nn.Module):
    """
    Evaluation class for OOD and ESCE tasks based on AEs.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        net_out: str,
        valid_loader: DataLoader,
        criterion: nn.Module,
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.net_out = net_out
        self.device = device
        self.model = model.to(device)
        # Remove trainiung hooks, add evaluation hooks
        # self.model.remove_all_hooks()        
        
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.auroc = AUROC(task = 'binary')
        self.umap_generator = UMapGenerator(method='probs',
                                            net_out=net_out)
        
        
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'valid_dists'):
            valid_dists = []
            for batch in self.valid_loader:
                input_ = batch['input'].to(0)
                
                if self.net_out == 'calgary':
                    net_out_volume = []
                    umap_volume  = []

                    for input_chunk in input_:
                        umap, net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                        net_out_volume.append(net_out.detach().cpu())
                        umap_volume.append(umap)

                    net_out = torch.cat(net_out_volume, dim=0)
                    umap = torch.cat(umap_volume, dim=0)
                    
                if self.net_out == 'mms':
                    umap, net_out = self.forward(input_.to(self.device))
                score = torch.norm(umap).cpu()
                valid_dists.append(score)
                
            self.valid_dists = torch.tensor(valid_dists)
            self.valid_labels = torch.zeros(len(self.valid_dists), dtype=torch.uint8)
        
        test_dists = []
        for batch in test_loader:
            input_ = batch['input']

            if self.net_out == 'calgary':
                net_out_volume = []
                umap_volume  = []

                for input_chunk in input_:
                    umap, net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    net_out_volume.append(net_out.detach().cpu())
                    umap_volume.append(umap)

                net_out = torch.cat(net_out_volume, dim=0)
                umap = torch.cat(umap_volume, dim=0)

            if self.net_out == 'mms':
                umap, net_out = self.forward(input_.to(self.device))

            score = torch.norm(umap).cpu()
            test_dists.append(score)
        self.test_dists = torch.tensor(test_dists).cpu()
        self.test_labels = torch.ones(len(self.test_dists), dtype=torch.uint8)
        
        self.pred =  torch.cat([self.valid_dists, self.test_dists]).squeeze()
        self.target = torch.cat([self.valid_labels, self.test_labels]).squeeze()
        print(self.pred.shape, self.target.shape)
        AUROC = self.auroc(self.pred, self.target)
        
        return AUROC     
        
    
    @torch.no_grad()
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeff = SpearmanCorrCoef()
        losses = []
        for batch in test_loader:
            input_ = batch['input'].to(0)
            target = batch['target']
            
            if self.net_out == 'calgary':
                net_out_volume = []
                umap_volume  = []

                for input_chunk in input_:
                    umap, net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    net_out_volume.append(net_out[:1].detach().cpu())
                    umap_volume.append(umap)
                    
                net_out = torch.cat(net_out_volume, dim=0)
                umap = torch.cat(umap_volume, dim=0)
            
            if self.net_out == 'mms':
                target[target == -1] = 0
                # convert to one-hot encoding
                target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
                umap, net_out = self.forward(input_.to(self.device))
            
            
            loss = self.criterion(net_out.cpu(), target.cpu())
            loss = loss.mean().cpu().float()
                    
            score = torch.norm(umap)
            losses.append(1-loss.view(1))
            corr_coeff.update(score.cpu(), 1-loss)
            
        return corr_coeff

    
    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        net_out = self.model(input_)
        umap = self.umap_generator(net_out)
        #score = torch.linalg.norm(umap, dim=(-2, -1))
        return umap, net_out
    
    
class DropoutEntropyDetector(nn.Module):
    """
    Evaluation class for OOD and ESCE tasks based on VAEs.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        n_samples: int,
        data: str,
        valid_loader: DataLoader,
        criterion: nn.Module = DiceScoreMMS(),
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.n_samples = n_samples
        # Remove trainiung hooks, add evaluation hooks
        # self.model.remove_all_hooks()        
        
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.umap_generator = UMapGenerator(method='entropy',
                                            net_out=data)
        
        
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'threshold'):
            valid_dists = []
            for batch in self.valid_loader:
                input_ = batch['input'].to(0)
                dist, _ = self.forward(input_)
                valid_dists.append(dist)
                    
            self.threshold = 0
            valid_dists = torch.cat(valid_dists, dim=0).cpu()
            self.threshold = torch.sort(valid_dists)[0][len(valid_dists) - (len(valid_dists) // 20)]
                    
        test_dists = []
        for batch in test_loader:
            input_ = batch['input'].to(self.device)
            dist, _ = self.forward(input_)
            test_dists.append(dist)
            
        test_dists = torch.cat(test_dists, dim=0).cpu()
        
        accuracy = (test_dists > self.threshold).sum() / len(test_dists)                
        return accuracy    
        
    
    @torch.no_grad()
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeff = SpearmanCorrCoef()
        for batch in test_loader:
            input_ = batch['input'].to(0)
            target = batch['target'].to(0)
            score, net_out = self.forward(input_)
            loss = self.criterion(net_out, target)
            if len(loss.shape) > 1:
                loss = loss.mean(dim=1)
            corr_coeff.update(score.view(1,), 1-loss)
            
        return corr_coeff

    
    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        net_out = self.model(input_)
        
        self.model.apply(_activate_dropout)            
        B, C, H, W = net_out.size()
        samples    = torch.zeros((self.n_samples, C, H, W), device=self.device)
        for i in range(self.n_samples):
            samples[i] = self.model(input_)
        umap  = self.umap_generator(samples)
        score = torch.linalg.norm(umap, dim=(-2, -1))
        return score, net_out
    
    
class EnsembleEntropyDetector(nn.Module):
    """
    Evaluation class for OOD and ESCE tasks based on AEs.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        net_out: str,
        valid_loader: DataLoader,
        criterion: nn.Module,
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.net_out = net_out
        self.device = device
        self.model = model.to(device)
        torch.manual_seed(42)
        self.ensemble_compositions = torch.cat([torch.arange(10).view(10,1), 
                                                torch.randint(0, 9, (10, 4))],
                                               dim=1)
        # Remove trainiung hooks, add evaluation hooks
        # self.model.remove_all_hooks()        
        
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.auroc = AUROC(task = 'binary')
        self.umap_generator = UMapGenerator(method='probs',
                                            net_out=net_out)
        
        
    @torch.no_grad()
    def testset_ood_detection(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        
        if not hasattr(self, 'valid_dists'):
            valid_dists = []
            for batch in self.valid_loader:
                input_ = batch['input'].to(0)
                net_out = self.forward(input_.to(self.device))
                umap = self.umap_generator(net_out)
                scores = (umap**2).mean(dim=(1,2)).cpu()
                valid_dists.append(scores)
            self.valid_dists = torch.stack(valid_dists, dim=0)
            self.valid_labels = torch.zeros(len(self.valid_dists), dtype=torch.uint8)
            
        test_dists = []
        for batch in test_loader:
            input_ = batch['input']
            net_out = self.forward(input_.to(self.device))
            umap = self.umap_generator(net_out)
            scores = (umap**2).mean(dim=(1,2)).cpu()
            test_dists.append(scores)
            
        self.test_dists = torch.stack(test_dists, dim=0).cpu()
        self.test_labels = torch.ones(len(self.test_dists), dtype=torch.uint8)
        
        
        self.pred = torch.cat([self.valid_dists, self.test_dists], dim=0).T
        self.target = torch.cat([self.valid_labels, self.test_labels])
        
        #print(ensemble_compositions[0], self.pred[ensemble_compositions[0]].mean(0).shape, self.target.shape)
        
        AUROC = [self.auroc(self.pred[self.ensemble_compositions[i]].mean(0), self.target) 
                 for i in range(10)]
        
        return AUROC
        #accuracy = (test_dists > self.thresholds).sum(dim=0) / len(test_dists)
        
        #return accuracy       
        
    
    @torch.no_grad()
    def testset_correlation(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        corr_coeffs = [SpearmanCorrCoef() for _ in range(10)]
        losses = []
        for batch in tqdm(test_loader):
            input_ = batch['input'].to(0)
            target = batch['target']
            
            if self.net_out == 'calgary':
                net_out_volume = []

                for input_chunk in input_:
                    net_out = self.forward(input_chunk.unsqueeze(0).to(self.device))
                    net_out_volume.append(net_out.detach().cpu())
                net_out = torch.stack(net_out_volume, dim=0)

            if self.net_out == 'mms':
                target[target == -1] = 0
                # convert to one-hot encoding
                target = F.one_hot(target.long(), num_classes=4).squeeze(1).permute(0,3,1,2)
                net_out = self.forward(input_.to(self.device))
            
            
            for i, corr_coeff in enumerate(corr_coeffs):
                ensemble_idxs = self.ensemble_compositions[i]
                
                if self.net_out == 'calgary':
                    loss = self.criterion(net_out[:, i:i+1].cpu(), target.cpu())
                    umap_volume  = []
                    for slc in net_out:
                        umap = self.umap_generator(slc[ensemble_idxs].mean(0, keepdim=True))
                        umap_volume.append(umap)
                    umap = torch.cat(umap_volume, dim=0)
                
                if self.net_out == 'mms':
                    #loss = self.criterion(net_out[i:i+1].cpu(), target.cpu())
                    loss = self.criterion(net_out[ensemble_idxs].mean(dim=0, keepdim=True).cpu(), target.cpu())
                    umap = self.umap_generator(net_out[ensemble_idxs].mean(0, keepdim=True))
                    
                score = torch.norm(umap).cpu()    
                loss = loss.mean().cpu().float()
                corr_coeff.update(score, 1-loss)
            
        return corr_coeffs

    
    @torch.no_grad()  
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        net_out = self.model(input_)
        return net_out