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
from utils import EarlyStopping, epoch_average, average_metrics
from model.dae import AugResDAE
from model.unet import UNet2D
from model.wrapper import Frankenstein
from losses import MNMCriterionAE, SampleDice, UnetDice
from trainer.ae_trainer import AETrainerACDC

nnUnet_prefix = '../../../nnUNet/'



# define single image dataloader from batchgenerator example here:
# https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
class SingleImageMultiViewDataLoader(batchgenerators.dataloading.data_loader.SlimDataLoaderBase):
    def __init__(self, data: ACDCDataset, batch_size: int = 2, return_orig: str = False):
        super(SingleImageMultiViewDataLoader, self).__init__(data, batch_size)
        # data is now stored in self._data.
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        
        data = self._data[randrange(len(self._data))]
        img = data['input'].numpy().astype(np.float32)
        tar = data['target'][0].numpy().astype(np.float32)
        
        img_batched = np.tile(img, (self.batch_size, 1, 1, 1))
        tar_batched = np.tile(tar, (self.batch_size, 1, 1, 1))
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img_batched, 
               'seg':  tar_batched}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = data['input'].unsqueeze(0)
            out['target_orig'] = data['target'].unsqueeze(0)
        
        return out
    
    
class MultiImageSingleViewDataLoader(batchgenerators.dataloading.data_loader.SlimDataLoaderBase):
    def __init__(self, data: ACDCDataset, batch_size: int = 2, return_orig: str = False):
        super(MultiImageSingleViewDataLoader, self).__init__(data, batch_size)
        # data is now stored in self._data.
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        sample = torch.randint(0, len(self._data), size=(self.batch_size,))
        data   = self._data[sample]
        img    = data['input']
        tar    = data['target']
        #print(tar.shape, img.shape)
#         img_batched = np.tile(img, (self.batch_size, 1, 1, 1))
#         tar_batched = np.tile(tar, (self.batch_size, 1, 1, 1))
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img.numpy().astype(np.float32), 
               'seg':  tar.numpy().astype(np.float32)}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = img
            out['target_orig'] = tar
        
        return out
    
    
import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable, Tuple, Union
from copy import deepcopy

class Frankenstein(nn.Module):
    """Wrapper class for segmentation models and feature transformations.

    Wraps (a copy of) the segmentation model and attaches feature
    trasformations to it via hooks (at potentially various positions
    simultaneously). Additionally, it provides control utilities for the
    hooks as well as different types for inference and training.
    """

    def __init__(
        self,
        seg_model: nn.Module,
        transformations: nn.ModuleDict,
        disabled_ids: list = [],
        copy: bool = True,
    ):
        super().__init__()
        self.seg_model = deepcopy(seg_model) if copy else seg_model

        self.transformations = transformations
        self.disabled_ids = disabled_ids
        self.transformation_handles = {}
        self.train_transformation_handles = {}
        self.inspect_transformation_handles = {}
        self.training_data = {}
        self.inspect_data = {}

    def hook_train_transformations(self, transformations: Dict[str, nn.Module]) -> None:
        for layer_id in transformations:
            layer = self.seg_model.get_submodule(layer_id)
            hook = self._get_train_transformation_hook(
                transformations[layer_id], layer_id
            )
            self.train_transformation_handles[
                layer_id
            ] = layer.register_forward_pre_hook(hook)

    def hook_transformations(
        self, transformations: Dict[str, nn.Module], n_samples: int
    ) -> None:
        for layer_id in transformations:
            layer = self.seg_model.get_submodule(layer_id)
            hook = self._get_transformation_hook(transformations[layer_id], n_samples)
            self.transformation_handles[layer_id] = layer.register_forward_pre_hook(
                hook
            )
            
    def hook_inspect_transformation(
        self, 
        transformations: Dict[str, nn.Module], 
        n_samples: int,
        arch: str = 'ae'
    ) -> None:
        for layer_id in transformations:
            if layer_id not in self.disabled_ids:
                layer = self.seg_model.get_submodule(layer_id)
                hook  = self._get_inspect_transformation_hook(transformations[layer_id], layer_id, n_samples, arch)
                self.inspect_transformation_handles[layer_id] = layer.register_forward_pre_hook(hook)
            

#     def _get_train_transformation_hook(
#         self, transformation: nn.Module, layer_id: str
#     ) -> Callable:
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # tuple, alternatively use x_in = x[0]
#             x_orig = x_in[:1]
#             #x_views = x_in[1:]
#             x_in_denoised = transformation(x_in)
            
#             if layer_id not in self.disabled_ids:
#                 mse = nn.functional.mse_loss(x_in_denoised, x_orig.detach(), reduction="mean")

#                 training_data = {
#                     "mse": mse,
#                 }

#                 self.training_data[layer_id] = training_data

#             #return torch.cat([x_orig, x_in_denoised], dim=0)
#             return x_in_denoised
            
#         return hook
    
    
    def _get_train_transformation_hook(
        self,
        transformation: nn.Module,
        layer_id: str
    ) -> Callable:
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # tuple, alternatively use x_in = x[0]
            #print(x_in.shape, x_in.shape[0] // 2)
            batch_size = x_in.shape[0] // 2
            x_orig, _  = torch.split(x_in, batch_size)
            x_in_denoised = transformation(x_in)
            
            if layer_id not in self.disabled_ids:
                mse = nn.functional.mse_loss(
                    x_in_denoised, 
                    x_orig.repeat(2,1,1,1).detach(),
                    reduction="mean"
                )

                training_data = {
                    "mse": mse,
                }

                self.training_data[layer_id] = training_data

            return torch.cat([x_orig, x_in_denoised[batch_size:]], dim=0)
            #return x_in_denoised
            
        return hook
    

    def _get_transformation_hook(
        self, transformation: nn.Module, n_samples: int = 1
    ) -> Callable:
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # weird tuple, can use x_in = x[0]
            if n_samples == 0:
                return x
            elif n_samples == -1:
                x_in_new = transformation(x_in)
                return x_in_new
            else:
                x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
                x_in_new = transformation(x_in_new)
                return torch.cat([x_in, x_in_new], dim=0)

        return hook
            
        
    def _get_inspect_transformation_hook(
            self, 
            transformation: nn.Module, 
            layer_id: str, 
            n_samples: int,
            arch: str = 'ae',
        ) -> Callable:
        
        @torch.no_grad()
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # weird tuple, can use x_in = x[0]
            if n_samples == 0:
                return x
            elif n_samples == -1:
                mu, log_var, x_in_new = transformation(x_in)
            else:
                x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
                if arch == 'ae':
                    x_in_new = transformation(x_in_new)
                elif arch == 'res_ae':
                    x_in_new, prior, residual = transformation(x_in_new)
                x_in_new = torch.cat([x_in, x_in_new], dim=0)
                
            if layer_id not in self.disabled_ids:
                training_data = {
                    'input'  : x_in_new[ :1],
                    'recon'  : x_in_new[1: ],
                }
                
                if arch == 'res_ae':
                    training_data['prior'] = prior
                    training_data['residual'] = residual
                
                self.inspect_data[layer_id] = training_data
            
            return x_in_new
        
        return hook
   
    

    def remove_train_transformation_hook(self, layer_id: str) -> None:
        self.train_transformation_handles[layer_id].remove()

    def remove_transformation_hook(self, layer_id: str) -> None:
        self.transformation_handles[layer_id].remove()
        
    def remove_inspect_transformation_hook(self, layer_id: str) -> None:
        self.inspect_transformation_handles[layer_id].remove()

    def remove_all_hooks(self):
        if hasattr(self, "train_transformation_handles"):
            for handle in self.train_transformation_handles:
                self.train_transformation_handles[handle].remove()
            self.train_transformation_handles = {}

        if hasattr(self, "transformation_handles"):
            for handle in self.transformation_handles:
                self.transformation_handles[handle].remove()
            self.transformation_handles = {}
            
        if hasattr(self, 'inspect_transformation_handles'):
            for handle in self.inspect_transformation_handles:
                self.inspect_transformation_handles[handle].remove()
            self.inspect_transformation_handles = {}
        

    def freeze_seg_model(self):
        self.seg_model.eval()
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def set_number_of_samples_to(self, n_samples: int):
        self.n_samples = n_samples

    def disable(self, layer_ids: list) -> None:
        for layer_id in layer_ids:
            self.transformations[layer_id].turn_off()

    def enable(self, layer_ids: list) -> None:
        for layer_id in layer_ids:
            self.transformations[layer_id].turn_on()

    def forward(self, x: Tensor):
        return self.seg_model(x)

    
import os, sys
import time
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision.transforms import Resize, CenterCrop
from typing import Iterable, Dict, Callable, Tuple
import matplotlib.pyplot as plt

import wandb
from tqdm.auto import tqdm
from utils import EarlyStopping, epoch_average, average_metrics

class AETrainerACDC:
    def __init__(
        self, 
        model: nn.Module, 
        unet: nn.Module, 
        criterion: Callable, 
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_batches_per_epoch: int, 
        num_val_batches_per_epoch: int, 
        description: str,
        root: str,
        target: str = 'output', #gt
        lr: float = 5e-4,
        n_epochs: int = 1000, 
        patience: int = 5, 
        es_mode: str = 'min', 
        eval_metrics: Dict[str, nn.Module] = None, 
        log: bool = True,
        debug: bool = False, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device       = device
        self.model        = model.to(self.device)
        self.unet         = unet.to(self.device)
        self.criterion    = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_val_batches_per_epoch = num_val_batches_per_epoch
        self.root         = root
        self.description  = description
        self.target       = target
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.es_mode      = es_mode
        self.eval_metrics = eval_metrics
        self.log          = log
        self.debug        = debug
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler    = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience)
        self.es           = EarlyStopping(mode=self.es_mode, patience=2*self.patience)
        self.scaler       = GradScaler()
        self.history      = {'train loss': [], 'valid loss' : [], 'train metrics': [], 'valid metrics': []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.crop   = CenterCrop([256, 256])
        
        
        if target == 'output':
            assert self.criterion.id_loss == 'huber'
        elif target == 'gt':
            assert self.criterion.id_loss in ['ce', 'bce']

        
#     def inference_step(self, x):
#         x = x.to(self.device)
#         with torch.no_grad():
#             unet_out = self.unet(x).detach()
#         samples  = self.model(x)
#         return unet_out, samples

    def inference_step(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0] // 2
#         with torch.no_grad():
#             unet_out = self.unet(x).detach()
        unet_out, samples = torch.split(self.model(x), batch_size)
        assert unet_out.shape == samples.shape, "Shapes dont match between unet out and denoised"
        return unet_out, samples
    

    def train_epoch(self):
        loss_list, metric_list, batch_sizes = [], [], []
        
        for it in range(self.num_batches_per_epoch):
            batch = self.train_loader.next()
            input_ = torch.cat([batch['data_orig'], batch['data']], dim=0)
            input_ = self.crop(input_)

            with autocast():
                unet_out, samples = self.inference_step(input_)
                
                if self.target == 'output':
                    loss, metrics = self.criterion(unet_out, samples, 
                                                  self.model.training_data)
                elif self.target == 'gt':
                    # take only first element from targets. Others are for multi scale supervision
                    #target = batch['target'][0].long().cuda()
                    #target = torch.cat([batch['target_orig'], batch['target'][0]], dim=0)
                    target = batch['target']
                    target = target.long().to(self.device)
                    target[target == -1] = 0
                    target = F.one_hot(target, num_classes=4).squeeze(1).permute(0,3,1,2)
                    target = self.crop(target)
                    loss, metrics = self.criterion(target.to(self.device), samples, 
                                                   self.model.training_data)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_list.append(loss.item())
            metric_list.append(metrics)
            batch_sizes.append(input_.shape[0])
            
        average_loss = epoch_average(loss_list, batch_sizes)
        metrics      = average_metrics(metric_list, batch_sizes)
#         print(metrics['output_diff'])
        
        self.history['train loss'].append(average_loss)
        self.history['train metrics'].append(metrics)
        
        if self.log:
            wandb.log({
                'train_loss': average_loss,
                'train_metrics': metrics
            }, commit=False)

        return average_loss
    
    
    @torch.no_grad()
    # Same story as in train_epoch
    def eval_epoch(self):
        loss_list, metric_list, batch_sizes = [], [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for it in range(self.num_val_batches_per_epoch):
            batch = self.valid_loader.next()
            input_ = torch.cat([batch['data_orig'], batch['data']], dim=0)
            input_ = self.crop(input_)
            #print(input_.shape)
            #print(batch['target_orig'].shape, batch['target'].shape)
#             for ele in batch['target_orig']:
#                 print(ele.shape)
            # take only first element from targets. Others are for multi scale supervision
            #target = torch.cat([batch['target_orig'], batch['target']], dim=0)
            target = batch['target']
            target = target.long().to(self.device)
            #print("target -1", target.min())
            target[target == -1] = 0
            target = F.one_hot(target, num_classes=4).squeeze(1).permute(0,3,1,2)
            target = self.crop(target)
            #print(target.shape, input_.get_device())

            unet_out, samples = self.inference_step(input_)
             
            #print(unet_out.shape, samples.shape)
            if self.target == 'output':
                loss, metrics = self.criterion(
                    unet_out, 
                    samples, 
                    self.model.training_data)
                
            elif self.target == 'gt':
                loss, metrics = self.criterion(
                    target.to(self.device), 
                    samples, 
                    self.model.training_data)

            loss_list.append(loss.item())
            metric_list.append(metrics)
            batch_sizes.append(input_.shape[0])
            
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    #print(unet_out.shape, samples.shape, target.shape)
                    epoch_metrics[key].append(metric(unet_out, samples, target).mean().detach().cpu())
        
#         print("hi")
#         print(batch_sizes)
#         print("eval before avrg", metrics['output_diff'], batch_sizes)
        average_loss = epoch_average(loss_list, batch_sizes)
        metrics      = average_metrics(metric_list, batch_sizes)
#         print("eval after avrg", metrics['output_diff'])
        self.history['valid loss'].append(average_loss)
        self.history['valid metrics'].append(metrics)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                avrg = epoch_average(epoch_scores, batch_sizes)
                self.history[key].append(avrg)
                if self.log:
                    wandb.log({
                        key: avrg
                    }, commit=False)
        
        if self.log:
            wandb.log({
                'valid_loss': average_loss,
                'valid_metrics': metrics
            }, commit=False)
        
        return average_loss
    
    
    def save_hist(self):
        if(not os.path.exists(self.root+'results-tmp/trainer_logs')):
            os.makedirs(self.root+'results-tmp/trainer_logs')
        savepath = f'{self.root}results-tmp/trainer_logs/{self.description}.npy'
        np.save(savepath, self.history)
        
        return
    
    
    def save_model(self):
        if(not os.path.exists(self.root+'pre-trained-tmp/trained_AEs')):
            os.makedirs(self.root+'pre-trained-tmp/trained_AEs')
        if(not os.path.exists(self.root+'results-tmp/trainer_logs')):
            os.makedirs(self.root+'results-tmp/trainer_logs')
        savepath = f'{self.root}pre-trained-tmp/trained_AEs/{self.description}_best.pt'
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_hist()
        
        return
    
    
    def load_model(self):
        savepath = f'{self.root}pre-trained-tmp/trained_AEs/{self.description}_best.pt'
        print(savepath)
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f'{self.root}results-tmp/trainer_logs/{self.description}.npy'
        self.history = np.load(savepath,allow_pickle='TRUE').item()
        
        return
    
    
    def plot_history(self):
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(12,8)
        fig.suptitle(self.description, fontsize=15)
        ax[0].plot(self.history['train loss'], label='train loss', c='lightcoral', lw=3)
        ax[0].plot(self.history['valid loss'], label='valid loss', c='cornflowerblue', lw=3)
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend()
        ax[0].set_title('Training and Validation Losses')
        
        loss_array = np.array(self.history['Sample Volumetric Dice'])
        ax[1].plot(self.history['Sample Volumetric Dice'], label='Sample', c='tab:blue', lw=3)
        ax[1].plot(self.history['UNet Volumetric Dice'], label='UNet', c='tab:red', lw=3)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Volumetric Dice")
        ax[1].legend(loc="lower right")
        ax[1].set_title(f'Volumetric Dice of Validation Set\nBest Sample Dice: {loss_array.max()} @ Epoch {loss_array.argmax()+1}')
        #plt.show()
        
        return fig

    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        
        self.model.remove_all_hooks()
        self.model.training_data = {}
        self.model.hook_train_transformations(self.model.transformations)
        self.unet.eval()
        self.model.freeze_seg_model()

        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        
        if self.log:
            wandb.log({}, commit=True)
        
        for epoch in progress_bar:
            #print(f"Model param sum: {torch.tensor([param.sum() for param in trainer.network.parameters()]).sum()}")

            
            self.model.train()
            self.model.freeze_seg_model()
            
            train_loss = self.train_epoch()
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)
            
            #epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            keys = ['train loss', 'valid loss', 'Sample Volumetric Dice', 'UNet Volumetric Dice']
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in keys] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]
            
            if self.log:
                wandb.log({}, commit=True)
            
            if self.es_mode == 'min':
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    if not self.debug:
                        self.save_model()
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    if not self.debug:
                        self.save_model()
            if(self.es.step(es_metric)):
                print('Early stopping triggered!')
                break
                
        self.training_time = time.time() - self.training_time
        if not self.debug:
            self.save_hist()
        self.load_model()
    

    
def main(args):    
    
    arguments = collections.deque(args)
    while arguments:
        arg = arguments.popleft()
        if arg in ['-i', '--iteration']:
            it = arguments.popleft()
            
    cfg = {
        'debug': False,
        'log': True,
        'description': f'acdc_AugResDAE{it}_localAug_multiImgSingleView_res_balanced', #'mms_vae_for_nnUNet_fc3_0_bs50',
        'project': 'MICCAI2023-loose_ends',

        # Data params
        'n': 0,
        'root': '../../',
        'data_path': 'data/mnm/',
        #'train_vendor': 'B',
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
        'disabled_ids': ['shortcut0', 'shortcut1', 'shortcut2'], #['shortcut0', 'shortcut1', 'shortcut2']
    }

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
    train_gen = MultiThreadedAugmenter(train_gen, train_augmentor, 1, 1, seeds=None)
    valid_gen = MultiImageSingleViewDataLoader(valid_data, batch_size=cfg['batch_size'], return_orig=True)
    #valid_gen = SingleThreadedAugmenter(valid_gen, valid_augmentor)
    valid_gen = MultiThreadedAugmenter(valid_gen, valid_augmentor, 1, 1, seeds=None)


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


    DAEs = nn.ModuleDict({key: AugResDAE(in_channels = dae_map[key][0], 
                                         in_dim      = dae_map[key][1],
                                         latent_dim  = dae_map[key][2],
                                         depth       = dae_map[key][3],
                                         block_size  = dae_map[key][4]) for key in dae_map})

    for layer_id in cfg['disabled_ids']:
         DAEs[layer_id] = nn.Identity()


    model = Frankenstein(unet, 
                         DAEs, 
                         disabled_ids=cfg['disabled_ids'],
                         copy=True)
    #TODO
    #model.cuda()
    print()
    if cfg['log']:
        wandb.watch(model)

    criterion = MNMCriterionAE(
        loss=cfg['loss'], 
        recon=cfg['reconstruction'], 
        diff=cfg['difference']
    )

    eval_metrics = {'Sample Volumetric Dice': SampleDice(data='MNM'),
                    'UNet Volumetric Dice': UnetDice(data='MNM')}


    ae_trainer = AETrainerACDC(model=model, 
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
                                device=torch.device('cuda'))

    #vae_trainer.device = torch.device("cpu")
    ae_trainer.fit()
    
    
if __name__ == '__main__':    
    main(sys.argv[1:])