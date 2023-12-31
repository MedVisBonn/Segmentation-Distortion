## UNet Trainer

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from typing import Dict, Callable
import time
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
import numpy as np
from dataset import CalgaryCampinasDataset
from utils import (
    EarlyStopping, 
    epoch_average, 

)
from data_utils import (
    slice_selection, 
    dataset_from_indices,
    volume_collate
)

class UNetTrainerCalgary():
    def __init__(
        self, 
        model: nn.Module, 
        criterion: Callable, 
        train_generator: DataLoader,
        valid_loader: DataLoader, 
        root: str, 
        description: str = 'untitled', 
        lr: float = 1e-4, 
        n_epochs: int = 250,
        num_batches_per_epoch: int = 50,
        patience: int = 5, 
        #warm_up=True,
        es_mode: str = 'min', 
        eval_metrics: Dict[str, nn.Module] = None,
        log: bool = True
    ):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = model.to(self.device)
        self.criterion    = criterion
        self.train_generator = train_generator
        self.valid_loader = valid_loader 
        self.root         = root
        self.description  = description
        self.lr           = lr
        self.num_batches_per_epoch = num_batches_per_epoch
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.es_mode      = es_mode
        self.eval_metrics = eval_metrics
        self.log          = log
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler    = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience)
        self.es           = EarlyStopping(mode=self.es_mode, patience=2*self.patience)
        self.scaler       = GradScaler()
        #self.warm_up      = warm_up
        self.history      = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
            
        if self.log:
            wandb.watch(self.model)
            
        self.training_time = 0
        
        
    def inference_step(self, x):
        return self.model(x.to(self.device))
    
    
    def save_hist(self):
        if(not os.path.exists(self.root+'pre-trained-tmp/trainer_logs')):
            os.makedirs(self.root+'pre-trained-tmp/trainer_logs')
        savepath = f'{self.root}pre-trained-tmp/trainer_logs/{self.description}.npy'
        np.save(savepath, self.history)
        
        return
    
    
    def save_model(self):
        if(not os.path.exists(self.root+'pre-trained-tmp/trained_UNets')):
            os.makedirs(self.root+'pre-trained-tmp/trained_UNets')
        if(not os.path.exists(self.root+'pre-trained-tmp/trainer_logs')):
            os.makedirs(self.root+'pre-trained-tmp/trainer_logs')
        savepath = f'{self.root}pre-trained-tmp/trained_UNets/{self.description}_best.pt'
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_hist()
        
        return
    
    
    def load_model(self):
        savepath = f'{self.root}pre-trained-tmp/trained_UNets/{self.description}_best.pt'
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f'{self.root}pre-trained-tmp/trainer_logs/{self.description}.npy'
        self.history = np.load(savepath,allow_pickle='TRUE').item()
        
        try:
            savepath = f'{self.root}results-tmp/unet/metrics_{self.description}.npy'
            self.evaluation = np.load(savepath, allow_pickle='TRUE').item()
        except:
            print("No metrics found")
        
        return
    
    
    def train_epoch(self):
        loss_list, batch_sizes = [], []
        
        for it in range(self.num_batches_per_epoch):
            
            batch = next(self.train_generator)
            input_ = batch['data']
            target = batch['target']#.long()
            
            if it == 0:
                assert input_.shape == target.shape, 'input and target shapes dont match'
                
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(input_)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(input_.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        
        if self.log:
            wandb.log({
                'train_loss': average_loss
            }, commit=False)
        
        
        return average_loss
    
    
    def get_sample(self, mode: str = 'valid'):
        if mode == 'valid':
            self.model.eval()
            data, target, _ = next(iter(self.valid_loader)).values()
            net_out = self.inference_step(data) 
            self.model.train()
        else:
            data, target, _ = next(iter(self.train_generator)).values()
            net_out = self.inference_step(data)
        x_hat = net_out
        
        return data.cpu(), target.cpu(), x_hat.cpu()
    
    
    @torch.no_grad()
    def eval_epoch(self):
        loss_list, batch_sizes, batch_sizes_chunks = [], [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for batch in self.valid_loader:
            input_  = batch['input']
            target  = batch['target']
            #print(input_.shape, target.shape )
            input_chunks  = torch.split(input_, 64, dim=0)
            target_chunks = torch.split(target, 64, dim=0)
            batch_sizes.append(input_.shape[0])
            net_out = []
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                #print(input_chunk.shape, target_chunk.shape )
                net_out_chunk = self.inference_step(input_chunk.to(self.device))
                net_out.append(net_out_chunk.detach().cpu())
                loss = self.criterion(net_out_chunk, target_chunk.to(self.device))
                loss_list.append(loss.item())
                batch_sizes_chunks.append(input_chunk.shape[0])
            
            net_out = torch.cat(net_out, dim=0)

            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())
        average_loss = epoch_average(loss_list, batch_sizes_chunks)
        self.history['valid loss'].append(average_loss)
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
            }, commit=False)
                
        return average_loss
    
    
    @torch.no_grad()
    def test_set(self, testloader: DataLoader) -> dict:
        self.model.eval()
        
        metric, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for batch in testloader:
            input_ = batch['input']
            target = batch['target']
            batch_sizes.append(input_.shape[0])
            
            input_chunks  = torch.split(input_, 32, dim=0)
            target_chunks = torch.split(target, 32, dim=0)
            net_out = []
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                net_out_chunk = self.inference_step(input_chunk.to(self.device))
                net_out.append(net_out_chunk.detach().cpu())
                
            net_out = torch.cat(net_out, dim=0)
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())
                    
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
                
        return epoch_metrics
    
    
    @torch.no_grad()
    def eval_all(self, cfg) -> dict:
        data_path = cfg['root'] + cfg['data_path']
        metrics = {}
        
        train_site = cfg['train_site']
        test_sites = [site for site in [1,2,3,4,5,6] if site != train_site]
        
        for split in ['train', 'validation']:
            dataset = CalgaryCampinasDataset(data_path=data_path, 
                                             site=train_site, 
                                             normalize=True, 
                                             volume_wise=True, 
                                             debug=cfg['debug'],
                                             split=split)
            
            dataloader = DataLoader(dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    drop_last=False, 
                                    collate_fn=volume_collate)
            
            metrics['Site ' + split + str(train_site)] = self.test_set(dataloader)
            
        
        for site in test_sites:
            dataset = CalgaryCampinasDataset(data_path=data_path, 
                                             site=site, 
                                             normalize=True, 
                                             volume_wise=True, 
                                             debug=cfg['debug'],
                                             split='all')
            
            dataloader = DataLoader(dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    drop_last=False, 
                                    collate_fn=volume_collate)
            
            metrics['Site ' + str(site)] = self.test_set(dataloader)
            
        return metrics
    

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
        
        if 'Volumetric Dice' in self.history:
            loss_array = np.array(self.history['Volumetric Dice'])
            ax[1].plot(self.history['Volumetric Dice'], label='Volumetric Dice', c='teal', lw=3)
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Volumetric Dice")
            ax[1].legend(loc="lower right")
            ax[1].set_title(f'Volumetric Dice of Validation Set\nBest Value: {loss_array.max()} @ Epoch {loss_array.argmax()+1}')
        
        
        if 'Surface Dice' in self.history:
            loss_array = np.array(self.history['Surface Dice'])
            ax[1].plot(self.history['Surface Dice'], label='Surface Dice', c='tab:purple', lw=3)
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Surface Dice")
            ax[1].legend(loc="lower right")
            ax[1].set_title(f'Surface Dice of Validation Set\nBest Value: {loss_array.max()} @ Epoch {loss_array.argmax()+1}')
        #plt.show()
        
        #plt.savefig(f"{self.description}_logs.png")
        
        return fig
    
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
#         if self.warm_up:
#             augment = self.train_loader.dataset.augment
        
        if self.log:
            wandb.log({}, commit=True)
        
        for epoch in progress_bar:
#             if self.warm_up:
#                 if epoch < 5:
#                     self.train_loader.dataset.augment = False
#                 else:
#                     self.train_loader.dataset.augment = augment
                
            self.model.train()
            train_loss = self.train_epoch()
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]
            
            if self.log:
                wandb.log({}, commit=True)            

            if self.es_mode == 'min':
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            if(self.es.step(es_metric)):
                print('Early stopping triggered!')
                break
                
        self.training_time = time.time() - self.training_time
        self.save_hist()
        self.load_model()

        

class UNetTrainerACDC():
    def __init__(
        self, 
        model: nn.Module, 
        criterion: Callable, 
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_batches_per_epoch,
        num_val_batches_per_epoch,
        root: str, 
        description: str = 'untitled', 
        lr: float = 1e-4, 
        n_epochs: int = 250, 
        patience: int = 5, 
        #warm_up=True,
        es_mode: str = 'min', 
        eval_metrics: Dict[str, nn.Module] = None,
        log: bool = True,
        device = 0
    ):
        self.device       = device
        self.model        = model.to(self.device)
        self.criterion    = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_val_batches_per_epoch = num_val_batches_per_epoch
        self.root         = root
        self.description  = description
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.es_mode      = es_mode
        self.eval_metrics = eval_metrics
        self.log          = log
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler    = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience)
        self.es           = EarlyStopping(mode=self.es_mode, patience=2*self.patience)
        self.scaler       = GradScaler()
        #self.warm_up      = warm_up
        self.history      = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
            
        if self.log:
            wandb.watch(self.model)
            
        self.training_time = 0
        
        
    def inference_step(self, x):
        return self.model(x.to(self.device))
    
    
    def save_hist(self):
        if(not os.path.exists(self.root+'pre-trained-tmp/trainer_logs')):
            os.makedirs(self.root+'pre-trained-tmp/trainer_logs')
        savepath = f'{self.root}pre-trained-tmp/trainer_logs/{self.description}.npy'
        np.save(savepath, self.history)
        
        return
    
    
    def save_model(self):
        if(not os.path.exists(self.root+'pre-trained-tmp/trained_UNets')):
            os.makedirs(self.root+'pre-trained-tmp/trained_UNets')
        if(not os.path.exists(self.root+'pre-trained-tmp/trainer_logs')):
            os.makedirs(self.root+'pre-trained-tmp/trainer_logs')
        savepath = f'{self.root}pre-trained-tmp/trained_UNets/{self.description}_best.pt'
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_hist()
        
        return
    
    
    def load_model(self):
        savepath = f'{self.root}pre-trained-tmp/trained_UNets/{self.description}_best.pt'
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f'{self.root}pre-trained-tmp/trainer_logs/{self.description}.npy'
        self.history = np.load(savepath,allow_pickle='TRUE').item()
        
        try:
            savepath = f'{self.root}/results-tmp/unet/metrics_{self.description}.npy'
            self.evaluation = np.load(savepath, allow_pickle='TRUE').item()
        except:
            print("No metrics found")
        
        return
    
    
    def train_epoch(self):
        loss_list, batch_sizes = [], []
        for it in range(self.num_batches_per_epoch):
            batch = next(self.train_loader)
            input_ = batch['data']
            target = batch['target'][0].long().cuda()
            target = F.one_hot(target).squeeze(1).permute(0,3,1,2)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(input_)
                loss = self.criterion(net_out, target.to(self.device))
            #loss.backward()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)
            #self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(input_.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        
        if self.log:
            wandb.log({
                'train_loss': average_loss
            }, commit=False)
        
        
        return average_loss
    
    
    def get_sample(self, mode: str = 'valid'):
        if mode == 'valid':
            self.model.eval()
            data, target, _ = next(iter(self.valid_loader)).values()
            net_out = self.inference_step(data) 
            self.model.train()
        else:
            data, target, _ = next(iter(self.train_loader)).values()
            net_out = self.inference_step(data)
        x_hat = net_out
        
        return data.cpu(), target.cpu(), x_hat.cpu()
    
    
    @torch.no_grad()
    def eval_epoch(self):
        loss_list, batch_sizes, batch_sizes_chunks = [], [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for it in range(self.num_val_batches_per_epoch):
            batch = next(self.valid_loader)
            input_ = batch['data']
            target = batch['target'][0].long().cuda()
            target = F.one_hot(target).squeeze(1).permute(0,3,1,2)
            #print(input_.shape, target.shape )
            input_chunks  = torch.split(input_, 64, dim=0)
            target_chunks = torch.split(target, 64, dim=0)
            batch_sizes.append(input_.shape[0])
            net_out = []
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                #print(input_chunk.shape, target_chunk.shape )
                net_out_chunk = self.inference_step(input_chunk.to(self.device))
                net_out.append(net_out_chunk.detach().cpu())
                loss = self.criterion(net_out_chunk, target_chunk.to(self.device))
                loss_list.append(loss.item())
                batch_sizes_chunks.append(input_chunk.shape[0])
            
            net_out = torch.cat(net_out, dim=0)
            target = target.cpu()
            
            #print(net_out.device, target.device)
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())
        average_loss = epoch_average(loss_list, batch_sizes_chunks)
        self.history['valid loss'].append(average_loss)
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
            }, commit=False)
                
        return average_loss
    
    
    @torch.no_grad()
    def test_set(self, testloader: DataLoader) -> dict:
        self.model.eval()
        
        metric, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for batch in testloader:
            input_ = batch['input']
            target = batch['target']
            batch_sizes.append(input_.shape[0])
            
            input_chunks  = torch.split(input_, 32, dim=0)
            target_chunks = torch.split(target, 32, dim=0)
            net_out = []
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                net_out_chunk = self.inference_step(input_chunk.to(self.device))
                net_out.append(net_out_chunk.detach().cpu())
                
            net_out = torch.cat(net_out, dim=0)
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())
                    
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
                
        return epoch_metrics
    
    
    @torch.no_grad()
    def eval_all(self, cfg) -> dict:
        raise "NotImplementedError"
    
    
    @torch.no_grad()
    def get_subset(self, dataloader: DataLoader, n_cases=10, part="tail") -> DataLoader:
        assert dataloader.batch_size == 1
        self.model.eval()
        loss_list = []
        for batch in dataloader:
            input_  = batch['input'].to(self.device)
            target  = batch['target'].to(self.device)
            net_out = self.inference_step(input_)
            loss    = self.criterion(net_out, target)
            loss_list.append(loss.item())
            
        loss_tensor = torch.tensor(loss_list)
        indices = torch.argsort(loss_tensor, descending=True)
        len_ = len(loss_list)
        if part == 'tail':
            indices = indices[:len_ // 10]
        elif part == 'head':
            indices = indices[-len_ // 10:]   
            
        indices_selection = slice_selection(dataloader.dataset, indices, n_cases=n_cases)
        subset            = dataset_from_indices(dataloader.dataset, indices_selection)
        
        return subset
    

    
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
        
        if 'Volumetric Dice' in self.history:
            loss_array = np.array(self.history['Volumetric Dice'])
            ax[1].plot(self.history['Volumetric Dice'], label='Volumetric Dice', c='teal', lw=3)
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Volumetric Dice")
            ax[1].legend(loc="lower right")
            ax[1].set_title(f'Volumetric Dice of Validation Set\nBest Value: {loss_array.max()} @ Epoch {loss_array.argmax()+1}')
        
        
        if 'Surface Dice' in self.history:
            loss_array = np.array(self.history['Surface Dice'])
            ax[1].plot(self.history['Surface Dice'], label='Surface Dice', c='tab:purple', lw=3)
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Surface Dice")
            ax[1].legend(loc="lower right")
            ax[1].set_title(f'Surface Dice of Validation Set\nBest Value: {loss_array.max()} @ Epoch {loss_array.argmax()+1}')
        #plt.show()
        
        #plt.savefig(f"{self.description}_logs.png")
        
        return fig
    
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
#         if self.warm_up:
#             augment = self.train_loader.dataset.augment
        
        if self.log:
            wandb.log({}, commit=True)
        
        for epoch in progress_bar:
#             if self.warm_up:
#                 if epoch < 5:
#                     self.train_loader.dataset.augment = False
#                 else:
#                     self.train_loader.dataset.augment = augment
                
            self.model.train()
            train_loss = self.train_epoch()
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]
            
            if self.log:
                wandb.log({}, commit=True)            

            if self.es_mode == 'min':
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            if(self.es.step(es_metric)):
                print('Early stopping triggered!')
                break
                
        self.training_time = time.time() - self.training_time
        self.save_hist()
        self.load_model()