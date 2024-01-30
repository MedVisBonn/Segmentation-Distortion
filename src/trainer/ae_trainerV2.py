import os, sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
from typing import Dict, Callable
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm


sys.path.append('..')
from utils import (
    EarlyStopping, 
    epoch_average, 
    average_metrics
)
from model.wrapper import FrankensteinV2
from losses import (
    CalgaryCriterionAE, 
    MNMCriterionAE,
    SampleDice, 
    UnetDice
)



class AETrainerCalgaryV2:
    def __init__(
        self, 
        model: nn.Module, 
        # unet: nn.Module, 
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
        debug: bool = False
    ):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = model.to(self.device)
        # self.unet         = unet.to(self.device)
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
        
        if target == 'output':
            assert self.criterion.id_loss == 'huber'
        elif target == 'gt':
            assert self.criterion.id_loss in ['ce', 'bce']
        
        #if self.log:
        #    run = wandb.init(reinit=True, name='log_' + self.description, project='Thesis-VAE')
        
        
#     def inference_step(self, x):
#         with torch.no_grad():
#             unet_out = self.unet(x.to(self.device)).detach()
#         samples  = self.model(x.to(self.device))
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
#             batch = next(self.train_loader)
#             input_ = batch['data']
            batch = self.train_loader.next()
            input_ = torch.cat([batch['data_orig'], batch['data']], dim=0)
            
            with autocast():
                unet_out, samples = self.inference_step(input_)
                
                if self.target == 'output':
                    loss, metrics = self.criterion(unet_out, samples, 
                                                  self.model.training_data)
                elif self.target == 'gt':
                    target = batch['target']
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
        
        self.history['train loss'].append(average_loss)
        self.history['train metrics'].append(metrics)
        
        if self.log:
            wandb.log({
                'train_loss': average_loss,
                'train_metrics': metrics
            }, commit=False)

        return average_loss
    
    
    @torch.no_grad()
    def eval_epoch(self):
        loss_list, metric_list, batch_sizes = [], [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
            
        for it in range(self.num_val_batches_per_epoch):
            batch = self.valid_loader.next()
            input_ = torch.cat([batch['data_orig'], batch['data']], dim=0)
            
            target = batch['target'].to(self.device)
            unet_out, samples = self.inference_step(input_)

            if self.target == 'output':
                loss, metrics = self.criterion(unet_out, samples, 
                                               self.model.training_data)
            elif self.target == 'gt':
                loss, metrics = self.criterion(target.to(self.device), samples, 
                                   self.model.training_data)

            loss_list.append(loss.item())
            metric_list.append(metrics)
            batch_sizes.append(input_.shape[0])
            
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    #print(key, unet_out.shape, samples.shape, target.shape)
                    epoch_metrics[key].append(metric(unet_out, samples, target).mean().detach().cpu())
                    
        average_loss = epoch_average(loss_list, batch_sizes)
        metrics      = average_metrics(metric_list, batch_sizes)
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
        # self.unet.eval()
        self.model.freeze_seg_model()

        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        
        if self.log:
            wandb.log({}, commit=True)
        
        for epoch in progress_bar:      
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
        
        
    
class AETrainerACDCV2:
    def __init__(
        self, 
        model: nn.Module, 
        # unet: nn.Module, 
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
        # self.unet         = unet.to(self.device)
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
                    self.model.training_data
                )
                
            elif self.target == 'gt':
                loss, metrics = self.criterion(
                    target.to(self.device), 
                    samples, 
                    self.model.training_data
                )

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
        # self.unet.eval()
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



def get_dae_trainer(
    cfg,
    model,
    train_loader,
    val_loader
):
    if cfg.run.data_key == 'brain':
        return get_dae_brain_trainer(
            cfg=cfg, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader
        )
    elif cfg.run.data_key == 'heart':
        return get_dae_heart_trainer(
            cfg=cfg, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader
        )
    else:
        raise ValueError(f"Unknown data_key {cfg.run.data_key}")



def get_dae_brain_trainer(
    cfg, 
    model, 
    train_loader, 
    val_loader 
):
    
    trainer_config = cfg.dae.trainer
    criterion = CalgaryCriterionAE(
        loss=trainer_config.loss, 
        recon=trainer_config.reconstruction, 
        diff=trainer_config.difference
    )
    eval_metrics = {
        'Sample Volumetric Dice': SampleDice(data='calgary'),
        'UNet Volumetric Dice': UnetDice(data='calgary')
    }

    trainer = AETrainerCalgaryV2(
        model=model, 
        criterion=criterion,
        train_loader=train_loader, 
        valid_loader=val_loader,
        num_batches_per_epoch=trainer_config.num_batches_per_epoch,
        num_val_batches_per_epoch=trainer_config.num_val_batches_per_epoch,
        root=cfg.fs.root,
        target=trainer_config.target,
        description=f'{cfg.run.data_key}_{cfg.run.name}_{cfg.run.iteration}',
        lr=trainer_config.lr, 
        eval_metrics=eval_metrics, 
        log=cfg.wandb.log,
        n_epochs=250, 
        patience=8
    )

    return trainer


def get_dae_heart_trainer(
    cfg, 
    model, 
    train_loader, 
    val_loader
):
    trainer_config = cfg.dae.trainer
    criterion = MNMCriterionAE(
        loss=trainer_config.loss, 
        recon=trainer_config.reconstruction, 
        diff=trainer_config.difference
    )
    eval_metrics = {
        'Sample Volumetric Dice': SampleDice(data='MNM'),
        'UNet Volumetric Dice': UnetDice(data='MNM')
    }

    trainer = AETrainerACDCV2(
        model=model, 
        criterion=criterion, 
        train_loader=train_loader, 
        valid_loader=val_loader,
        num_batches_per_epoch=trainer_config.num_batches_per_epoch,
        num_val_batches_per_epoch=trainer_config.num_val_batches_per_epoch,
        root=cfg.fs.root,
        target=trainer_config.target,
        description=f'{cfg.run.data_key}_{cfg.run.name}_{cfg.run.iteration}',
        lr=trainer_config.lr, 
        eval_metrics=eval_metrics, 
        log=cfg.wandb.log,
        n_epochs=250, 
        patience=8,
        device=torch.device('cuda')
    )

    return trainer