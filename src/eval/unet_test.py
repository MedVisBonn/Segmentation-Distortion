
from typing import Dict, Union
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from utils import  epoch_average
from data_utils import volume_collate
from losses import (
    DiceScoreCalgary, 
    SurfaceDiceCalgary,
    DiceScoreMMS
)



def get_df_from_dict(
    cfg: OmegaConf,
    metrics: Dict
):
    # convert dict into seaborn-friendly pandas format
    df = pd.DataFrame.from_dict(metrics).T
    df['Domain'] = df.index
    df.reset_index(drop=True, inplace=True)
    df = pd.melt(
        df, 
        id_vars=['Domain'],
        value_vars=df.columns.drop('Domain')
    )
    # add additional identifiers from config
    df['Iteration'] = cfg.run.iteration
    df['Model'] = cfg.unet[cfg.run.data_key].pre
    df['Data'] = cfg.run.data_key

    return df


def eval_set(
    cfg: OmegaConf,
    model: nn.Module,
    data: Union[Dataset, MultiThreadedAugmenter],
    loader_format: str = 'set',
) -> Dict:
    if cfg.run.data_key == 'brain':

        eval_metrics = {
            'Volumetric Dice': DiceScoreCalgary(),
        }

        if loader_format == 'set':
            eval_metrics['Surface Dice'] = SurfaceDiceCalgary()
            data.volume_wise = True
            dataloader = DataLoader(
                data,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=volume_collate
            )        
            metrics = eval_brain_set(
                model=model, 
                dataloader=dataloader, 
                eval_metrics=eval_metrics
            )

        elif loader_format == 'generator':
            metrics = eval_brain_generator(
                model=model, 
                generator=data, 
                n_batches=250,
                eval_metrics=eval_metrics
            )
        else:
            raise ValueError(
                f'Invalid loader format. No loader format named {loader_format}.'
                f'Choose from "set" or "generator".'
            )

    elif cfg.run.data_key == 'heart':

        eval_metrics = {
            "Volumetric Dice": DiceScoreMMS()
        }

        if loader_format == 'set':
            dataloader = DataLoader(
                data, 
                batch_size=32, 
                shuffle=False, 
                drop_last=False
            )
            metrics = eval_heart_set(
                model=model, 
                dataloader=dataloader, 
                eval_metrics=eval_metrics
            )

        elif loader_format == 'generator':
            metrics = eval_heart_generator(
                model=model, 
                generator=data, 
                n_batches=250,
                eval_metrics=eval_metrics
            )
        else:
            raise ValueError(
                f'Invalid loader format. No loader format named {loader_format}.'
                f'Choose from "set" or "generator".'
            )
    
    else:
        raise ValueError(
            f'Invalid data key. No dataset named {cfg.run.data_key}'
        )

    return metrics



@torch.no_grad()
def eval_brain_set(
    model: nn.Module, 
    dataloader: DataLoader, 
    eval_metrics: Dict
) -> Dict:
    model.eval()
    batch_sizes = []
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    for batch in dataloader:
        input_ = batch['input']
        target = batch['target']
        batch_sizes.append(input_.shape[0])

        input_chunks  = torch.split(input_, 32, dim=0)
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


@torch.no_grad()
def eval_brain_generator(
    model: nn.Module, 
    generator: MultiThreadedAugmenter, 
    n_batches: int,
    eval_metrics: Dict
) -> Dict:
    model.eval()
    batch_sizes = []
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    for it in range(n_batches):
        batch = next(generator)
        input_ = batch['input']
        target = batch['target']
        # batch_sizes.append(input_.shape[0])

        # input_chunks  = torch.split(input_, 32, dim=0)
        # net_out = []
        # for input_chunk in input_chunks:
        #     net_out_chunk = model(input_chunk.cuda())
        #     net_out.append(net_out_chunk.detach().cpu())

        # net_out = torch.cat(net_out, dim=0)

        batch_sizes.append(input_.shape[0])
        net_out = model(input_.cuda())
        for key, metric in eval_metrics.items():
            epoch_metrics[key].append(metric(net_out,target).detach().mean().cpu())

    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)

    return epoch_metrics



@torch.no_grad()
def eval_heart_set(
    model: nn.Module, 
    dataloader: DataLoader, 
    eval_metrics: Dict
) -> Dict:
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
        target = F.one_hot(
            target.long(), 
            num_classes=4
        ).squeeze(1).permute(0,3,1,2)
        # get model output
        net_out = model(input_.cuda())
        
        batch_sizes.append(input_.shape[0])
        for key, metric in eval_metrics.items():
            epoch_metrics[key].append(
                metric(net_out, target).detach().mean().cpu()
            )
            
    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
        
    return epoch_metrics



@torch.no_grad()
def eval_heart_generator(
    model: nn.Module, 
    generator: MultiThreadedAugmenter,
    n_batches: int,
    eval_metrics: Dict
) -> Dict:
    model.eval()
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    # saves batch sizes for each batch for averaging
    batch_sizes = []

    for it in range(n_batches):
        batch = next(generator)
        input_ = batch['data']
        target = batch['target'].cuda()
        # convert -1 labels to background
        target[target == -1] = 0
        # convert to one-hot encoding
        target = F.one_hot(
            target.long(), 
            num_classes=4
        ).squeeze(1).permute(0,3,1,2)
        # get model output
        net_out = model(input_.cuda())
        
        batch_sizes.append(input_.shape[0])
        for key, metric in eval_metrics.items():
            epoch_metrics[key].append(
                metric(net_out, target).detach().mean().cpu()
            )
            
    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
        
    return epoch_metrics