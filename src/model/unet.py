"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""

from typing import (
    Dict, 
    Tuple, 
    List, 
    Union
)
from omegaconf import OmegaConf
import torch
from torch import nn
from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d
from monai.networks.nets import UNet, SwinUNETR



def get_unet(
    cfg: OmegaConf,
    return_state_dict=False
) -> Union[nn.Module, Tuple[nn.Module, Dict]]:
    
    unet_cfg = cfg.unet[cfg.run.data_key]

    if unet_cfg.arch == 'default':
        unet = get_default_unet_arch(cfg)
    elif unet_cfg.arch == 'monai':
        unet = get_monai_unet_arch(cfg)
    elif unet_cfg.arch == 'swinunetr':
        unet = get_monai_swinunetr_arch(cfg)

    if return_state_dict:
        root = cfg.fs.root
        unet_name = f'{cfg.run.data_key}_{unet_cfg.pre}_{cfg.run.iteration}'
        model_path = f'{root}{unet_cfg.training.save_loc}/trained_UNets/{unet_name}_best.pt'
        state_dict = torch.load(model_path)['model_state_dict']
        return unet, state_dict
    else:
        return unet
    

def get_default_unet_arch(
    cfg: OmegaConf,
    return_swivels: bool = False
):
    unet_cfg       = cfg.unet[cfg.run.data_key]
    n_chans_in     = unet_cfg.n_chans_in
    n_chans_out    = unet_cfg.n_chans_out
    n_filters_init = unet_cfg.n_filters_init

    unet = UNet2D(
        n_chans_in=n_chans_in, 
        n_chans_out=n_chans_out, 
        n_filters_init=n_filters_init
    )

    # swivels are the attachment points for our denoiser. The
    # change with the unet architecture (they specify the 
    # layers by name)
    if return_swivels:
        swivels = ['shortcut0', 'shortcut1', 'shortcut2', 'up3']
        return unet, swivels
    else:
        return unet



def get_monai_unet_arch(
    cfg: OmegaConf,
    return_swivels: bool = False
) -> nn.Module:
    
    unet_cfg       = cfg.unet[cfg.run.data_key]
    in_channels    = unet_cfg.n_chans_in
    out_channels   = unet_cfg.n_chans_out
    n_filters_init = unet_cfg.n_filters_init
    depth          = unet_cfg.depth
    num_res_units  = unet_cfg.num_res_units
    channels       = [n_filters_init * 2 ** i for i in range(depth)]
    strides        = [2] * (depth - 1)

    unet = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units
    )
    if return_swivels:
        swivels = [
            f'model.1.{"submodule.1." * i}swivel' for i in range(unet_cfg.depth)
        ]
        return unet, swivels


def get_monai_swinunetr_arch(
    cfg: OmegaConf
) -> nn.Module:
    unet_cfg       = cfg.unet[cfg.run.data_key]
    in_channels    = unet_cfg.n_chans_in
    out_channels   = unet_cfg.n_chans_out

    return SwinUNETR(
        img_size=(256, 256),
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2
    )



class UNet2D(nn.Module):
    def __init__(
        self,
        n_chans_in,
        n_chans_out,
        kernel_size=3,
        padding=1,
        pooling_size=2,
        n_filters_init=8,
        dropout=False,
        p=0.1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_size = pooling_size
        n = n_filters_init
        if dropout:
            dropout_layer = nn.Dropout(p)
        else:
            dropout_layer = nn.Identity()

        self.init_path = nn.Sequential(
            nn.Conv2d(
                n_chans_in, n, self.kernel_size, padding=self.padding, bias=False
            ),
            nn.ReLU(),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n),
        )
        self.attachment = nn.Identity()
        self.shortcut0 = nn.Conv2d(n, n, 1)

        self.down1 = nn.Sequential(
            # nn.BatchNorm2d(n),
            nn.Conv2d(
                n, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False
            ),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 2),
        )
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, 1)

        self.down2 = nn.Sequential(
            # nn.BatchNorm2d(n * 2),
            nn.Conv2d(
                n * 2, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False
            ),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 4),
        )
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, 1)

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(n * 4),
            nn.Conv2d(
                n * 4, n * 8, kernel_size=pooling_size, stride=pooling_size, bias=False
            ),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 8)
            # dropout_layer
        )

        self.up3 = nn.Sequential(
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 8),
            nn.ConvTranspose2d(
                n * 8,
                n * 4,
                kernel_size=self.pooling_size,
                stride=self.pooling_size,
                bias=False,
            ),
            nn.ReLU(),
            dropout_layer,
        )

        self.up2 = nn.Sequential(
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 4),
            nn.ConvTranspose2d(
                n * 4,
                n * 2,
                kernel_size=self.pooling_size,
                stride=self.pooling_size,
                bias=False,
            ),
            nn.ReLU(),
            dropout_layer,
        )

        self.up1 = nn.Sequential(
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            ResBlock2d(
                n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding
            ),
            nn.BatchNorm2d(n * 2),
            nn.ConvTranspose2d(
                n * 2,
                n,
                kernel_size=self.pooling_size,
                stride=self.pooling_size,
                bias=False,
            ),
            nn.ReLU(),
            dropout_layer,
        )

        self.out_path = nn.Sequential(
            ResBlock2d(n, n, kernel_size=1),
            PreActivation2d(n, n_chans_out, kernel_size=1),
            nn.BatchNorm2d(n_chans_out),
        )

    def forward(self, x):
        x0 = self.init_path(x)
        x0 = self.attachment(x0)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x2_up = self.up3(x3)
        x1_up = self.up2(x2_up + self.shortcut2(x2))
        x0_up = self.up1(x1_up + self.shortcut1(x1))
        x_out = self.out_path(x0_up + self.shortcut0(x0))

        return x_out


class UNetEnsemble(nn.Module):
    def __init__(
        self,
        unets: List,
        reduce: str = 'none'
    ):
        super().__init__()
        self.ensemble = nn.ModuleList(unets)
        self.reduce = reduce
        
    def forward(self, x):
        x_out = torch.cat([module(x.clone()) for module in self.ensemble])
        if self.reduce=='none':
            return x_out
        elif self.reduce=='mean':
            return x_out.mean(0, keepdims=True)
        
        
# class UNetEnsemble(nn.Module):
#     def __init__(
#         self,
#         unets: List,
#         reduce: str
#     ):
#         super().__init__()
#         self.reduce=reduce
#         self.ensemble = nn.ModuleList(unets)

#     def forward(self, x):
#         x_out = torch.stack([module(x.clone()) for module in self.ensemble], dim=0)
#         if self.reduce=='none':
#             return x_out
#         elif self.reduce=='mean':
#             print(x_out.shape)
#             return x_out.mean(0)