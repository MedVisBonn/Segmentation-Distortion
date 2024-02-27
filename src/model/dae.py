from typing import List
from omegaconf import OmegaConf
from math import log2
import torch
from torch import Tensor, nn
from model.ae import AE, ChannelAE
from monai.networks.blocks import ResidualUnit
from monai.networks.nets import UNet
from model.wrapper import ModelAdapter



def get_daes(
    unet: nn.Module,
    cfg: OmegaConf,
    return_state_dict: bool = False
)-> nn.ModuleDict:
    
    if cfg.dae.placement == 'bottleneck':
        cfg.dae.identity_swivels = [
            i for i in range(0, len(cfg.dae.swivels.keys())-1)
        ]

    elif cfg.dae.placement == 'all':
        cfg.dae.identity_swivels = []

    # parse disabled_ids from identity_swivels and attachment names
    cfg.dae.trainer.disabled_ids = [
        list(cfg.dae.swivels.keys())[i] for i in cfg.dae.identity_swivels
    ]

    model = cfg.dae.model
    arch = cfg.dae.arch
    swivels = cfg.dae.swivels
    disabled_ids = cfg.dae.trainer.disabled_ids

    if model == 'unet':
        daes = get_unetDAE(arch, swivels, disabled_ids)
    elif model == 'channelDAE':
        data_key = cfg.run.data_key
        if 'monai' in cfg.unet[data_key].pre:
            arch.spatial = arch.spatial.monai
        elif 'default' in cfg.unet[data_key].pre:
            arch.spatial = arch.spatial.default
        daes = get_channelDAE(arch, swivels, disabled_ids)
    elif model == 'ResDAE':
        daes = get_resDAE(arch, swivels, disabled_ids)
    else:
        raise NotImplementedError(f'Model {model} not implemented.')
    
    for layer, idx in zip(disabled_ids, cfg.dae.identity_swivels):
        print(f'Disabling layer {layer}, index {idx} in identity_swivels.')
        daes.append(IdentityHook(layer))

    model = ModelAdapter(
        seg_model=unet,
        transformations=daes,
        disabled_ids=disabled_ids,
        copy=True
    )

    if return_state_dict:
        root = cfg.fs.root
        data_key = cfg.run.data_key
        iteration = cfg.run.iteration
        model_name = f'{data_key}_{cfg.dae.name}_{cfg.dae.postfix}_' + \
            f'{cfg.unet[cfg.run.data_key].pre}_{iteration}_best.pt'
        model_name = model_name.replace('__', '')
        # model_name = f'{data_key}_{cfg.dae.name}_{iteration}_best.pt'
        model_path = f'{root}pre-trained-tmp/trained_AEs/{model_name}'
        state_dict = torch.load(model_path)['model_state_dict']
        return model, state_dict
    else:
        return model



def get_unetDAE(
    arch: OmegaConf,
    disabled_ids: List[str]
) -> nn.ModuleDict:
    daes = nn.ModuleDict({
        layer: UnetDAE(
            in_channels = arch[layer].channel, 
            num_res_units = arch.num_res_units,
            residual = arch.residual  
        ) for layer in arch if layer not in disabled_ids
    })

    return daes



def get_channelDAE(
    arch: OmegaConf,
    swivels: OmegaConf,
    disabled_ids: List[str]
) -> nn.ModuleDict:
    for i, layer in enumerate(swivels.keys()):
        if layer not in disabled_ids:
            print(f"{arch.spatial[i]}, {swivels[layer].channel}")

    daes = nn.ModuleList({
        ChannelDAE(
            in_channels = swivels[layer].channel, 
            in_dim      = arch.spatial[i],
            depth       = arch.depth,
            block_size  = arch.block,
            swivel      = layer,
            residual    = arch.residual,
        ) for i, layer in enumerate(swivels.keys()) if layer not in disabled_ids
    })

    return daes



def get_resDAE(
    arch: OmegaConf,
    swivels: OmegaConf,
    disabled_ids: List[str]
) -> nn.ModuleList:
    daes = nn.ModuleList({
        ResDAE(
            in_channels = swivels[layer].channel, 
            depth       = arch.depth,
            residual    = arch.residual,
            swivel      = layer
        ) for layer in swivels if layer not in disabled_ids
    })

    return daes



class IdentityHook(nn.Module):
    def __init__(self, swivel: str):
        super().__init__()
        self.swivel = swivel

    def forward(self, x: Tensor) -> Tensor:
        return x


class ChannelDAE(nn.Module):
    """ Residual Denoising Autoencoder (resDAE) for denoising feature maps.
    """
    def __init__(
        self, 
        in_channels: int,
        in_dim: int,
        swivel: str,
        depth: int = 3, 
        block_size: int = 1,
        residual: str = True,
    ):
        super().__init__()
        self.on = True
        self.swivel = swivel
        self.residual = residual
        self.ae = ChannelAE(
            in_channels, 
            in_dim, 
            depth=depth, 
            block_size=block_size,
            residual=True
        )


    def turn_off(self) -> None:
        self.on = False
    

    def turn_on(self) -> None:
        self.on = True


    def forward(self, x: Tensor) -> Tensor:
        if self.on:
            ae_out = self.ae(x)
            if self.residual: 
                return x + ae_out
            else:
                return ae_out
        else:
            return x



class UnetDAE(nn.Module):
    """ Residual Denoising Autoencoder (resDAE) for denoising feature maps.
    """
    def __init__(
        self, 
        in_channels: int, 
        num_res_units: int = 1,
        residual: str = True
    ):
        super().__init__()
        self.on = True
        self.residual = residual
        
        # in_channels * 2**i until we reach 256
        channels = [2**(i+1) for i in range(int(log2(in_channels)), 8)]
        # stride 2 in each step
        strides = [2 for _ in range(len(channels)-1)]
        # monai UNet
        self.ae = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units, 
            up_kernel_size=3
        )


    def turn_off(self) -> None:
        self.on = False


    def turn_on(self) -> None:
        self.on = True


    def forward(self, x: Tensor) -> Tensor:
        if self.on:
            ae_out = self.ae(x)
            if self.residual: 
                return x + ae_out
            else:
                return ae_out
        else:
            return x


class ResDAE(nn.Module):
    """
    Residual Denoising Autoencoder (ResDAE) model.

    Args:
        in_channels (int): Number of input channels.
        depth (int): Number of residual units in the model.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
    """

    def __init__(
        self, 
        in_channels, 
        depth, 
        swivel,
        residual=True
    ):
        super(ResDAE, self).__init__()
        self.on = True
        self.swivel = swivel
        self.residual = residual

        self.model = nn.ModuleList(
            ResidualUnit(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=in_channels,
                act="PReLU",
                norm=("instance", {"affine": True}), # "BATCH",
                adn_ordering="AN"
            ) for _ in range(depth)
        )

    def turn_off(self) -> None:
        """
        Turn off the ResDAE model.
        """
        self.on = False

    def turn_on(self) -> None:
        """
        Turn on the ResDAE model.
        """
        self.on = True

    def forward(self, x):
        """
        Forward pass of the ResDAE model.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResDAE model.
        """
        if self.on:
            for layer in self.model:
                x_out = layer(x)
            if self.residual:
                return x + x_out
            else:
                return x_out
        else:
            return x


# class resDAE(nn.Module):
#     """ Residual Denoising Autoencoder (resDAE) for denoising feature maps.
#     """
#     def __init__(
#         self, 
#         in_channels, 
#         in_dim, 
#         latent_dim=128, 
#         depth=3, 
#         latent='dense', 
#         block_size=1,
#         w_prior=2
#     ):
#         super().__init__()
#         self.w_prior = w_prior
#         self.on      = True
#         self.inspect = False
        
#         self.ae    = AE(in_channels, in_dim, latent_dim=latent_dim, depth=depth, latent=latent, block_size=block_size)
#         self.prior = AE(in_channels, in_dim, latent_dim=latent_dim, depth=depth, latent=latent, block_size=block_size)


#     def turn_off(self) -> None:
#         self.on = False
    

#     def turn_on(self) -> None:
#         self.on = True
    

#     def forward(self, x: Tensor) -> Tensor:
#         if self.on:
#             # TODO: is this save?
#             if self.training:
#                 with torch.no_grad():
#                     x += self.prior(x.detach()) * self.w_prior
#             res = self.ae(x)
                    
#             if self.inspect:
#                 prior = self.prior(x.detach())                
#                 return x + res, res, prior
#             else:
#                 return x + res
#         else:
#             return x