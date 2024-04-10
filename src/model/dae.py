from typing import List
from omegaconf import OmegaConf
from math import log2
import torch
from torch import Tensor, nn
from model.ae import AE, ChannelAE
from monai.networks.blocks import ResidualUnit
from monai.networks.nets import UNet
from model.wrapper import ModelAdapter, MaskedAutoencoderAdapter



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
    elif model == 'compressionDAE':
        data_key = cfg.run.data_key
        if 'monai' in cfg.unet[data_key].pre:
            arch.spatial = arch.spatial.monai
        elif 'default' in cfg.unet[data_key].pre:
            arch.spatial = arch.spatial.default
        daes = get_compressionDAE(arch, swivels, disabled_ids)
    elif model == 'ResDAE':
        daes = get_resDAE(arch, swivels, disabled_ids)
    elif model == 'ResMAE':
        daes = get_resMAE(arch, swivels, disabled_ids)
    else:
        raise NotImplementedError(f'Model {model} not implemented.')
    
    return_mask = True if model == 'ResMAE' else False
    for layer, idx in zip(disabled_ids, cfg.dae.identity_swivels):
        print(f'Disabling layer {layer}, index {idx} in identity_swivels.')
        daes.append(IdentityHook(swivel=layer, return_mask=return_mask))

    if model == 'ResMAE':
        model = MaskedAutoencoderAdapter(
            seg_model=unet,
            transformations=daes,
            disabled_ids=disabled_ids,
            copy=True
        )
    else:
        model = ModelAdapter(
            seg_model=unet,
            transformations=daes,
            disabled_ids=disabled_ids,
            copy=True
        )

    if return_state_dict:
        weight_dir = cfg.dae.weight_dir
        data_key = cfg.run.data_key
        iteration = cfg.run.iteration
        model_name = f'{data_key}_{cfg.dae.name}{cfg.dae.postfix}_' + \
            f'{cfg.unet[cfg.run.data_key].pre}_{iteration}_best.pt'
        model_name = model_name.replace('__', '_')
        # model_name = f'{data_key}_{cfg.dae.name}_{iteration}_best.pt'
        model_path = f'{weight_dir}{model_name}'
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


def get_compressionDAE(
    arch: OmegaConf,
    swivels: OmegaConf,
    disabled_ids: List[str]
) -> nn.ModuleDict:
    
    daes = nn.ModuleList({
        CompressionDAE(
            in_channels = swivels[layer].channel, 
            in_dim      = arch.spatial[i],
            depth       = arch.depth,
            block_size  = arch.block,
            latent_dim  = arch.latent_dim,
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
    daes = nn.ModuleList([
        ResDAE(
            in_channels = swivels[layer].channel, 
            depth       = arch.depth,
            residual    = arch.residual,
            swivel      = layer
        ) for layer in swivels if layer not in disabled_ids
    ])

    return daes



def get_resMAE(
    arch: OmegaConf,
    swivels: OmegaConf,
    disabled_ids: List[str]
) -> nn.ModuleList:
    maes = nn.ModuleList([
        ResMAEV2(
            in_channels = swivels[layer].channel, 
            depth       = arch.depth,
            p           = arch.p,
            mask        = arch.mask, 
            swivel      = layer
        ) for layer in swivels if layer not in disabled_ids
    ])

    return maes


class IdentityHook(nn.Module):
    def __init__(
        self, 
        swivel: str,
        return_mask: bool = False
    ):
        super().__init__()
        self.swivel = swivel
        self.return_mask = return_mask

    def forward(self, x: Tensor) -> Tensor:
        if self.return_mask:
            return x, None
        else:
            return x


class Masking(nn.Module):
    def __init__(
        self,
        p: float,
        mask = 'per_channel', # per_pixel, random
    ):
        super().__init__()
        self.p = p
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            B, C, H, W = x.shape
            if self.mask == 'per_channel':
                mask = torch.rand((B, C, 1, 1), device=x.device)
            elif self.mask == 'per_pixel':
                mask = torch.rand((B, 1, H, W), device=x.device)
            elif self.mask == 'random':
                mask = torch.rand_like(x, device=x.device)
            else:
                raise NotImplementedError(f'Mask type {self.mask} not implemented.')
            mask = mask.bernoulli_(1-self.p)

            return x * mask, mask
        else:
            return x, None


class ResMAEV2(nn.Module):
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
        p,
        swivel,
        mask: str = 'per_channel'
    ):
        super(ResMAEV2, self).__init__()
        self.on = True
        self.swivel = swivel

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
        self.masking = Masking(
            mask=mask,
            p=p
        )


    def forward(self, x):
        """
        Forward pass of the masked residual AE model.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResDAE model.
        """
        x_out, mask = self.masking(x)
        if not self.training:
            assert mask is None

        for layer in self.model:
            x_out = layer(x_out)

        if self.training:
            return x_out, mask
        else:
            return x_out, 0


class ResMAE(nn.Module):
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
        p,
        swivel,
        residual: bool = True,
        masking: str = 'per_channel'
    ):
        super(ResMAE, self).__init__()
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
        if masking == 'per_channel':
            self.masking = nn.Dropout2d(p=p)


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
            x_out = self.masking(x)
            for layer in self.model:
                x_out = layer(x_out)
            if self.residual:
                return x + x_out
            else:
                return x_out
        else:
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


class CompressionDAE(nn.Module):
    """ Regular AE with vector bottleneck.
    """
    def __init__(
        self, 
        in_channels: int,
        in_dim: int,
        latent_dim: int,
        swivel: str,
        depth: int = 3, 
        block_size: int = 1,
        residual: str = True,
    ):
        super().__init__()
        self.on = True
        self.swivel = swivel
        self.residual = residual
        self.ae = AE(
            in_channels, 
            in_dim, 
            depth=depth, 
            block_size=block_size,
            latent_dim=latent_dim,
            latent = "dense",
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
            x_out = x.clone()
            for layer in self.model:
                x_out = layer(x_out)
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