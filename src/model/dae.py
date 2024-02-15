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
    
    model = cfg.dae.model
    arch = cfg.dae.arch
    disabled_ids = cfg.dae.trainer.disabled_ids

    if model == 'unet':
        daes = get_unetDAE(arch)
    elif model == 'channelDAE':
        daes = get_channelDAE(arch)
    elif model == 'resDAE':
        daes = get_resDAE(arch)
    else:
        raise NotImplementedError(f'Model {model} not implemented.')
    
    for layer in disabled_ids:
        print(f'Disabling layer {layer}.')
        daes[layer] = nn.Identity()

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
        model_name = f'{data_key}_{cfg.dae.name}_{iteration}_best.pt'
        model_path = f'{root}pre-trained-tmp/trained_AEs/{model_name}'
        state_dict = torch.load(model_path)['model_state_dict']
        return model, state_dict
    else:
        return model



def get_unetDAE(
    arch: OmegaConf,
) -> nn.ModuleDict:
    daes = nn.ModuleDict({
        layer: UnetDAE(
            in_channels = arch[layer].channel, 
            num_res_units = arch[layer].num_res_units,
            residual = True  
        ) for layer in arch
    })

    return daes



def get_channelDAE(
    arch: OmegaConf,
) -> nn.ModuleDict:
    daes = nn.ModuleDict({
        layer: ChannelDAE(
            in_channels = arch[layer].channel, 
            in_dim      = arch[layer].spatial,
            latent_dim  = arch[layer].latent,
            depth       = arch[layer].depth,
            block_size  = arch[layer].block,
            residual    = True
        ) for layer in arch
    })

    return daes



def get_resDAE(
    arch: OmegaConf,
) -> nn.ModuleDict:
    daes = nn.ModuleDict({
        layer: ResDAE(
            in_channels = arch[layer].channel, 
            depth       = arch[layer].depth,
            residual    = True
        ) for layer in arch
    })

    return daes

        
class ChannelDAE(nn.Module):
    """ Residual Denoising Autoencoder (resDAE) for denoising feature maps.
    """
    def __init__(
        self, 
        in_channels: int, 
        in_dim: int, 
        latent_dim: int = 128, 
        depth: int = 3, 
        latent: str = 'dense', 
        block_size: int = 1,
        residual: str = True,
    ):
        super().__init__()
        self.on = True
        self.residual = residual
        self.ae = ChannelAE(
            in_channels, 
            in_dim, 
#            latent_dim=latent_dim, 
            depth=depth, 
#            latent=latent, 
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

    def __init__(self, in_channels, depth, residual=True):
        super(ResDAE, self).__init__()
        self.on = True
        self.residual = residual

        self.model = nn.ModuleList(
            ResidualUnit(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=in_channels,
                act="PReLU",
                norm="BATCH",
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