from omegaconf import OmegaConf
import torch
from torch import Tensor, nn
from model.ae import AE, ChannelAE



def get_daes(
    arch: OmegaConf
) -> nn.Module:
    daes = nn.ModuleDict({
        layer: AugResDAE(
            in_channels = arch[layer].channel, 
            in_dim      = arch[layer].spatial,
            latent_dim  = arch[layer].latent,
            depth       = arch[layer].depth,
            block_size  = arch[layer].block,
            residual    = True)
        for layer in arch
    })

    return daes


class resDAE(nn.Module):
    """ Residual Denoising Autoencoder (resDAE) for denoising feature maps.
    """
    def __init__(
        self, 
        in_channels, 
        in_dim, 
        latent_dim=128, 
        depth=3, 
        latent='dense', 
        block_size=1,
        w_prior=2
    ):
        super().__init__()
        self.w_prior = w_prior
        self.on      = True
        self.inspect = False
        
        self.ae    = AE(in_channels, in_dim, latent_dim=latent_dim, depth=depth, latent=latent, block_size=block_size)
        self.prior = AE(in_channels, in_dim, latent_dim=latent_dim, depth=depth, latent=latent, block_size=block_size)


    def turn_off(self) -> None:
        self.on = False
    

    def turn_on(self) -> None:
        self.on = True
    

    def forward(self, x: Tensor) -> Tensor:
        if self.on:
            # TODO: is this save?
            if self.training:
                with torch.no_grad():
                    x += self.prior(x.detach()) * self.w_prior
            res = self.ae(x)
                    
            if self.inspect:
                prior = self.prior(x.detach())                
                return x + res, res, prior
            else:
                return x + res
        else:
            return x

        
class AugResDAE(nn.Module):
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
        residual: str = True
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
