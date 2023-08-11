import torch
from torch import Tensor, nn
from model.ae import AE 


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
