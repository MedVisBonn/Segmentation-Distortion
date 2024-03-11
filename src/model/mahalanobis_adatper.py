import torch
from torch import Tensor, nn
from sklearn.covariance import LedoitWolf



class PoolingMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel:     str,
        ledoitWolf: bool = True,
        # hook_fn:    str  = 'pre',
        transform:  bool = False,
        device:     str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.ledoitWolf = ledoitWolf
        # self.hook_fn = self.register_forward_pre_hook if hook_fn == 'pre' else self.register_forward_hook
        self.transform = transform
        self.device = device
        # other attributes
        self.training_representations = []
        # methods
        self._pool = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2))


    ### private methods ###

    @torch.no_grad()
    def _reduce(self, x: Tensor) -> Tensor:
        # reduce dimensionality with 3D pooling to below 1e4 entries
        while torch.prod(torch.tensor(x.shape[1:])) > 1e4:
            x = self._pool(x)
        x = self._pool(x)
        # reshape to (batch_size, 1, n_features)
        x = x.reshape(x.shape[0], 1, -1)
        return x


    @torch.no_grad()
    def _collect(self, x: Tensor) -> None:
        # reduces dimensionality, moves to cpu and stores
        x = self._reduce(x).cpu()
        self.training_representations.append(x)


    @torch.no_grad()
    def _merge(self) -> None:
        # concatenate batches from training data
        self.training_representations = torch.cat(
            self.training_representations,
            dim=0
        )


    @torch.no_grad()
    def _estimate_gaussians(self) -> None:
        self.mu = self.training_representations.mean(0, keepdims=True).detach().to(self.device)
        if self.ledoitWolf:
            self.sigma = torch.from_numpy(
                LedoitWolf().fit(
                    self.training_representations.squeeze(1)
                ).covariance_
            )
        else:
            self.sigma = torch.cov(self.training_representations.squeeze(1).T)
        self.sigma_inv = torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)


    def _distance(self, x: Tensor) -> Tensor:
        assert self.sigma_inv is not None, 'fit the model first'
        # assert self.device == x.device, 'input and model device must match'
        x_reduced  = self._reduce(x)
        x_centered = x_reduced - self.mu
        dist       = x_centered @ self.sigma_inv @ x_centered.permute(0,2,1)

        return torch.sqrt(dist)


    ### public methods ###

    def fit(self):
        self._merge()
        self._estimate_gaussians()
        del self.training_representations


    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._collect(x)
        
        else:
            self.batch_distances = self._distance(x)

        # implements identity function from a hooks perspective
        if self.transform:
            raise NotImplementedError('Implement transformation functionality')
        else:
            return x
        


class BatchNormMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel:    str,
        transform: bool = False,
        device:    str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.transform = transform
        self.device = device


    ### private methods ###
    
    def _reduce(
        self, 
        x: Tensor
    ) -> Tensor:
        x = x.mean(dim=(2,3)).unsqueeze(1)

        return x


    def _distance(
        self, 
        x: Tensor
    ) -> Tensor:
        x = self._reduce(x)
        x = (x - self.mu)**2
        dist = (x * self.sigma_inv).sum(dim=(1,2), keepdim=True)

        return torch.sqrt(dist)


    ### public methods ###
    @torch.no_grad()
    def store_bn_params(
        self,
        params
    ) -> None:
        self.mu  = params.running_mean.detach().unsqueeze(0).to(self.device)
        self.var = params.running_var.detach().unsqueeze(0).to(self.device)
        self.sigma_inv = 1 / torch.sqrt(self.var) 


    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        if self.training:
            pass
        else:
            self.batch_distances = self._distance(x).detach().view(-1)

        # implements identity function from a hooks perspective
        if self.transform:
            raise NotImplementedError('Implement transformation functionality')
        else:
            return x