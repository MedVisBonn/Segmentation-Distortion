from typing import List
import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn
from sklearn.covariance import LedoitWolf
from model.wrapper import PoolingMahalanobisWrapper, BatchNormMahalanobisWrapper




def get_pooling_mahalanobis_detector(
    swivels: List[str],
    unet: nn.Module = None,
    pool: str = 'avg2d',
    sigma_algorithm: str = 'default',
    dist_fn: str = 'squared_mahalanobis',
    fit: str  = 'raw', # None, 'raw', 'augmented'
    iid_data: DataLoader = None,
    transform: bool = False,
    lr: float = 1e-3,
    sequential: bool = False,
    device: str  = 'cuda:0',
): 

    pooling_detector = [
        PoolingMahalanobisDetector(
            swivel=swivel,
            device=device,
            pool=pool,
            sigma_algorithm=sigma_algorithm,
            transform=transform,
            dist_fn=dist_fn,
            lr=lr,
        ) for swivel in swivels
    ]
    pooling_wrapper = PoolingMahalanobisWrapper(
        model=unet,
        adapters=nn.ModuleList(pooling_detector),
        copy=True,
        sequential=sequential,
    )
    pooling_wrapper.hook_adapters()
    pooling_wrapper.to(device)
    if fit == 'raw':
        for batch in iid_data:
            x = batch['input'].to(device)
            _ = pooling_wrapper(x)
    elif fit == 'augmented':
        for _ in range(250):
            batch = next(iid_data)
            x = batch['data'].to(device)
            _ = pooling_wrapper(x)
    if fit is not None:
        pooling_wrapper.fit()
        pooling_wrapper.eval()

    return pooling_wrapper



def get_batchnorm_mahalanobis_detector(
    swivels: List[str],
    unet:    nn.Module = None,
    reduce: bool = True,
    aggregate: str = 'mean',
    transform: bool = False,
    lr: float = 1e-3,
    device:  str  = 'cuda:0'
):
    batchnorm_detector = [
        BatchNormMahalanobisDetector(
            swivel=swivel,
            reduce=reduce,
            aggregate=aggregate,
            transform=transform,
            lr=lr,
            device=device,
        ) for swivel in swivels
    ]
    batchnorm_wrapper = BatchNormMahalanobisWrapper(
        model=unet,
        adapters=nn.ModuleList(batchnorm_detector),
    )
    batchnorm_wrapper.hook_adapters()
    batchnorm_wrapper.to(device)
    batchnorm_wrapper.eval()
    
    return batchnorm_wrapper



class PoolingMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel: str,
        pool: str = 'avg2d',
        sigma_algorithm: str = 'default',
        sigma_diag_eps: float = 2e-1,
        transform: bool = False,
        dist_fn: str = 'squared_mahalanobis',
        lr: float = 1e-3,
        device: str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.pool = pool
        self.sigma_algorithm = sigma_algorithm
        self.sigma_diag_eps = sigma_diag_eps
        # self.hook_fn = self.register_forward_pre_hook if hook_fn == 'pre' else self.register_forward_hook
        self.transform = transform
        self.dist_fn = dist_fn
        self.lr = lr
        self.device = device
        # other attributes
        self.active = True
        self.training_representations = []
        # methods
        if self.pool == 'avg2d':
            self._pool = nn.AvgPool2d(
                kernel_size=(2,2), 
                stride=(2,2)
            )
        elif self.pool == 'avg3d':
            self._pool = nn.AvgPool3d(
                kernel_size=(2,2,2),
                stride=(2,2,2)
            )
        elif self.pool == 'none':
            self._pool = None
        else:
            raise NotImplementedError('Choose from: avg2d, avg3d, none')
        
        self.to(device)


    ### private methods ###

    def _reduce(self, x: Tensor) -> Tensor:
        if 'avg' in self.pool:
            # reduce dimensionality with 3D pooling to below 1e4 entries
            while torch.prod(torch.tensor(x.shape[1:])) > 1e4:
                x = self._pool(x)
            x = self._pool(x)
        elif self.pool == 'none':
            pass
        # reshape to (batch_size, 1, n_features)
        x = x.reshape(x.shape[0], 1, -1)
        return x


    @torch.no_grad()
    def _collect(self, x: Tensor) -> None:
        # reduces dimensionality as per self._pool, moves to cpu and stores
        x = self._reduce(x.detach()).cpu()
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
        self.register_buffer(
            'mu',
            self.training_representations.mean(0, keepdims=True).detach().to(self.device)
        )
        
        if self.sigma_algorithm == 'diagonal':
            self.register_buffer(
                'var',
                torch.var(self.training_representations.squeeze(1), dim=0).detach()
            )
            sigma = torch.sqrt(self.var)
            sigma = torch.max(sigma, torch.tensor(self.sigma_diag_eps))
            self.register_buffer(
                'sigma_inv', 
                1 / sigma.detach().to(self.device)
            )

            # self.sigma_inv = torch.max(self.sigma_inv, torch.tensor(self.sigma_diag_eps).to(self.device))
        
        elif self.sigma_algorithm == 'default':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma',
                torch.cov(self.training_representations.squeeze(1).T)
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        elif self.sigma_algorithm == 'ledoitWolf':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma', 
                torch.from_numpy(
                    LedoitWolf().fit(
                        self.training_representations.squeeze(1)
                    ).covariance_
                )
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        else:
            raise NotImplementedError('Choose from: lediotWolf, diagonal, default')



    def _distance(self, x: Tensor) -> Tensor:
        assert self.sigma_inv is not None, 'fit the model first'
        x_reduced  = self._reduce(x)
        x_centered = x_reduced - self.mu
        if self.sigma_algorithm == 'diagonal':
            dist = x_centered**2 * self.sigma_inv
            dist = dist.sum((1,2))
        else:
            dist = x_centered @ self.sigma_inv @ x_centered.permute(0,2,1)
        dist = dist.view((dist.shape[0], ))
        assert len(dist.shape) == 1, 'distances should be 1D over batch dimension'
        assert dist.shape[0] == x.shape[0], 'distance and input should have same batch size'

        if self.dist_fn == 'squared_mahalanobis':
            return dist
        elif self.dist_fn == 'mahalanobis':
            return torch.sqrt(dist)
        else:
            raise NotImplementedError('Choose from: squared_mahalanobis, mahalanobis')


    ### public methods ###

    def fit(self):
        self._merge()
        self._estimate_gaussians()
        # del self.training_representations


    def on(self):
        self.active = True


    def off(self):
        self.active = False


    def forward(self, x: Tensor) -> Tensor:
        if self.active:
            if self.training:
                self._collect(x)
            
            else:
                self.batch_distances = self._distance(x).detach().view(-1)
                if self.transform:
                    x = x.clone().detach().requires_grad_(True)
                    dist = self._distance(x).sum()
                    dist.backward()
                    x.grad.data = torch.nan_to_num(x.grad.data, nan=0.0)
                    x.data.sub_(self.lr * x.grad.data)
                    x.grad.data.zero_()
                    x.requires_grad = False
        else:
            pass
        return x



class BatchNormMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel:    str,
        reduce: bool = True,
        aggregate: str = 'mean',
        transform: bool = False,
        lr: float = 1e-3,
        device:    str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.reduce = reduce
        self.aggregate = aggregate
        self.lr = lr
        self.transform = transform
        self.device = device
        self.to(device)

    ### private methods ###
    
    def _reduce(
        self, 
        x: Tensor
    ) -> Tensor:
        
        if self.aggregate == 'mean':
            x = x.mean(dim=(2,3), keepdim=True)
        elif self.aggregate == 'max':
            x = (x - self.mu.view((1, -1, 1, 1)).abs())
            x = x.amax(dim=(2,3)).unsqueeze(1)
        elif self.aggregate == 'none':
            pass
        else:
            raise NotImplementedError(f'{self.aggregate} not implemented, choose from: [mean, max]')

        return x


    def _distance(
        self, 
        x: Tensor
    ) -> Tensor:
        x = self._reduce(x)
        x = (x - self.mu)**2
        dist = (x * self.sigma_inv)
        if self.reduce:
            dist = torch.sqrt(dist.sum(dim=(1,2,3), keepdim=True))
        return dist


    ### public methods ###
    @torch.no_grad()
    def store_bn_params(
        self,
        params
    ) -> None:
        self.register_buffer('mu', params.running_mean.detach().view(1, -1, 1, 1).to(self.device))
        self.register_buffer('var', params.running_var.detach().view(1, -1, 1, 1).to(self.device))
        self.register_buffer('sigma_inv', 1 / torch.sqrt(self.var))


    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        # implements identity function from a hooks perspective
        if self.training:
            pass
        else:
            if self.reduce:
                self.batch_distances = self._distance(x).detach().cpu().view(-1)
            else:
                self.batch_distances = self._distance(x).detach().cpu().view(x.shape[0], -1)
        
        if self.transform:
            x = x.clone().detach().requires_grad_(True)
            dist = self._distance(x).mean()
            dist.backward()
            x.data.sub_(self.lr * x.grad.data)
            x.grad.data.zero_()
            x.requires_grad = False

        return x