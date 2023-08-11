"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""

from typing import Iterable, Dict, Callable, Tuple, List
import torch
from torch import nn
from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d


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