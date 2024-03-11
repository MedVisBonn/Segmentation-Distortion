
import torch
from torch import Tensor, nn
from torchmetrics import ROC
from torchmetrics.utilities.compute import _auc_compute_without_check



class AUROC(nn.Module):
    def __init__(
        self, 
        ret_curves: bool = True
    ):
        super().__init__()
        # args
        self.ret_curves = ret_curves
        # metrics
        self.roc_values = ROC(task='binary')

    @torch.no_grad()
    def auroc(
        self,
        confids: Tensor,
        target: Tensor
    ):
        
        confids = (confids - confids.min()) / (confids.max() - confids.min())
        fpr, tpr, _ = self.roc_values(
            confids,
            target
        )

        ret = _auc_compute_without_check(
            x=fpr,
            y=tpr,
            direction=1.0
        )

        if self.ret_curves:
            return ret, fpr, tpr
        else:
            return ret

    @torch.no_grad()
    def forward(
        self,
        confids: Tensor,
        target: Tensor
    ):
        return self.auroc(confids=confids, target=target)