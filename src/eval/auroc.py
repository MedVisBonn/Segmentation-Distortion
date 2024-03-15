
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import ROC
from torchmetrics.utilities.compute import _auc_compute_without_check
from utils import UMapGenerator


def get_auroc_output(
    model: nn.Module,
    iid_data: Dataset,
    ood_data: Dataset,
    net_out: str, 
    dae: bool,
    umap: str,
    device: str = 'cuda:0',
):
    if dae == True:
        model.remove_all_hooks()        
        model.hook_inference_transformations(model.transformations, n_samples=1)
        # Put model in evaluation state
        model.to(device)
        model.eval()
        model.freeze_seg_model()

    auroc = AUROC()
    umap_generator = UMapGenerator(
        method=umap,
        net_out=net_out,
    ).to(device)
    
    batch_size = 32
    iid_loader = DataLoader(iid_data, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=False)

    score_iid = []
    for batch in iid_loader:
        input_ = batch['input'].to(device)
        output = model(input_)
        scores = umap_generator(output, batch_size=input_.shape[0]).cpu()
        score_iid.append(scores.mean(dim=(1,2,3)))
    target_iid = torch.zeros(len(iid_data))


    score_ood = []
    for batch in ood_loader:
        input_ = batch['input'].to(device)
        output = model(input_)
        scores = umap_generator(output, batch_size=input_.shape[0]).cpu()
        score_ood.append(scores.mean(dim=(1,2,3)))
    target_ood = torch.ones(len(ood_data))
    score = torch.cat(score_iid + score_ood)
    target = torch.cat([target_iid, target_ood]).long()

    return auroc(confids=score, target=target)



def get_auroc_mahalanobis(
    wrapper: nn.Module,
    iid_data: Dataset,
    ood_data: Dataset,
    device: str = 'cuda:0',
):
    auroc = AUROC()
    
    batch_size = 32
    iid_loader = DataLoader(iid_data, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=False)

    score_iid = []
    for batch in iid_loader:
        input_ = batch['input'].to(device)
        _ = wrapper(input_)
        score_iid.append({
            adapter.swivel: adapter.batch_distances.cpu().view(-1)
            for adapter in wrapper.adapters
        })
    target_iid = torch.zeros(len(iid_data))


    score_ood = []
    for batch in ood_loader:
        input_ = batch['input'].to(device)
        _ = wrapper(input_)
        score_ood.append({
            adapter.swivel: adapter.batch_distances.cpu().view(-1)
            for adapter in wrapper.adapters
        })
    target_ood = torch.ones(len(ood_data))
    
    score = {
        swivel: torch.cat([conf[swivel] for conf in score_iid + score_ood])
        for swivel in score_iid[0].keys()
    }
    
    target = torch.cat([target_iid, target_ood]).long()


    ret = {
        swivel: auroc(confids=score[swivel], target=target)
        for swivel in score.keys()
    }

    return ret


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


