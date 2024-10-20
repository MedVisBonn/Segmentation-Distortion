from typing import List, Tuple
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchmetrics.functional.classification import dice
from utils import UMapGenerator



def get_eaurc_output(
    model: nn.Module,
    data: Dataset,
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
        model.freeze_seg_model()
    model.eval()

    
    umap_generator = UMapGenerator(
        method=umap,
        net_out=net_out,
    ).to(device)    
    eaurc = EAURC()

    batch_size = 32

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    score, risk = [], []
    for batch in loader:
        # grab data
        input_ = batch['input'].to(device)
        target = batch['target'].to(device).long()
        target[target == -1] = 0
        # inference
        output = model(input_)
        if net_out == 'brain':
            segmap = (torch.sigmoid(output[:batch_size]) > 0.5) * 1
        elif net_out == 'heart':
            segmap = torch.argmax(output[:batch_size], dim=1, keepdims=True)
        # collect risk and score
        scores = umap_generator(output, batch_size=input_.shape[0]).cpu()
        risk.append(
            torch.tensor([1 - dice(
                    s.flatten(), 
                    t.flatten(),
                    num_classes=2 if net_out == 'brain' else 4,
                    zero_division=1,
                    average='micro'
                ) for s,t in zip(segmap,target)]
            ).cpu()
        )
        score.append(scores.mean(dim=(1,2,3)))
    # aggregate results
    risk = torch.cat(risk)
    score = torch.cat(score)

    # compute eaurc and return
    return eaurc(score=score, risks=risk)




def get_eaurc_mahalanobis(
    wrapper: nn.Module,
    data: Dataset,
    net_out: str,
    device: str = 'cuda:0',
):
    eaurc = EAURC()
    batch_size = 32

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    score, risk = [], []
    for batch in loader:
        # grab data
        input_ = batch['input'].to(device)
        target = batch['target'].to(device).long()
        n_empty = (target>0).sum((1,2,3))==0
        n = input_.shape[0]
        target[target == -1] = 0
        # inference
        output = wrapper(input_)
        if net_out == 'brain':
            segmap = (torch.sigmoid(output) > 0.5) * 1
        elif net_out == 'heart':
            segmap = torch.argmax(output, dim=1, keepdims=True)
        # collect risk and score
        # print(segmap.shape, torch.unique(segmap), target.shape, target.unique())
        risk.append(
            torch.tensor([1 - dice(
                    s.flatten(), 
                    t.flatten(),
                    num_classes=2 if net_out == 'brain' else 4,
                    zero_division=1,
                    average='micro'
                ) for s,t in zip(segmap,target)]
            ).cpu()
        )
        score.append(
            wrapper.aggregate_adapter_scores()
        )
        # score.append({
        #     adapter.swivel: adapter.batch_distances.cpu().view(-1)
        #     for adapter in wrapper.adapters
        # })
    # aggregate results
    risk = torch.cat(risk)
    score = torch.cat(score)
    # print(f'Avrg DSC: {risk.mean():.4f}, Shape: {risk.shape}')
    # score = {
    #     swivel: torch.cat([conf[swivel] for conf in score])
    #     for swivel in score[0].keys()
    # }
    # compute eaurc
    # ret = {
    #     swivel: eaurc(score=score[swivel], risks=risk)
    #     for swivel in score.keys()
    # }
    ret = eaurc(score=score, risks=risk)

    return ret



def get_eaurc_mahalanobis_propagated(
    wrapper: nn.Module,
    data: Dataset,
    umap: str,
    net_out: str,
    device: str = 'cuda:0',
):
    eaurc = EAURC()
    batch_size = 32

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    umap_generator = UMapGenerator(
        method=umap,
        net_out=net_out,
    ).to(device)
    
    score, risk = [], []
    for batch in loader:
        # grab data
        input_ = batch['input'].to(device)
        target = batch['target'].long()

        n_empty = (target>0).sum((1,2,3))==0
        n = input_.shape[0]
        target[target == -1] = 0
        # inference
        wrapper.set_transform(False)
        output_original = wrapper(input_)
        wrapper.set_transform(True)
        output_transformed = wrapper(input_)
        # output = wrapper(input_)
        if net_out == 'brain':
            segmap = (torch.sigmoid(output_original) > 0.5) * 1
        elif net_out == 'heart':
            segmap = torch.argmax(output_original, dim=1, keepdims=True)

        # collect risk and score
        # print(segmap.shape, torch.unique(segmap), target.shape, target.unique())
        risk.append(
            torch.tensor([1 - dice(
                    s.flatten(), 
                    t.flatten(),
                    num_classes=2 if net_out == 'brain' else 4,
                    zero_division=1,
                    average='micro'
                ) for s,t in zip(segmap,target)]
            ).cpu()
        )
        scores = umap_generator(
            torch.cat(
                [output_original, output_transformed],
                dim=0
            ), 
            batch_size=input_.shape[0]
        ).cpu()
        score.append(scores.mean(dim=(1,2,3)))
        # score.append(
        #     wrapper.aggregate_adapter_scores()
        # )
        # score.append({
        #     adapter.swivel: adapter.batch_distances.cpu().view(-1)
        #     for adapter in wrapper.adapters
        # })
    # aggregate results
    risk = torch.cat(risk)
    score = torch.cat(score)
    # print(f'Avrg DSC: {risk.mean():.4f}, Shape: {risk.shape}')
    # score = {
    #     swivel: torch.cat([conf[swivel] for conf in score])
    #     for swivel in score[0].keys()
    # }
    # compute eaurc
    # ret = {
    #     swivel: eaurc(score=score[swivel], risks=risk)
    #     for swivel in score.keys()
    # }
    ret = eaurc(score=score, risks=risk)

    return ret



class EAURC(nn.Module):
    def __init__(
        self,
        ret_curves: bool = True
    ):
        super().__init__()
        # args
        self.ret_curves = ret_curves
        
    @torch.no_grad()
    def _rc_curve_stats(
        self,
        confids: np.array, 
        risks: np.array
    ) -> Tuple[List[float], List[float], List[float]]:
        coverages = []
        selective_risks = []
        assert (
            len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)
        )

        n_samples = len(risks)
        idx_sorted = np.argsort(confids)

        coverage = n_samples
        error_sum = sum(risks)

        coverages.append(coverage / n_samples)
        selective_risks.append(error_sum / n_samples)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - risks[idx_sorted[i]]
            tmp_weight += 1
            if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_samples)
                selective_risks.append(error_sum / (n_samples - 1 - i))
                weights.append(tmp_weight / n_samples)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            selective_risks.append(selective_risks[-1])
            weights.append(tmp_weight / n_samples)

        return coverages, selective_risks, weights

    @torch.no_grad()
    def _aurc(
        self,
        confids: np.array, 
        risks: np.array
    ):
        _, risks, weights = self._rc_curve_stats(confids=confids, risks=risks)

        ret = sum(
            [(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]
        )

        if self.ret_curves:
            return ret, risks, weights
        else:
            return ret

    @torch.no_grad()
    def eaurc(
        self,
        score: Tensor, 
        risks: Tensor
    ):
        """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
        n = len(risks)
        score = score.numpy()
        score = (score - score.min()) / (score.max() - score.min())
        confids = 1 - score
        risks = risks.numpy()
        # optimal confidence sorts risk. Asencding here because we start from coverage 1/n
        selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
        aurc_opt = selective_risks.sum() / n
        ret, risks, weights = self._aurc(confids=confids, risks=risks)
        
        if self.ret_curves:
            risks   = torch.flip(torch.tensor(risks[:-1]), [0])
            weights = torch.cumsum(torch.flip(torch.tensor(weights), [0]), 0)
            selective_risks = np.interp(
                np.linspace(0, 1, len(weights)),
                np.linspace(0,1,len(selective_risks)),
                selective_risks
            )
            # selective_risks = torch.nn.functional.interpolate(selective_risks, size=weights.shape)
            return ret - aurc_opt, ret, risks, weights, selective_risks
        else:
            return ret - aurc_opt, ret

    @torch.no_grad()
    def forward(
        self,
        score: Tensor,
        risks: Tensor
    ):
        return self.eaurc(score=score, risks=risks)
    

