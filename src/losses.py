import torch
from torch import nn
import numpy as np
import torchmetrics
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient


def KL_loss_VAE(mu, log_var):
    kl_loss = 0.5 * torch.mean(torch.sum(torch.exp(log_var) + mu**2 - 1. - log_var, axis=1))
    
    return kl_loss


def loss_VAE(preds, gt, mu, log_var, lambda_=0.01):
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(preds, gt) + lambda_*KL_loss_VAE(mu, log_var) 

    return loss


class FocalLoss(nn.Module):
    """
    BCE loss with focal factor from
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, gamma=2.):
        super().__init__()
        self.gamma = gamma
        self.BCELoss = nn.BCELoss()
        
    def forward(self, preds, target):
        factor = - (1 - preds)**self.gamma
        return (factor * self.BCELoss(preds, target)).mean()
    
    
class WeightedMSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, target, weight):
        return (weight * (preds - target) ** 2).mean()
    
    
class CrossEntropyTargetArgmax(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, preds, target):
        return self.loss_fn(preds, torch.argmax(target, dim=1))
    
    
class SurfaceDiceCalgary(nn.Module):
    def __init__(self, tolerance=1., voxel_spacing=[1., 1., 1.]):
        super().__init__()
        self.tolerance = tolerance
        self.voxel_spacing = voxel_spacing
        
    def forward(self, net_out, target):
        preds = (torch.sigmoid(net_out) > 0.5) * 1
        preds = preds.squeeze().numpy().astype(bool)
        target = target.squeeze().numpy().astype(bool)
        surface_distance = compute_surface_distances(target, preds, self.voxel_spacing)
        surface_dice     = compute_surface_dice_at_tolerance(surface_distance, self.tolerance)
        return torch.tensor(surface_dice)

    
class DiceScoreCalgary(nn.Module):
    
    def __init__(self, eps=1e-6):
        self.eps = eps
        super().__init__()
        
    # does not aggregate over batch dim automatically
    def forward(self, input_, target):
        batch_size = input_.size(0)

        input_ = (torch.sigmoid(input_) > 0.5) * 1
        iflat  = input_.reshape(batch_size, -1)
        tflat  = target.reshape(batch_size, -1)
        
        intersection = (iflat * tflat).sum(dim=1)
        union        = iflat.sum(dim=1) + tflat.sum(dim=1)
        dice         = (2.0 * intersection + self.eps) / (union + self.eps)
        
        return dice
        
    # aggregates over batch dim
    def forward_aggregated(self, input_, target):
        input_ = (torch.sigmoid(input_) > 0.5) * 1
        iflat  = input_.reshape(-1)
        tflat  = target.reshape(-1)

        intersection = (iflat * tflat).sum()
        union        = iflat.sum() + tflat.sum()
        dice         = (2.0 * intersection + self.eps) / (union + self.eps)
 
        return  dice


class DiceScoreMMS(nn.Module):
    
    def __init__(self, eps=1e-8):
        self.eps = eps
        super().__init__()
        
    # does not aggregate over batch dim automatically
    def forward(self, input_, target):
        batch_size = input_.size(0)
        n_classes  = input_.size(1)

        input_ = torch.nn.functional.one_hot(torch.argmax(input_, dim=1), num_classes=4).permute(0,3,1,2)
        iflat  = input_.reshape(batch_size, n_classes, -1)
        tflat  = target.reshape(batch_size, n_classes, -1)
        
        # remove background_class for calculation
        iflat  = iflat[:, 1:]
        tflat  = tflat[:, 1:]
        
        intersection = (iflat * tflat).sum(dim=2)
        union        = iflat.sum(dim=2) + tflat.sum(dim=2)
        dice         = (2.0 * intersection + self.eps) / (union + self.eps)
        
        return dice #[:, 1:]
    
    
class AccMMS(nn.Module):
    
    def __init__(self):
        super().__init__()
        # init acc from torchmetrics for the M&M data, i.e.
        # multiclass with 4 classes, micro averaging, 
        # background included (ignore_index=0 if needed),
        # and sample-wise averaging to get a score per slice
        self.acc = torchmetrics.Accuracy(
            task='multiclass', 
            num_classes=4, 
            multidim_average='samplewise', 
            average='micro',
#             ignore_index=0
        )
        
    # does not aggregate over batch dim automatically
    def forward(self, input_, target):
        target = torch.argmax(target, dim=1)
        return self.acc(input_, target)

        
class SampleDice(nn.Module):
    def __init__(self, data='calgary'):
        super().__init__()
        self.dice = DiceScoreCalgary() if data == 'calgary' else DiceScoreMMS()
    
    def forward(self, unet_out, samples, target):
        return self.dice(samples, target)
    
    
class SampleSurfaceDice(nn.Module):
    def __init__(self):
        self.dice = SurfaceDiceCalgary()
        
    def forward(self, unet_out, samples, target):
        self.dice(samples, target)
    
    
class UnetDice(nn.Module):
    def __init__(self, data='calgary'):
        super().__init__()
        self.dice = DiceScoreCalgary() if data == 'calgary' else DiceScoreMMS()
        
    def forward(self, unet_out, samples, target):
        return self.dice(unet_out, target)
    
    
class UnetSurfaceDice(nn.Module):
    def __init__(self):
        self.dice = SurfaceDiceCalgary()
        
    def forward(self, unet_out, samples, target):
        self.dice(unet_out, target)
    
    
# class CalgaryCriterionVAE(nn.Module):
#     def __init__(self, diff=True, loss='huber'):
#         super().__init__()
#         #self.bce   = nn.BCELoss()
#         self.diff  = diff
#         #self.mse   = nn.MSELoss()
#         #self.loss  = nn.HuberLoss()
        
#         self.id_loss = loss
#         if loss == 'huber':
#             self.loss = nn.HuberLoss()
#         elif loss == 'bce':
#             self.loss = nn.BCEWithLogitsLoss()
        
#     def forward(self, unet_out, samples, train_data, beta):
        
#         #preds = torch.sigmoid(samples)
#         #with torch.no_grad():
#         #    target = torch.sigmoid(unet_out.detach())    
#         #loss = self.bce(preds, target)
        
        
#         loss = self.loss(samples, unet_out)
#         metrics = {'output_mse': loss.item()}
#         if not self.diff:
#             loss *= 0
        
#         for layer in train_data:
#             metrics[layer] = {}
            
#             mse = train_data[layer]['mse']
#             kl  = train_data[layer]['kl']
#             loss += mse + beta[layer] * kl
            
#             metrics[layer]['mse'] = mse.item()
#             metrics[layer]['kl']  = kl.item()
#             metrics[layer]['mu']  = train_data[layer]['mu'].detach().mean().item()
#             metrics[layer]['var'] = torch.exp(train_data[layer]['log_var']).detach().mean().item()
            
#         return loss, metrics
    
    
class CalgaryCriterionAE(nn.Module):
    def __init__(self, recon=True, diff=True, loss='huber'):
        super().__init__()
        self.diff  = diff
        self.recon = recon
        self.id_loss = loss
        if loss == 'huber':
            self.loss = nn.HuberLoss()
        elif loss == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, unet_out, samples, train_data):
        
        if self.diff:
            loss = self.loss(samples, unet_out)
            metrics = {'output_diff': loss.item()}
        else:
            loss = 0
            metrics={}
            
        if self.recon:
            mse = 0
            for layer in train_data:
                metrics[layer] = {}
                mse_layer = train_data[layer]['mse']
                mse += mse_layer
                metrics[layer]['mse'] = mse_layer.item()
            loss += mse
            
        return loss, metrics
    
    
class MNMCriterionAE(nn.Module):
    def __init__(self, recon=True, diff=True, loss='huber'):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss()
        self.diff  = diff
        self.recon = recon
        self.id_loss = loss
        if loss == 'huber':
            self.loss = nn.HuberLoss()
        elif loss == 'ce':
            self.loss = CrossEntropyTargetArgmax()
        self.m = nn.Softmax(dim=1)
        
    def forward(self, unet_out, samples, train_data):
        
        if self.diff:
            #print("Diff: yse")
            #print(samples.shape, samples.min(), samples.max(), unet_out.shape, unet_out.min(), unet_out.max())
            loss = self.loss(samples, unet_out)
            metrics = {'output_diff': loss.item()}
            #print(loss.item())
        else:
            loss=0
            metrics={}
        
        
        #print("Recon: yes")
        if self.recon:
            mse = 0
            for layer in train_data:
                metrics[layer] = {}
                mse_layer = train_data[layer]['mse']
                mse += mse_layer
                metrics[layer]['mse'] = mse_layer.item()
        
            loss += mse
            
        return loss, metrics    
    
    
class MNMCriterion(nn.Module):
    def __init__(self, diff=True, loss='huber'):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss()
        self.diff  = diff
        self.id_loss = loss
        if loss == 'huber':
            self.loss = nn.HuberLoss()
        elif loss == 'ce':
            self.loss = CrossEntropyTargetArgmax()
        # self.ce    = nn.CrossEntropyLoss()
        self.m = nn.Softmax(dim=1)
        
    def forward(self, unet_out, samples, train_data, beta):
        
        #samples_pred = m(samples)
        #with torch.no_grad():
        #    unet_out_target = self.m(unet_out)
            
        #assert samples.shape == unet_out_target.shape, "input and output differ in shape!"
        
        
        loss = self.loss(samples, unet_out)
        
        #print('output:', loss.item())
        # loss = self.huber(samples, unet_out)
        #loss = self.ce(samples, unet_out_target)
        metrics = {'output_diff': loss.item()}
        if not self.diff:
            loss *= 0
        
        for layer in train_data:
            metrics[layer] = {}
            
            mse   = train_data[layer]['mse']# * 100
            #print("mse:", mse.item())
            kl    = train_data[layer]['kl']
            #print('kl:', (beta[layer] * kl).item())
            loss += mse + beta[layer] * kl
            
            metrics[layer]['mse'] = mse.item()
            metrics[layer]['kl']  = kl.item()
            metrics[layer]['mu']  = train_data[layer]['mu'].detach().mean().item()
            metrics[layer]['var'] = torch.exp(train_data[layer]['log_var']).detach().mean().item()
            
        return loss, metrics
    

    
    
    
    
class RefineCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, unet_out, samples, train_data, beta):
   
        loss = torch.tensor(0)
         
        metrics = {'output_mse': loss.item()}
        
        for layer in train_data:
            metrics[layer] = {}
            
            mse = train_data[layer]['mse']
            kl  = train_data[layer]['kl']
            loss += mse + beta[layer] * kl
            
            metrics[layer]['mse'] = mse.item()
            metrics[layer]['kl']  = kl.item()
            metrics[layer]['mu']  = train_data[layer]['mu'].detach().mean().item()
            metrics[layer]['var'] = torch.exp(train_data[layer]['log_var']).detach().mean().item()
            
        return loss, metrics
    
    
    
