import torch
from torch import nn as nn
from torch.nn import functional as F

from ndbox.utils import LOSS_REGISTRY, weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super(L1Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}, supported modes: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super(MSELoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}, supported modes: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,  pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)
