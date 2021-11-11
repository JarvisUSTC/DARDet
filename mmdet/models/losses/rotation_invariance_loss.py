import mmcv
import torch
import torch.nn as nn
import numpy as np

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def modulated_rotation_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    assert pred.shape[-1] == 5
    pred[...,-1] = pred[...,-1] # transform to angle
    pred[...,0:4] = pred[...,0:4]/(torch.max(target[...,2:4],-1)[0].reshape(-1,1))
    target_norm = target.clone().detach()
    target_norm[...,0:4] = target[...,0:4]/(torch.max(target[...,2:4],-1)[0].reshape(-1,1))
    l_cp = torch.abs(pred[...,0:2] - target_norm[...,0:2]).sum(-1)
    l_1 = l_cp + torch.abs(pred[...,2:]-target_norm[...,2:]).sum(-1)
    l_2 = l_cp + torch.abs(pred[...,2:3]-target_norm[...,3:4]).sum(-1) + torch.abs(pred[...,3:4]-target_norm[...,2:3]).sum(-1) + \
            torch.abs(np.pi/2-torch.abs(pred[...,4:5]-target_norm[...,4:5]).sum(-1))
    loss = torch.min(torch.stack([l_1,l_2]),0)[0]
    return loss



@LOSSES.register_module()
class ModulatedRotationLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ModulatedRotationLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * modulated_rotation_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
