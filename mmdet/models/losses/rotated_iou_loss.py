import torch
import torch.nn as nn
from .rotation_giou import IoU_Rotated_Rectangle, GIoU_Rotated_Rectangle
from mmdet.ops import box_iou_rotated_differentiable
from ..builder import LOSSES
# from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def iou_loss(pred, target, linear=False, eps=1e-6, iou='piou'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if iou=='piou':
        ious = box_iou_rotated_differentiable(pred, target).clamp(min=eps)
    elif iou=='iou':
        temp_list = []
        pred_iou = pred.clone()
        pred_iou[...,2:4] = pred_iou[...,2:4] - 1
        target_iou = target.clone().detach()
        target_iou[...,2:4] = target_iou[...,2:4] - 1
        for pred_, target_ in zip(pred_iou,target_iou):
            temp_list.append(IoU_Rotated_Rectangle(pred_, target_, radian=True))
        ious = torch.stack(temp_list).clamp(min=eps)
    elif iou=='giou':
        temp_list = []
        for pred_, target_ in zip(pred,target):
            temp_list.append(GIoU_Rotated_Rectangle(pred_, target_, radian=True))
        ious = torch.stack(temp_list)
    else:
        raise NotImplemented
    if linear or iou=='giou':
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss

@LOSSES.register_module()
#@LOSSES.register_module
class RotatedIoULoss(nn.Module):

    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0, iou='piou'):
        super(RotatedIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.iou = iou

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
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
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            iou=self.iou,
            **kwargs)
        return loss
