import utils.utils
import torch.nn.functional as F
import torch
import torch.nn as nn

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, obj='bbox'):
        super().__init__()
        self.obj = obj

    def forward(self, pred_bboxes, target_bboxes, target_scores):
        """IoU loss."""
        N_pos = torch.clamp(target_scores.sum(), min=1)
        
        if self.obj == 'bbox':
            iou = utils.utils.bbox_iou(pred_bboxes, target_bboxes, align=True, DIoU = True, CIoU = True)
        else:#cbbox
            iou = utils.utils.cbbox_iou(pred_bboxes, target_bboxes, align=True, DIoU = True)
        
        loss_iou = (1.0 - iou)
        
        loss_iou = weight_reduce_loss(loss_iou, avg_factor=N_pos)
        
        return loss_iou
    
class smooth_l1_loss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    def forward(self, input_, target, label, beta=0.9):
        """Computes SmoothL1Loss"""
        if beta < 1e-5:
            loss = torch.abs(input_ - target)
        else:
            n = torch.abs(input_ - target)
            cond = n < beta
            loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
            #print(loss)
        
        N_pos = torch.clamp(label.sum(), min=1)
        #print(N_pos)
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss#.mean(0).sum()
    
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    def forward(self, pred, label, gamma=2.0, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        #print(loss.shape)
        
        N_pos = torch.clamp(label.sum(), min=1)
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss#.mean(0).sum()

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    #@staticmethod
    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        #pred_score: refers to the predicted IACS
        #gt_score: refers to the IOU(gt box if exist, predicted box)
        #label: refers to 0/1 tensor to indicate object or not object 
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label

        loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
        
        N_pos = max(label.sum(), 1)
        
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss