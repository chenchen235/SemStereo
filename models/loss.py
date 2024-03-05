import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn
from utils import *

import numpy as np

import math

def count_label(labels, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for label in labels:
        label = label.to('cpu')
        counts += np.bincount(label.flatten(), minlength=num_classes)

    return counts.tolist()

def model_loss_train(disp_ests, disp_gts, img_masks):
    weights = [1.0, 0.6, 0.5, 0.3]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)

def model_loss_test(disp_ests, disp_gts, img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon) #* weight
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, num_classes: int=5, ignore_index=5):
    # Dice loss (objective to minimize) between 0 and 1

    if multiclass:
        input = F.softmax(input, dim=1).float()
        target = F.one_hot(target.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()
        if ignore_index: # last
            input = input[:, :-1, ...]
            target = target[:, :-1, ...]
            num_classes = num_classes - 1

        fn = multiclass_dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)
    else:
        input = F.sigmoid(input.squeeze(1))
        fn = dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)       

def focal_loss(preds, targets, alpha=None, gamma=2, ignore_index=-1, smoothing=0.0):
    """
    preds: tensor of predicted probabilities (batch_size, num_classes)
    targets: tensor of true labels (batch_size)
    alpha: tensor of class weights (num_classes)
    gamma: scalar focusing parameter
    ignore_index: index of category to ignore while computing loss
    smoothing: label smoothing parameter
    """
    num_classes = preds.shape[1]
    if alpha == None:
        alpha = torch.ones(num_classes)

    # Apply label smoothing
    if smoothing > 0.0:
        targets = (1 - smoothing) * targets + smoothing / num_classes

    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets.long(), num_classes).float()
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

    # Ignore specific category if required
    if ignore_index >= 0:
        targets_one_hot[:, ignore_index] = 0.0

    # Compute cross-entropy loss
    # print(preds.shape, targets_one_hot.shape)
    ce_loss = F.binary_cross_entropy_with_logits(preds, targets_one_hot)

    # Compute focal loss
    preds_softmax = F.softmax(preds, dim=1)
    pt = preds_softmax * targets_one_hot + (1 - preds_softmax) * (1 - targets_one_hot)
    focal_weight = targets_one_hot * pt.pow(gamma)
    focal_loss = ce_loss * focal_weight.sum(dim=1)

    return focal_loss.mean()

def model_label_loss(masks_preds, true_masks, num_classes, attention_weights_only, ignore=5):

    if ignore:
        criterion = nn.CrossEntropyLoss(ignore_index = ignore).cuda() if num_classes > 1 else nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda() if num_classes > 1 else nn.BCEWithLogitsLoss()

    ce_loss = criterion(masks_preds, true_masks.long())
    dice_loss_ = dice_loss(masks_preds, true_masks, multiclass=True, num_classes = num_classes, ignore_index=ignore)
    loss = ce_loss + dice_loss_
    if attention_weights_only:
        return loss * 1.6
    else:
        return loss * 2.4

def LRSC_loss(label_est_r, disp_ests, y):

    b, height, width = y.size()
    y = y.unsqueeze(1)
    y_warped = torch.full((b, height, width), 5, device=y.device)
    y_coords, x_coords = torch.meshgrid([torch.arange(0, height, device=y.device),
                                 torch.arange(0, width, device=y.device)]) # (H *W)
    x_coords = x_coords.unsqueeze(0).expand(b, -1, -1)
    x_disp = x_coords - disp_ests[0]
    x_disp = torch.clamp(x_disp, min=0, max=(width - 1))    
    pixel_values = torch.gather(y, 3, x_disp.unsqueeze(1).long())
    pixel_values = pixel_values.to(y_warped.dtype)
    y_warped = pixel_values.squeeze(1)
    label_loss_r = label_ce(label_est_r, y_warped, ignore=-1)  #(B, C, D, H, W)
    return label_loss_r

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss

class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss

def label_ce(masks_preds, true_masks, ignore=-1):
    if ignore is None:
        criterion_ce = CrossEntropy()
    else:
        criterion_ce = CrossEntropy(ignore)
    ce_loss = criterion_ce(masks_preds, true_masks.long())
    return ce_loss
    
def label_b(masks_preds, true_masks):
    criterion_b = nn.BCELoss()#BondaryLoss()
    sigmoid = nn.Sigmoid()
    # print("pred: ", sigmoid(masks_preds))
    # print("true: ", true_masks.float())
    b_loss = criterion_b(sigmoid(masks_preds), true_masks.float())#.long()
    return b_loss
