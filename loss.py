import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a
        
    
class dice_loss(nn.Module):
    def __init__(self, batch=True, cosh=False):
        super(dice_loss, self).__init__()
        self.batch = batch
        self.cosh = cosh
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        
        if self.cosh:
            b = torch.log(torch.cosh(b))
            
        return b



def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))

class my_focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        
        Parameters:
        - alpha: balancing factor for positive and negative classes
        - gamma: focusing parameter
        - reduction: specifies the reduction to apply to the output, 'none' | 'mean' | 'sum'
        """
        super(my_focal_loss, self).__init__()
        self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
        
        self.bce_loss = nn.BCELoss()
        
    def __call__(self, y_true, y_pred):
        # Convert y_pred to probabilities with sigmoid
        y_pred = torch.sigmoid(y_pred)
        
        # Calculate the binary cross entropy
        BCE_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        # Calculate the focal weight
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = (1 - pt) ** self.gamma

        # Apply the focal weight to the BCE loss
        focal_loss = self.alpha * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class combo_loss_v2(nn.Module):
    def __init__(self, beta=0.75, alpha=0.6, batch=True):
        super(combo_loss_v2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.batch = batch
    
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
           
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def neg_weighted_bce_loss(self, y_true, y_pred, eps=1e-9):
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)
        
        inputs = torch.clamp(y_pred, eps, 1.0 - eps)
        
        out = - (self.beta * ((y_true * torch.log(inputs)) + ((1 - self.beta) * (1.0 - y_true) * torch.log(1.0 - inputs))))
        
        weighted_ce = out.mean()
        
        return weighted_ce
    
#         # Apply the beta weighting for false negatives in BCE
#         bce_weighted = self.beta * y_true + (1 - self.beta) * (1 - y_true)
#         bce_loss = F.binary_cross_entropy(y_pred, y_true, weight=bce_weighted, reduction='mean')
        
#         return -bce_loss
    
    def __call__(self, y_true, y_pred):
        weighted_ce = self.neg_weighted_bce_loss(y_pred, y_true)
        dice = self.soft_dice_loss(y_true, y_pred)
        
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice)
        
        return combo

class double_loss_assigned(nn.Module):
    def __init__(self, pos_weight=12.0, neg_weight=1.0):
        super(double_loss_assigned, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())

    def batched_dice_coeff(self, y_true, y_pred):
        # Convert y_pred to probabilities with sigmoid
        y_pred = torch.sigmoid(y_pred)

        epsilon=1e-8

        i_true_fg = torch.sum(y_true)
        j_true_fg = torch.sum(y_pred)
        
        intersection_true_fg = torch.sum(y_true * y_pred)
        union_fg = i_true_fg + j_true_fg   # Foreground union

        i_true_bg = torch.sum(1-y_true)
        j_true_bg = torch.sum(1-y_pred)

        intersection_true_bg = torch.sum((1-y_true) * (1-y_pred))
        union_bg = i_true_bg + j_true_bg   # Background union

        dice_fg = (2 * intersection_true_fg + epsilon) / (union_fg + epsilon)
        dice_bg = (2 * intersection_true_bg + epsilon) / (union_bg + epsilon)

        score_dice = (self.pos_weight * dice_fg + self.neg_weight * dice_bg) / (self.pos_weight + self.neg_weight)

        return score_dice.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.batched_dice_coeff(y_true, y_pred)
        return loss
    
    def __call__(self, y_true, y_pred):
        a = self.bce_logit_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        
        return a+b
        
class double_loss_batch(nn.Module):
    def __init__(self):
        super(double_loss_batch, self).__init__()
        

    def batched_dice_coeff(self, y_true, y_pred):
        # Convert y_pred to probabilities with sigmoid
        y_pred = torch.sigmoid(y_pred)

        epsilon=1e-7

        # first, count occurrences of zeros
        neg_elem_count = torch.numel(y_true) - torch.sum(y_true)
        # foreground weight
        pos_weight = neg_elem_count / torch.sum(y_true)

        # constant background weight
        neg_weight = 1.0

        i_true_fg = torch.sum(y_true)
        j_true_fg = torch.sum(y_pred)
        
        intersection_true_fg = torch.sum(y_true * y_pred)
        union_fg = i_true_fg + j_true_fg   # Foreground union

        i_true_bg = torch.sum(1-y_true)
        j_true_bg = torch.sum(1-y_pred)

        intersection_true_bg = torch.sum((1-y_true) * (1-y_pred))
        union_bg = i_true_bg + j_true_bg   # Background union

        dice_fg = (2 * intersection_true_fg + epsilon) / (union_fg + epsilon)
        dice_bg = (2 * intersection_true_bg + epsilon) / (union_bg + epsilon)

        score_dice = (pos_weight * dice_fg + neg_weight * dice_bg) / (pos_weight + neg_weight)

        return score_dice.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.batched_dice_coeff(y_true, y_pred)
        return loss
    
    def inner_bce_logit_loss(self, y_true, y_pred):
        # first, count occurrences of zeros
        neg_elem_count = torch.numel(y_true) - torch.sum(y_true)
        # foreground weight
        pos_weight = neg_elem_count / torch.sum(y_true)

        bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())

        return bce_logit_loss(y_pred, y_true)
    
    def __call__(self, y_true, y_pred):
        a = self.inner_bce_logit_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        
        return a+b

        
class double_loss(nn.Module):
    def __init__(self, pos_weight=2.5, neg_weight=1., batch=True):
        super(double_loss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())
    
    def weighted_dice_coeff(self, y_true, y_pred):
        # Convert y_pred to probabilities with sigmoid
        y_pred = torch.sigmoid(y_pred)
        
        i_true = self.pos_weight * torch.sum(y_true)
        j_true = self.pos_weight * torch.sum(y_pred)
        intersection_true = torch.sum(self.pos_weight * y_true * y_pred)
                
        i_true_bg = torch.sum(1-y_true)
        j_true_bg = torch.sum(1-y_pred)
        intersection_true_bg = torch.sum((1-y_true) * (1-y_pred))
        
        # dice_fg = (2 * intersection_true + epsilon) / (union_fg + epsilon)
        # dice_bg = (2 * intersection_true_bg + epsilon) / (union_bg + epsilon)

        score_nom = 2 * ((self.pos_weight * intersection_true) + (self.neg_weight * intersection_true_bg))
        score_denom = (self.pos_weight * (i_true + j_true)) + (self.neg_weight * (i_true_bg + j_true_bg))
        
        # score_nom = 2 * (intersection_true)
        # score_denom = (i_true + j_true)

        score_dice = score_nom/score_denom
        
        return score_dice.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.weighted_dice_coeff(y_true, y_pred)
        return loss
    
    def __call__(self, y_true, y_pred):
        a = self.bce_logit_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        
        return a+b

class weighted_bce_loss(nn.Module):
    def __init__(self, pos_weight=10.):
        super(weighted_bce_loss, self).__init__()
        self.bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())
        
    def __call__(self, y_true, y_pred):
        return self.bce_logit_loss(y_pred, y_true)

    
class weighted_dice_loss(nn.Module):
    def __init__(self, pos_weight=10., neg_weight=1.):
        super(weighted_dice_loss, self).__init__()
        self.pos_weight=pos_weight
        self.neg_weight=neg_weight
    
    def weighted_dice_coeff(self, y_true, y_pred):
        # Convert y_pred to probabilities with sigmoid
        y_pred = torch.sigmoid(y_pred)
        
        i_true = torch.sum(y_true)
        j_true = torch.sum(y_pred)
        intersection_true = torch.sum(y_true * y_pred)
                
        i_true_bg = torch.sum(1-y_true)
        j_true_bg = torch.sum(1-y_pred)
        intersection_true_bg = torch.sum((1-y_true) * (1-y_pred))
        
        score_nom = 2 * ((self.pos_weight * intersection_true) + (self.neg_weight * intersection_true_bg))
        score_denom = (self.pos_weight * (i_true + j_true)) + (self.neg_weight * (i_true_bg + j_true_bg))
        
        score_dice = score_nom/score_denom
        
        return score_dice.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.weighted_dice_coeff(y_true, y_pred)
        return loss
    
    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred)

    
class my_dual_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=10., neg_weight=1., reduction='mean', squared=False, cosh=True, batch=True):
        """
        Focal Loss for binary classification.
        
        Parameters:
        - alpha: balancing factor for positive and negative classes
        - gamma: focusing parameter
        - reduction: specifies the reduction to apply to the output, 'none' | 'mean' | 'sum'
        """
        super(my_dual_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        self.batch = batch
        self.squared = squared
        self.cosh = cosh
    
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
            
        if self.squared:
            i = i ** 2
            j = j ** 2
            
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def weighted_dice_coeff(self, y_true, y_pred):
        i_true = torch.sum(y_true)
        j_true = torch.sum(y_pred)
        intersection_true = torch.sum(y_true * y_pred)
                
        i_true_bg = torch.sum(1-y_true)
        j_true_bg = torch.sum(1-y_pred)
        intersection_true_bg = torch.sum((1-y_true) * (1-y_pred))
        
        score_nom = 2 * ((self.pos_weight * intersection_true) + (self.neg_weight * intersection_true_bg))
        score_denom = (self.pos_weight * (i_true + j_true)) + (self.neg_weight * (i_true_bg + j_true_bg))
        
        score_dice = score_nom/score_denom
        
        return score_dice.mean()
    
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
#         loss = 1 - self.weighted_dice_coeff(y_true, y_pred)
        return loss
    
    def focal_loss(self, y_true, y_pred):
        # Calculate the binary cross entropy
        BCE_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        # Calculate the focal weight
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_mat = torch.where(y_true == 1, self.alpha, 1 - self.alpha)

        # Apply the focal weight to the BCE loss
        focal_loss = alpha_mat * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def __call__(self, y_true, y_pred):
        a = self.soft_dice_loss(y_true, y_pred)
        
        if self.cosh:
            a = torch.log(torch.cosh(a))
            
        b = self.focal_loss(y_true, y_pred)
        
        return a+b
    
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)

        return a + b

class my_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(my_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        return a

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input):
        target1 = torch.squeeze(target, dim=1)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target2 = target1.view(-1,1).long()

        logpt = F.log_softmax(input, dim=1)
        # print(logpt.size())
        # print(target2.size())
        logpt = logpt.gather(1,target2)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
