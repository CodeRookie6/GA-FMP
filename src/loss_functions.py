#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/27 20:54
# @Author : fhh
# @FileName: loss_functions.py
# @Software: PyCharm

import torch
import torch.nn as nn




#定义了一个自定义的损失函数 BCEFocalLoss，这是一种修改过的交叉熵损失函数，用于处理类别不平衡的情况，并引入了焦点损失（Focal Loss）的概念
class BCEFocalLoss(nn.Module):
    """Focal loss"""
    #义了类的初始化方法，初始化了焦点损失的参数 gamma、损失的缩减方式 reduction 和类别权重 class_weight。super(BCEFocalLoss, self).__init__()
    # 调用了父类 nn.Module 的初始化方法。
    def __init__(self, gamma=2, reduction='mean', class_weight=None):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reducation = reduction
        self.class_weight = class_weight

    def forward(self, data, label):
        sigmoid = nn.Sigmoid()
        pt = sigmoid(data).detach()
        #创建了一个 sigmoid 激活函数，并将模型的输出 data 通过 sigmoid 函数进行处理，得到预测的概率值 p，
        # 并使用 .detach() 方法使得这部分计算不参与梯度计算。
        if self.class_weight is not None:
            label_weight = ((1 - pt) ** self.gamma) * self.class_weight
            # label_weight = torch.exp((1 - pt)) * self.class_weight
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma) * self.class_weight
        else:
            label_weight = (1 - pt) ** self.gamma
            # label_weight = torch.exp((1 - pt))
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma)
        #根据是否提供了类别权重 class_weight，计算了焦点损失的权重 label_weight。若提供了类别权重，则将其乘以焦点损失的计算中。

        focal_loss = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')
        return focal_loss(data, label)
        #利用 PyTorch 中内置的 nn.BCEWithLogitsLoss 损失函数，传入焦点损失的权重 label_weight，计算最终的焦点损失，并返回计算结果。

#定义了一个自定义的损失函数 AsymmetricLoss，即不对称损失，用于处理类别不平衡的情况，并引入了不对称损失（Asymmetric Loss）的概念
#coverage略有提升，大概在一个点，其他值略有下降，没上面那个下降这么明显
class AsymmetricLoss(nn.Module):
    """Asymmetric loss"""
    #定义了类的初始化方法，初始化了不对称损失的参数 gamma_neg、gamma_pos、clip、eps、disable_torch_grad_focal_loss 和损失的缩减方式
    # reduction。super(AsymmetricLoss, self).__init__() 调用了父类 nn.Module 的初始化方法
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 reduction='mean'):

        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        #计算模型输出 x 的概率，通过 sigmoid 函数将输出转换为概率值 x_sigmoid，并计算正样本和负样本的概率值 xs_pos 和 xs_neg。

        # Asymmetric Clipping
        #对负样本的概率进行裁剪，防止负样本概率过小。
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        #算基本的交叉熵损失，考虑到正样本和负样本之间的不对称性。
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        #应用了不对称焦点损失的思想，对损失进行进一步的调整，考虑到正负样本之间的不平衡性
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        #根据指定的缩减方式 reduction 对损失进行相应的缩减操作，可以是均值或者求和。
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss.sum()
        return loss


#这段代码定义了一个名为 BinaryDiceLoss 的自定义损失函数，用于处理二分类任务中的 Dice 损失，即骰子损失函数
class BinaryDiceLoss(nn.Module):
    """Dice loss"""
    #定义了类的初始化方法，初始化了 Dice 损失的平滑参数 smooth、幂次参数 p 和损失的缩减方式 reduction。super(BinaryDiceLoss, self).__init__()
    # 调用了父类 nn.Module 的初始化方法。
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    #定义了类的前向传播方法，用于计算 Dice 损失。
    def forward(self, input, target):
        #断言输入的预测值和目标值的批大小相同，以确保它们的维度匹配，断言输入指的是通过 assert 语句来验证输入数据的条件是否满足的操作
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        #将输入经过 sigmoid 函数处理，使其处于 [0, 1] 范围内，然后将预测值和目标值重新调整为二维张量，以便后续计算。
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        #计算 Dice 损失的分子和分母部分，分别对预测值和目标值进行幂次计算，并计算它们的交集和并集。
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        #根据 Dice 损失的公式计算损失值
        loss = 1 - (2 * num) / den

        #根据指定的缩减方式 reduction 对损失进行相应的缩减操作，可以是均值、求和或者不进行缩减。如果指定了未知的缩减方式，则抛出异常。
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


#这是一个用于计算 DCS 损失的 PyTorch 模块。
class DCSLoss(nn.Module):
    """DCS loss"""
    #定义了类的初始化方法，初始化了 DCS 损失的平滑参数 smooth、幂次参数 p、权重参数 alpha 和损失的缩减方式 reduction。
    # super(DCSLoss, self).__init__() 调用了父类 nn.Module 的初始化方法。
    def __init__(self, smooth=1e-4, p=2, alpha=0.01, reduction='mean'):
        super(DCSLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        #断言输入的预测值 input 和目标值 target 的批大小相同，以确保它们的维度匹配。
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        #将输入经过 sigmoid 函数处理，使其处于 [0, 1] 范围内，然后将预测值和目标值重新调整为二维张量，以便后续计算。
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        #根据 DCS 损失的公式计算分子 num 和分母 den，其中应用了预测值的增强和目标值的权重
        pre_pos = predict*((1-predict)**self.alpha)
        num = torch.sum(torch.mul(pre_pos, target), dim=1)
        den = torch.sum(pre_pos.pow(self.p) + target.pow(self.p), dim=1)+self.smooth

        #根据 DCS 损失的公式计算损失值。
        loss = 1 - (2 * num + self.smooth) / den

        #根据指定的缩减方式 reduction 对损失进行相应的缩减操作，可以是均值、求和或者不进行缩减。如果指定了未知的缩减方式，则抛出异常。
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


#这是一个用于计算多标签 Focal-Dice 损失的 PyTorch 模块。
class FocalDiceLoss(nn.Module):
    """Multi-label focal-dice loss"""

    #定义了类的初始化方法，初始化了 Focal-Dice 损失的参数，包括正类别和负类别的幂次参数 p_pos 和 p_neg、正类别和负类别的截断阈值
    # clip_pos 和 clip_neg、正类别权重 pos_weight 和损失的缩减方式 reduction。super(FocalDiceLoss, self).__init__()
    # 调用了父类 nn.Module 的初始化方法。
    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        #断言输入的预测值 input 和目标值 target 的批大小相同，以确保它们的维度匹配。
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        #将输入经过 sigmoid 函数处理，使其处于 [0, 1] 范围内，然后将预测值和目标值重新调整为二维张量，以便后续计算。
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        #计算正类别的 Focal-Dice 损失的分子部分 num_pos 和分母部分 den_pos，并应用了截断操作和幂次运算。
        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        #计算负类别的 Focal-Dice 损失的分子部分 num_neg 和分母部分 den_neg，并应用了截断操作和幂次运算。
        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        #计算正类别和负类别的损失，并根据正类别的权重进行加权处理。
        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        #根据指定的缩减方式 reduction 对损失进行相应的缩减操作，可以是均值、求和或者不进行缩减。如果指定了未知的缩减方式，则抛出异常。
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



"""以下为添加的损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#论文doi：10.1186/s13321-022-00657-w
#coverage明显增加，但其他指标明显下降
class LDAM_loss(nn.Module):
    """LDAM Loss for handling class imbalance in binary classification."""

    #def __init__(self, max_m=0.5, class_weight="balanced"):
    def __init__(self, max_m=0.2, class_weight="balanced"):
        super(LDAM_loss, self).__init__()
        self.max_m = max_m
        self.class_weight = class_weight

    def forward(self, y_pred, y_true):
        # Assuming y_true are provided as {0, 1} labels
        if self.class_weight == "balanced":
            # Calculate class weights dynamically based on the batch
            majority = (y_true == 0).sum().float()
            minority = (y_true == 1).sum().float()

            # Avoid division by zero
            if majority == 0 or minority == 0:
                majority_weight = minority_weight = 1.0
            else:
                majority_weight = (1 / majority) * (y_true.numel() / 2)
                minority_weight = (1 / minority) * (y_true.numel() / 2)

            # Normalize weights
            total = majority_weight + minority_weight
            majority_weight /= total
            minority_weight /= total
        else:
            majority_weight = minority_weight = 1.0

        # Compute margins dynamically
        #默认设备用这个
        m_list = torch.tensor([1.0 / torch.sqrt(majority_weight), 1.0 / torch.sqrt(minority_weight)]).cuda()
        m_list = m_list * (self.max_m / torch.max(m_list))

        # Compute modified predictions
        batch_m = torch.where(y_true == 1, m_list[1], m_list[0])
        y_m = y_pred - batch_m

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(y_m, y_true, reduction='none')

        # Apply class weights
        loss = torch.where(y_true == 1, loss * minority_weight, loss * majority_weight)

        return loss.mean()



#召回率提升，其他下降
import torch
import torch.nn as nn
import torch.nn.functional as F

class APLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(APLLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, logits, targets):
        """Calculate the Asymmetric Polynomial Loss."""
        # Calculating probabilities
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1 - probs

        # Asymmetric Clipping
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1)

        # Calculate positive and negative polynomial terms
        pos_loss = targets * (torch.log(pos_probs.clamp(min=self.eps)) +
                              self.epsilon_pos * (1 - pos_probs.clamp(min=self.eps)) +
                              self.epsilon_pos_pow * 0.5 * torch.pow(1 - pos_probs.clamp(min=self.eps), 2))
        neg_loss = (1 - targets) * (torch.log(neg_probs.clamp(min=self.eps)) +
                                    self.epsilon_neg * (neg_probs.clamp(min=self.eps)))

        # Combine positive and negative loss
        loss = pos_loss + neg_loss

        # Apply focusing technique
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                pt = pos_probs * targets + neg_probs * (1 - targets)
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focusing_weight = torch.pow(1 - pt, one_sided_gamma)
                loss *= focusing_weight

        return -loss.sum()

#基本没影响
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class PartialSelectiveLoss(nn.Module):
    def __init__(self, clip=0.05, gamma_pos=0, gamma_neg=4, gamma_unann=2, alpha_pos=1.0, alpha_neg=1.0, alpha_unann=0.5, prior_path=None, partial_loss_mode=None, prior_threshold=None, likelihood_topk=None):
        super(PartialSelectiveLoss, self).__init__()
        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unann = gamma_unann
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unann = alpha_unann
        self.partial_loss_mode = partial_loss_mode
        self.prior_threshold = prior_threshold
        self.likelihood_topk = likelihood_topk

        self.targets_weights = None

        if prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully.")

    def forward(self, logits, targets):
        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()

        targets_weights = self.targets_weights
        targets_weights, xs_neg = self.edit_targets_partial_labels(targets, targets_weights, xs_neg, prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()

    def edit_targets_partial_labels(self, targets, targets_weights, xs_neg, prior_classes):
        if self.partial_loss_mode is None:
            targets_weights = 1.0
        elif self.partial_loss_mode == 'negative':
            targets_weights = 1.0
        elif self.partial_loss_mode == 'ignore':
            targets_weights = torch.ones(targets.shape, device=targets.device)
            targets_weights[targets == -1] = 0
        elif self.partial_loss_mode == 'ignore_normalize_classes':
            alpha_norm, beta_norm = 1, 1
            targets_weights = torch.ones(targets.shape, device=targets.device)
            n_annotated = 1 + torch.sum(targets != -1, axis=1)
            g_norm = alpha_norm * (1 / n_annotated) + beta_norm
            n_classes = targets_weights.shape[1]
            targets_weights *= g_norm.repeat([n_classes, 1]).T
            targets_weights[targets == -1] = 0
        elif self.partial_loss_mode == 'selective':
            if targets_weights is None or targets_weights.shape != targets.shape:
                targets_weights = torch.ones(targets.shape, device=targets.device)
            else:
                targets_weights[:] = 1.0
            num_top_k = self.likelihood_topk * targets_weights.shape[0]
            xs_neg_prob = xs_neg
            if prior_classes is not None:
                if self.prior_threshold:
                    idx_ignore = torch.where(prior_classes > self.prior_threshold)[0]
                    targets_weights[:, idx_ignore] = 0
                    targets_weights += (targets != -1).float()
                    targets_weights = targets_weights.bool()
            self.negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)
        return targets_weights, xs_neg

    @staticmethod
    def negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k):
        with torch.no_grad():
            targets_flatten = targets.flatten()
            cond_flatten = torch.where(targets_flatten == -1)[0]
            targets_weights_flatten = targets_weights.flatten()
            xs_neg_prob_flatten = xs_neg_prob.flatten()
            ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
            targets_weights_flatten[cond_flatten[ind_class_sort[:num_top_k]]] = 0

#准确率不提召回率也降
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLDAMLoss(nn.Module):
    #def __init__(self, gamma=2.0, max_m=0.2, class_weight="balanced"):
    def __init__(self, gamma=6.0, max_m=0.2, class_weight="balanced"):
        super(FocalLDAMLoss, self).__init__()
        self.gamma = gamma  # Focal loss 的聚焦参数
        self.max_m = max_m
        self.class_weight = class_weight

    def forward(self, y_pred, y_true):
        # 根据类别计算权重
        if self.class_weight == "balanced":
            majority = (y_true == 0).sum().float()
            minority = (y_true == 1).sum().float()

            if majority == 0 or minority == 0:
                majority_weight = minority_weight = 1.0
            else:
                majority_weight = (1 / majority) * (y_true.numel() / 2)
                minority_weight = (1 / minority) * (y_true.numel() / 2)

            total = majority_weight + minority_weight
            majority_weight /= total
            minority_weight /= total
        else:
            majority_weight = minority_weight = 1.0

        # 计算动态边界
        m_list = torch.tensor([1.0 / torch.sqrt(majority_weight), 1.0 / torch.sqrt(minority_weight)]).cuda()
        m_list = m_list * (self.max_m / torch.max(m_list))

        # 使用 LDAM 的修改预测
        batch_m = torch.where(y_true == 1, m_list[1], m_list[0])
        y_m = y_pred - batch_m

        # 计算二元交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(y_m, y_true, reduction='none')

        # 应用 Focal Loss 修改
        p_t = torch.sigmoid(y_m)
        p_t = torch.where(y_true == 1, p_t, 1 - p_t)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # 应用类别权重
        loss = torch.where(y_true == 1, loss * minority_weight, loss * majority_weight)

        return loss.mean()

#准确率不提，召回率也降
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 gamma=2.0, reduction='mean', disable_torch_grad_focal_loss=True):
        super(FocalAsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.gamma = gamma  # Focal loss 的聚焦参数
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        # 计算 Sigmoid 概率
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 不对称裁剪对负样本的概率
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 计算基础交叉熵损失
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # 不对称焦点处理
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y) + self.gamma * (1 - pt)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        # 如果需要禁用梯度计算
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                loss *= one_sided_w
        else:
            loss *= one_sided_w

        # 根据缩减方式进行最终损失计算
        if self.reduction == 'mean':
            return -loss.mean()
        else:
            return -loss.sum()

#为0
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super(DiceAsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 不对称损失的 Sigmoid 概率计算部分
        inputs_sigmoid = torch.sigmoid(inputs)
        xs_pos = inputs_sigmoid
        xs_neg = 1 - inputs_sigmoid

        # 对负样本概率进行裁剪
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 骰子损失的计算部分
        smooth = 1.0
        intersection = (inputs_sigmoid * targets).sum()
        dice_score = (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_score

        # 不对称损失的基本交叉熵计算
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        basic_loss = los_pos + los_neg

        # 应用不对称焦点调整
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        asymmetric_loss = basic_loss * one_sided_w

        # 结合骰子损失和不对称损失
        combined_loss = dice_loss + asymmetric_loss

        # 根据缩减方式进行最终损失计算
        if self.reduction == 'mean':
            return -combined_loss.mean()
        elif self.reduction == 'sum':
            return -combined_loss.sum()
        else:
            return -combined_loss

        return combined_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceAPLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(DiceAPLLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

        # Parameters for the polynomial expansion
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, logits, targets):
        # Calculating probabilities
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1 - probs

        # Asymmetric Clipping
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1)

        # Calculate positive and negative polynomial terms
        pos_loss = targets * (torch.log(pos_probs.clamp(min=self.eps)) +
                              self.epsilon_pos * (1 - pos_probs.clamp(min=self.eps)) +
                              self.epsilon_pos_pow * 0.5 * torch.pow(1 - pos_probs.clamp(min=self.eps), 2))
        neg_loss = (1 - targets) * (torch.log(neg_probs.clamp(min=self.eps)) +
                                    self.epsilon_neg * (neg_probs.clamp(min=self.eps)))

        apl_loss = pos_loss + neg_loss

        # Apply focusing technique
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                pt = pos_probs * targets + neg_probs * (1 - targets)
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focusing_weight = torch.pow(1 - pt, one_sided_gamma)
                apl_loss *= focusing_weight

        # Dice loss calculation
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_score

        # Combine APL Loss and Dice Loss
        combined_loss = apl_loss - dice_loss  # Here we assume that both losses are normalized similarly.

        return combined_loss.sum()  # Or mean, depending on what reduction you want

#PartialSelectiveLoss和APLLoss的结合，没什么用
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, clip=0.05, gamma_neg=4, gamma_pos=0, alpha_pos=1.0, alpha_neg=1.0, epsilon_pos=1.0, epsilon_neg=0.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=True):
        super(CombinedLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.clip = clip
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        # 计算 sigmoid 概率
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1 - probs

        # 不对称裁剪
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1)

        # 使用 APLLoss 的多项式项
        pos_polynomial_loss = targets * (torch.log(pos_probs.clamp(min=1e-8)) +
                                         self.epsilon_pos * (1 - pos_probs.clamp(min=1e-8)) +
                                         self.epsilon_pos_pow * 0.5 * torch.pow(1 - pos_probs.clamp(min=1e-8), 2))
        neg_polynomial_loss = (1 - targets) * (torch.log(neg_probs.clamp(min=1e-8)) +
                                               self.epsilon_neg * (neg_probs.clamp(min=1e-8)))
        loss = self.alpha_pos * pos_polynomial_loss + self.alpha_neg * neg_polynomial_loss

        # 结合 PartialSelectiveLoss 的不对称聚焦权重
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                pt = pos_probs * targets + neg_probs * (1 - targets)
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focusing_weight = torch.pow(1 - pt, one_sided_gamma)
                loss *= focusing_weight

        return -loss.sum()






import torch
import torch.nn as nn

class MarginalFocalDiceLoss(nn.Module):
    # 初始化函数，设置各个参数
    def __init__(self, max_m=0.2, p_pos=3, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.7, class_weight="balanced", reduction='mean'):
        super(MarginalFocalDiceLoss, self).__init__()
        self.max_m = max_m
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight
        self.class_weight = class_weight
        self.reduction = reduction

    # 前向传播计算损失
    def forward(self, y_pred, y_true):
        # 计算 LDAM 的类权重
        if self.class_weight == "balanced":
            majority = (y_true == 0).sum().float()
            minority = (y_true == 1).sum().float()

            if majority == 0 or minority == 0:
                majority_weight = minority_weight = 1.0
            else:
                majority_weight = (1 / majority) * (y_true.numel() / 2)
                minority_weight = (1 / minority) * (y_true.numel() / 2)

            total = majority_weight + minority_weight
            majority_weight /= total
            minority_weight /= total
        else:
            majority_weight = minority_weight = 1.0

        # 为 LDAM 计算带边际的修正预测
        m_list = torch.tensor([1.0 / torch.sqrt(majority_weight), 1.0 / torch.sqrt(minority_weight)]).cuda()
        m_list = m_list * (self.max_m / torch.max(m_list))
        batch_m = torch.where(y_true == 1, m_list[1], m_list[0])
        y_m = y_pred - batch_m

        # 对 FocalDiceLoss 进行 Sigmoid 操作并重塑形状
        predict = torch.sigmoid(y_m).contiguous().view(y_true.shape[0], -1)
        target = y_true.contiguous().view(y_true.shape[0], -1)

        # 计算 FocalDiceLoss 的各个部分
        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        # 确保损失值为正
        loss = torch.clamp(loss, min=0)

        # 根据 LDAM 的类权重计算加权损失
        weighted_loss = torch.where(y_true.view(-1, 1) == 1, loss * minority_weight, loss * majority_weight)

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        return loss




