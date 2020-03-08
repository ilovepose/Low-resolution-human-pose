# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Chen Wang (wangchen199179@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def _neg_loss(pred, gt):
    """
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    :param pred: (batch x joints x h x w)
    :param gt: (batch x joints x h x w)
    """
    alpha = 0.1
    beta  = 0.02
    thre  = 0.01

    #pos_inds = gt.gt(thre).float()  # gt > thre
    #neg_inds = gt.le(thre).float()  # gt <= 1

    #focal_weights = torch.pow(1-pred+alpha, 2)*pos_inds + \
    #                torch.pow(pred+beta, 2)*neg_inds

    #st = torch.where(torch.ge(gt, 0.01), pred-alpha, 1-pred-beta)
    #focal_weights = torch.abs(1. - st)
    zeros = torch.zeros_like(pred)
    st = torch.where(torch.ge(gt, 0.01), zeros+alpha, zeros+beta)
    focal_weights = torch.abs(gt - pred)+st

    focal_l2 = torch.pow(pred - gt, 2) * focal_weights.detach()
    loss = focal_l2.mean()
    return loss


class FocalL2Loss(nn.Module):
    """nn.Module warpper for focal loss"""
    def __init__(self):
        super(FocalL2Loss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class JointsOffsetLoss(nn.Module):
    def __init__(self, use_target_weight, offset_weight, smooth_l1):
        super(JointsOffsetLoss, self).__init__()
        self.use_target_weight=use_target_weight
        self.offset_weight = offset_weight
        self.criterion = FocalL2Loss()#  nn.MSELoss(reduction='mean')
        self.criterion_offset = nn.SmoothL1Loss(reduction='mean') if smooth_l1 else nn.L1Loss(reduction='mean')

    def forward(self, output, hm_hps, target, target_offset, target_weight, epoch):
        """
        calculate loss
        :param output: [batch, joints, height, width]
        :param hm_hps: [batch, 2*joints, height, width]
        :param target: [batch, joints, height, width]
        :param target_offset: [batch, 2*joints, height, width]
        :param target_weight: [batch, joints]
        :param stride: downsample ratio
        :return: loss
        """
        # if epoch==250: self.criterion = FocalL2Loss()
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        offsets_pred = hm_hps.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)
        offsets_gt = target_offset.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)

        joint_loss, offset_loss = 0, 0

        for idx in range(num_joints):
            offset_pred = offsets_pred[idx] * heatmaps_gt[idx]  # [batch_size, 2, h*w]
            offset_gt = offsets_gt[idx] * heatmaps_gt[idx]  # [batch_size, 2, h*w]
            heatmap_pred = heatmaps_pred[idx].squeeze()  # [batch_size, h*w]
            heatmap_gt = heatmaps_gt[idx].squeeze()  # [batch_size, h*w]
            if self.use_target_weight:
                joint_loss  += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                offset_loss += 0.5 * self.criterion_offset(
                    offset_pred.mul(target_weight[:, idx].unsqueeze(2)),
                    offset_gt.mul(target_weight[:, idx].unsqueeze(2))
                )
            else:
                joint_loss  += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
                offset_loss += 0.5 * self.criterion_offset(offset_pred, offset_gt)

        loss = joint_loss + self.offset_weight * offset_loss

        return loss / num_joints
