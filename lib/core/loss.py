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


class JointsOffsetLoss(nn.Module):
    def __init__(self, use_target_weight, offset_weight,
                 pixel_hm, pred_mask, gt_mask,
                 alpha, beta, gama, smooth_l1, bce):
        super(JointsOffsetLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.offset_weight = offset_weight
        self.use_pixel_hm = pixel_hm
        self.use_pred_mask = pred_mask
        self.use_gt_mask = gt_mask
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_offset = nn.SmoothL1Loss(reduction='mean') if smooth_l1 else nn.L1Loss(reduction='mean')


    def forward(self, output, hm_hps, target, target_offset,
                mask_01, mask_g, target_weight):
        """
        calculate loss
        :param output: [batch, joints, height, width]
        :param hm_hps: [batch, 2*joints, height, width]
        :param target: [batch, joints, height, width]
        :param target_offset: [batch, 2*joints, height, width]
        :param mask_01: [batch, joints, height, width]
        :param mask_g: [batch, joints, height, width]
        :param target_weight: [batch, joints, 1]
        :return: loss=joint_loss+weight*offset_loss
        """
        batch_size, num_joints, _, _ = output.shape

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        offsets_pred = hm_hps.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)
        offsets_gt = target_offset.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)

        # focal weight on heat map
        if self.use_pixel_hm:
            mask_hm = self._focal_pixel_weight(output, target)
            mask_hm = mask_hm.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        else:
            mask_hm = [torch.Tensor([1.0]).cuda() for _ in range(num_joints)]

        # focal mask on offset map
        if self.use_pred_mask:
            mask_om = output.detach() * mask_01
            mask_om_normalize = self._mask_renormalize(mask_om)
            mask_om = mask_om_normalize * mask_01
        elif self.use_gt_mask:
            mask_om = mask_g
        else:
            mask_om = mask_01  # 0-1 mask or gaussian mask
        mask_om = mask_om.reshape((batch_size, num_joints, -1)).split(1, dim=1)

        del batch_size, _

        joint_l2_loss, offset_loss = 0.0, 0.0

        for idx in range(num_joints):
            offset_pred = offsets_pred[idx] * mask_om[idx]  # [batch_size, 2, h*w]
            offset_gt = offsets_gt[idx] * mask_om[idx]      # [batch_size, 2, h*w]
            heatmap_pred = heatmaps_pred[idx].squeeze() * mask_hm[idx].squeeze()  # [batch_size, h*w]
            heatmap_gt = heatmaps_gt[idx].squeeze() * mask_hm[idx].squeeze()      # [batch_size, h*w]

            if self.use_target_weight:
                joint_l2_loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                offset_loss += self.criterion_offset(
                    offset_pred.mul(target_weight[:, idx].unsqueeze(2)),
                    offset_gt.mul(target_weight[:, idx, None])
                )
            else:
                joint_l2_loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
                offset_loss += self.criterion_offset(offset_pred, offset_gt)

        loss = joint_l2_loss + self.offset_weight * offset_loss

        return loss / num_joints

    def _focal_pixel_weight(self, pred, gt):
        """
        Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        :param pred: (batch x joints x h x w)
        :param gt: (batch x joints x h x w)
        :return: focal_pixel_weight rescaled to [0, 1]
        """
        zeros = torch.zeros_like(pred)
        ase=torch.abs(pred-gt)
        exp_ase=torch.exp(4*ase)
        focal_pixel_hm = torch.where(torch.ge(gt, 0.3), exp_ase, zeros+self.beta)  # zeros+self.alpha
        # focal_pixel_hm = torch.where(torch.ge(gt, 0.01), pred - self.alpha, 1 - pred - self.beta)
        # focal_pixel_hm = torch.abs(1. - focal_pixel_hm)  # [b,c,h, w]
        return focal_pixel_hm

    def _focal_softmax(self, output, target, thres=0.2):
        mask = target.gt(thres).float()  # [batch, height*width]
        output_softmax = self._masked_softmax(output, mask)
        return output_softmax, mask

    @staticmethod
    def _masked_softmax(inp, mask):
        """
        softmax with mask
        :param inp: predicted heat map, [batch, height*width]
        :param mask: 0-1 matrix, [batch, height*width]
        :return:
        """
        inp_exp = torch.exp(inp)
        inp_exp_masked = inp_exp * mask
        inp_exp_sum = torch.sum(inp_exp_masked, dim=-1, keepdim=True)
        return (inp_exp_masked + 1e-5) / (inp_exp_sum + 1e-5)

    def _mask_renormalize(self, mask):
        """
        rescale value to [0, 1]
        :param mask: [B, C, H, W]
        :return:
        """
        max_val, _ = self._find_extremum(mask, maxmin='max')  # [B, C]
        min_val, _ = self._find_extremum(mask, maxmin='min')  # [B, C]
        max_dist = max_val - min_val
        del max_val
        return (mask - min_val[..., None, None]) / (max_dist[..., None, None] + 1e-5)

    @staticmethod
    def _find_extremum(matrix, maxmin, loc=False):
        """
        :param matrix:
        :param loc:
        :return:
        max_val: [B, C]
        max_ind: [B, C, 2] which are (xs, ys)
        """
        if maxmin == 'max':
            val, ind_x = torch.max(matrix,  dim=3)  # [B, C, H]
            val, ind_y = torch.max(val,    dim=2)  # [B, C]
        elif maxmin == 'min':
            val, ind_x = torch.min(matrix, dim=3)  # [B, C, H]
            val, ind_y = torch.min(val,   dim=2)  # [B, C]
        else:
            raise NameError

        ind = None
        if loc:
            batch_size, num_channel, _, _ = matrix.shape
            ind = torch.empty((batch_size, num_channel, 2))  # [B, C, 2]
            ind[:, :, 1] = ind_y
            indx = torch.gather(ind_x, dim=2, index=ind_y.unsqueeze(dim=2))
            ind[:, :, 0] = indx.squeeze(dim=2)
        return val, ind


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target, mask=None):
        """
        calculate local-CE loss with output and target
        :param output: [batch, width*height]，最小值不能为0
        :param target: [batch, width*height]
        :return:
        """
        loss = -target * torch.log(output+1e-5)
        if mask is not None:
            loss *= mask
        return torch.mean(loss)
