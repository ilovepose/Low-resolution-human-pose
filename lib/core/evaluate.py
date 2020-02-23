# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds
from utils.transforms import transform_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 0 or target[n, c, 1] > 0:  # 为什么大于1 和 and ???
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy2(output, hm_hps, target, target_hm_hps, off_out, stride, locref_stdev, thr=0.5):
    num_joints = output.shape[1]
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((output.shape[0], 2)) * np.array([h, w]) / 10

    int_pred, maxvals = get_max_preds(output)  # [batch, joint, 2]
    int_target, _ = get_max_preds(target)
    offset_pred = get_offset(hm_hps, int_pred)  # [batch, joints, 2]
    offset_target = get_offset(target_hm_hps, int_target)
    if off_out:
        pred   = int_pred + offset_pred*locref_stdev
        target = int_target + offset_target*locref_stdev
        offset_norm = np.ones((pred.shape[0], 2))
        final_norm = norm
    else:
        pred   = (int_pred+0.5)*stride + offset_pred*locref_stdev
        target = (int_target+0.5)*stride + offset_target*locref_stdev
        offset_norm = np.ones((pred.shape[0], 2))*stride
        final_norm = norm * stride

    int_dists = calc_dists(int_pred, int_target, norm)
    offset_dists = dist(offset_pred, offset_target, int_target, offset_norm)
    dists = dist(pred, target, int_target, final_norm)

    acc = np.zeros((num_joints + 1, 3))
    avg_acc = np.zeros(3)
    cnt = np.zeros(3)

    for i in range(num_joints):
        acc[i + 1, 0] = dist_acc(dists[i], thr)
        acc[i + 1, 1] = dist_acc(int_dists[i], thr)
        acc[i + 1, 2] = dist_acc(offset_dists[i], thr)
        if acc[i + 1, 0] > 0:
            avg_acc[0] += acc[i + 1, 0]
            cnt[0] += 1
        if acc[i + 1, 1] > 0:
            avg_acc[1] += acc[i + 1, 1]
            cnt[1] += 1
        if acc[i + 1, 2] > 0:
            avg_acc[2] += acc[i + 1, 2]
            cnt[2] += 1

    for j in range(3):
        if cnt[j] != 0:
            avg_acc[j] = avg_acc[j] / cnt[j]
            acc[0, j] = avg_acc[j]
    return acc, avg_acc, cnt, pred


def get_offset(hm_hps, idx):
    """

    :param hm_hps:
    :param idx:
    :return: offset array [batch, joints, 2]
    """
    n_b, n_j, _ = idx.shape
    offset = np.empty_like(idx)
    for i in range(n_b):
        for j in range(n_j):
            offset[i, j, 0] = hm_hps[i, 2*j, int(idx[i, j, 1]), int(idx[i, j, 0])]
            offset[i, j, 1] = hm_hps[i, 2*j+1, int(idx[i, j, 1]), int(idx[i, j, 0])]
    return offset


def dist(offset_preds, offset_targets, target, norm):
    """
    rewrite calc_dist()
    :param offset_preds:
    :param offset_targets:
    :return:
    """
    mask = np.greater(target, 0).sum(axis=2)
    mask = np.greater(mask, 1)
    if norm.ndim != offset_targets.ndim or norm.ndim != offset_preds.ndim:
        norm = np.expand_dims(norm, axis=1)
    norm_offset_preds = offset_preds / norm
    norm_offset_targets = offset_targets / norm
    tmp = norm_offset_preds - norm_offset_targets
    dists = np.linalg.norm(tmp, axis=2)
    non_dists = -1 * np.ones_like(dists)
    dists = dists * mask + non_dists * (1-mask)
    return dists.transpose(1, 0)  # [joints, batch]


def get_final_preds_offset(off_out, hms, hm_offs, locref_stdev,
                           center, scale, stride, in_size, out_size):
    coords, maxvals = get_max_preds(hms)
    offset_pred = get_offset(hm_offs, coords)

    if off_out:
        coords = coords + offset_pred*locref_stdev
        size = out_size
    else:
        coords = (coords+0.5) * stride + offset_pred*locref_stdev
        size = in_size

    preds = coords.copy()

    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], size
        )
    return preds, maxvals


# if __name__ == '__main__':
#     import os.path as osp
#     import sys
#     this_dir = osp.dirname(__file__)
#     lib_path = osp.join(this_dir, '..')
#     sys.path.insert(0, lib_path) if lib_path not in sys.path else 0
#
#     from core.inference import get_max_preds
#
#     num_joints=2
#     output = np.zeros([1, num_joints, 5, 5]); output[0, 0, 2, 2] = 1; output[0, 1, 1, 1] = 1
#     hm_hps = np.zeros([1, 2*num_joints, 5, 5]); hm_hps[0,:2,2,2]=np.array([-0.3,0.4]); hm_hps[0,2:,1,1]=np.array([-0.4,0.4])
#     target = np.zeros([1, num_joints, 5, 5]); target[0, 0, 2, 2] = 1; target[0,1,1,1]=1
#     target_hm_hps = np.zeros([1, 2*num_joints, 5, 5]); target_hm_hps[0, :2, 2, 2] = np.array([-0.2, 0.4]);target_hm_hps[0,2:,1,1]=np.array([-0.2,0.2])
#     acc, avg_acc, cnt, pred = accuracy2(output, hm_hps, target, target_hm_hps)
#     print(acc)
