import numpy as np
# import torch.nn as nn
import _init_paths
from testing import pickle_read as pr
from utils.image import gaussian_radius, draw_dense_reg
from core.evaluate import get_final_preds_offset

num_joints = 17
target_type = 'gaussian'
heatmap_size = [48, 64]
sigma = 2
feat_stride = np.array([4, 4])
OFF_OUT = True
use_target_weight = True
# criterion = nn.MSELoss(reduction='mean')
offset_weight = 1.0

d = pr('../debug.pickle')

target = d['target'].detach().cpu().numpy()
target_off = d['target_off'].detach().cpu().numpy()
target_weight = d['target_weight'].detach().cpu().numpy()
meta = d['meta']
output = d['output'].detach().cpu().numpy()
output_offset = d['output_offset'].detach().cpu().numpy()
stride = d['stride']
in_size = d['in_size']
outsize = d['out_size']
loss = d['loss'].detach().cpu().numpy()
c = meta['center'].detach().numpy()
scale = meta['scale'].detach().numpy()
joints = meta['joints'].detach().numpy()
joints_heatmap = meta['joints_heatmap'].detach().numpy()
joints_vis = meta['joints_vis'].detach().numpy()


def generate_target(joints, joints_heatmap, joints_vis):
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    if target_type == 'gaussian':
        hm_hp = np.zeros((num_joints,
                          heatmap_size[1],
                          heatmap_size[0]),
                         dtype=np.float32)
        hm_hp_offset = np.zeros((2 * num_joints,
                                 heatmap_size[1],
                                 heatmap_size[0]),
                                dtype=np.float32)

        tmp_size = sigma * 3
        if OFF_OUT:
            joint_hm_int = (joints_heatmap[:, :2]+0.5).astype(np.int8)
            hp_offset = joints_heatmap[:, :2] - joint_hm_int
            locref_stdev = 1.0
        else:
            joint_hm_int = (joints[:, :2]/feat_stride).astype(np.int8)
            hp_offset = joints[:, :2] - (joint_hm_int+0.5)*feat_stride  # offset at input scale
            locref_stdev = 7.281

        for joint_id in range(num_joints):
            #mu_x, mu_y = joint_hm_int[joint_id]
            #
            # Check that any part of the gaussian is in-bounds
            #ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  #
            #br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            #if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
            #        or br[0] < 0 or br[1] < 0:
            #    # If not, just return the image as is
            #    target_weight[joint_id] = 0
            #    continue
            target_weight[joint_id]=adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            # Generate gaussian
            if target_weight[joint_id] > 0.5:
                # generate offset matrix and heatmap
                draw_dense_reg(hm_hp_offset[2 * joint_id:2 * joint_id + 2], hm_hp[joint_id],
                               joint_hm_int[joint_id], hp_offset[joint_id, :2], tmp_size,
                               feat_stride, locref_stdev, is_offset=True, is_circle_mask=False)

    return hm_hp, hm_hp_offset, target_weight

def adjust_target_weight(joint, target_weight, tmp_size):
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
            or br[0] <= 0 or br[1] <= 0:
        # If not, just return the image as is
        target_weight = 0
    return target_weight


def loss(output, hm_hps, target, target_offset, target_weight, stride):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)  # every joints heatmaps
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)  # every target heatmaps
    offsets_pred = hm_hps.reshape((batch_size, 2*num_joints, -1)).split(1, 1)
    offsets_gt = target_offset.reshape((batch_size, 2*num_joints, -1)).split(1, 1)
    loss = 0
    offset_x_loss, offset_y_loss = 0, 0

    for idx in range(num_joints):
        sidx = 2*idx+1
        heatmap_pred = heatmaps_pred[idx].squeeze()  # [batch_size, h*w]
        heatmap_gt = heatmaps_gt[idx].squeeze()  # [batch_size, h*w]
        offset_x_pred = offsets_pred[2*idx].squeeze()  # [batch_size, h*w]
        offset_x_gt = offsets_gt[2*idx].squeeze()  # [batch_size, h*w]
        offset_y_pred = offsets_pred[sidx].squeeze()  # [batch_size, h*w]
        offset_y_gt = offsets_gt[sidx].squeeze()  # [batch_size, h*w]
        if use_target_weight:
            loss += 0.5 * criterion(
                heatmap_pred.mul(target_weight[:, idx]),
                heatmap_gt.mul(target_weight[:, idx])
            )
            offset_x_loss += 0.5 * criterion(
                offset_x_pred.mul(target_weight[:, idx]),
                offset_x_gt.mul(target_weight[:, idx])
            )
            offset_y_loss += 0.5 * criterion(
                offset_y_pred.mul(target_weight[:, idx]),
                offset_y_gt.mul(target_weight[:, idx])
            )
        else:
            loss += 0.5 * criterion(heatmap_pred, heatmap_gt)
            offset_x_loss += 0.5 * criterion(offset_x_pred, offset_x_gt)
            offset_y_loss += 0.5 * criterion(offset_y_pred, offset_y_gt)
        if OFF_OUT:
            offset_loss = offset_x_loss + offset_y_loss
            loss = loss + offset_weight * offset_loss
        else:
            offset_loss = offset_x_loss / stride[0] + offset_y_loss / stride[1]
            loss = loss + offset_weight * offset_loss

    return loss / num_joints

hm_hp, hm_hp_offset, target_weight =  generate_target(joints[0], joints_heatmap[0], joints_vis[0])
# total_loss = loss(output, hm_hps, target, target_off, target_weight, np.array([4 ,4]))
# total_loss = total_loss.detach().cpu().numpy()


#preds, maxvals = get_final_preds_offset(OFF_OUT, output, output_offset,
#                                        c, scale, stride, in_size, outsize)
print('joints_vis: ', joints_vis[0,1])
print('joints； ', joints[0,1])
print('target shape: ', hm_hp.shape)
print('my target: ', hm_hp[0, 4:7, 10:13])
print('my offset[0]: ', hm_hp_offset[0, 4:7, 10:13])
print("target[0,0]:  ", target[1,0])
print('target_off[0,1]: ', target_off[0,0,4:7, 10:13])
print('over!')

