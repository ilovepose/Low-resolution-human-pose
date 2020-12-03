# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# Modified by Chen Wang (wangchen199179@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.image import draw_dense_reg


logger = logging.getLogger(__name__)


def get_warpmatrix(theta,size_input,size_dst,size_target):
    '''

    :param theta: angle
    :param size_input:[w,h]
    :param size_dst: [w,h]
    :param size_target: [w,h]/200.0
    :return:
    '''
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2,3),dtype=np.float32)
    scale_x = size_target[0]/size_dst[0]
    scale_y = size_target[1]/size_dst[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = math.sin(theta) * scale_y
    matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * size_input[0]
    matrix[1, 0] = -math.sin(theta) * scale_x
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = 0.5*size_target[0]*math.sin(theta)-0.5*size_target[1]*math.cos(theta)+0.5*size_input[1]
    return matrix


def rotate_points(src_points, angle,c, dst_img_shape,size_target, do_clip=True):
    # src_points: (num_points, 2)
    # img_shape: [h, w, c]
    size_target = size_target * 200.0
    src_img_center = c
    scale_x = (dst_img_shape[0]-1.0)/size_target[0]
    scale_y = (dst_img_shape[1]-1.0)/size_target[1]
    radian = angle / 180.0 * math.pi
    radian_sin = -math.sin(radian)
    radian_cos = math.cos(radian)
    dst_points = np.zeros(src_points.shape, dtype=src_points.dtype)
    src_x = src_points[:, 0] - src_img_center[0]
    src_y = src_points[:, 1] - src_img_center[1]
    dst_points[:, 0] = radian_cos * src_x + radian_sin * src_y
    dst_points[:, 1] = -radian_sin * src_x + radian_cos * src_y
    dst_points[:, 0] += size_target[0]*0.5
    dst_points[:, 1] += size_target[1]*0.5
    dst_points[:, 0] *= scale_x
    dst_points[:, 1] *= scale_y
    if do_clip:
        dst_points[:, 0] = np.clip(dst_points[:, 0], 0, dst_img_shape[1] - 1)
        dst_points[:, 1] = np.clip(dst_points[:, 1], 0, dst_img_shape[0] - 1)
    return dst_points


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE  # gaussian or binary
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)  # 
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.feat_stride = (self.image_size-1.0) / (self.heatmap_size-1.0)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1
        self.locref_stdev = cfg.DATASET.LOCREF_STDEV
        self.mask_sigma = cfg.MODEL.MASK_SIGMA
        self.kpd = cfg.LOSS.KPD
        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if np.sum(joints_vis[:, 0]) > self.num_joints_half_body and \
                    np.random.rand() < self.prob_half_body:
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_warpmatrix(r, c*2.0, self.image_size-1.0, s)
        input = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        joints[:, 0:2] = rotate_points(joints[:, 0:2], r, c, self.image_size, s, False)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] == 0.0:
                joints[i, 0:2] = np.array([0.0, 0.0])

        target, target_offset, mask_01, mask_g, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_offset = torch.from_numpy(target_offset)
        mask_01 = torch.from_numpy(mask_01)
        mask_g = torch.from_numpy(mask_g)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_offset, mask_01, mask_g, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        """
        generate heat-map according to joints, joints_heat-map and joints_vis
        :param joints: joint location in input size
        :param joints_vis: visible(1) or not(0)
        :return:
        hm_hp: joint heat-map
        hm_hp_offset: joint offset
        target_weight: visible(1) or not(0)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        hm = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                      dtype=np.float32)
        om = np.zeros((2*self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                      dtype=np.float32)
        mask = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                        dtype=np.float32)
        mask_g = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                         dtype=np.float32)

        tmp_size = int(self.sigma * 3 + 0.5)

        for joint_id in range(self.num_joints):
            joints_hm = joints[joint_id, :2] / self.feat_stride

            target_weight[joint_id] = \
                 self.adjust_target_weight(joints_hm, target_weight[joint_id], tmp_size)

            if not target_weight[joint_id]:
                continue

            # Generate gaussian heatmap and offset matrix
            if target_weight[joint_id] > 0.5:
                hm[joint_id], om[2*joint_id:2*joint_id+2], mask[joint_id], mask_g[joint_id] = \
                    draw_dense_reg(self.heatmap_size,
                                   joints_hm,
                                   self.sigma,
                                   self.locref_stdev,
                                   self.mask_sigma,
                                   self.kpd)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        if self.target_type == 'binary':
            hm = mask

        return hm, om, mask, mask_g, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        mu_x = int(joint[0] + 0.5)
        mu_y = int(joint[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] <= 0 or br[1] <= 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight
