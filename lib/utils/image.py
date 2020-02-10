# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Xingyi Zhou (zhouxy2017@gmail.com)
# Modified by Chen Wang (wangchen199179@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np
from config import cfg


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3  # guassian 半边长
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap  # have an error in condition
    size = 2 * tmp_size + 1  # 有size个
    x = np.arange(0, size, 1, np.float32)  # [0,...,2*tmp_size]
    y = x[:, np.newaxis]
    x0 = y0 = size // 2  # tmp_size
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])  #
    return heatmap


def draw_dense_reg(regmap, heatmap, center, offset, radius, stride, locref_stdev, is_offset=False, is_circle_mask=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    offset = offset[:, None, None]  # shape: [2, 1, 1]
    dim = offset.size
    reg = np.ones((dim, diameter, diameter), dtype=np.float32) * offset  # [2, d, d]

    if is_offset and dim == 2:
        delta = np.arange(radius, -radius-1, -1, dtype=np.int8)
        delta_x = delta * stride[0]
        delta_y = delta * stride[1]
        delta = np.zeros_like(reg)
        delta[0], delta[1] = np.meshgrid(delta_x, delta_y)
        reg += delta
        del delta, delta_x, delta_y

    if is_circle_mask:
        threshold = stride*radius
        circle_mask(reg, thres=np.sum(threshold**2))
        locref_stdev = threshold[:, np.newaxis, np.newaxis]

    reg /= locref_stdev

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)  # distances to edge
    top, bottom = min(y, radius), min(height - y, radius + 1)  # distances to edge

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        regmap[:, y - top:y + bottom, x - left:x + right] = (1 - idx) * masked_regmap + idx * masked_reg
        heatmap[y - top:y + bottom, x - left:x + right] = (1 - idx) * masked_heatmap + idx * masked_gaussian


def circle_mask(offset_mat, thres):
    dist_mat = np.linalg.norm(offset_mat, ord=2, axis=0, keepdims=True)
    mask = dist_mat <= thres
    offset_mat *= mask


# r = 3 * cfg.MODEL.SIGMA
# d = 2 * r + 1
# GaussianDistribution = gaussian2D((d, d), sigma=d / 6)
#
#
# def offset_encode(reg):
#     return (reg + 0.5 + GaussianDistribution) / 2.0
#
#
# def offset_decode(reg):
#     return reg * 2.0 - GaussianDistribution - 0.5
#
#
# if __name__ == '__main__':
#     regmap = np.zeros([2, 9, 9])
#     heatmap = np.zeros([9, 9]);heatmap[1,1]=1
#     center = np.array([4, 4])
#     value = np.array([-0.0, 0.0])
#     radius = 2
#     stride = np.array([1,1])
#     draw_dense_reg(regmap, heatmap, center, value, radius, stride, True, True)
#     print(regmap)
#     print(heatmap)
