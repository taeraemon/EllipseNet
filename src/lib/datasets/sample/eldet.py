from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import random
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

minimum, maximum = 0.6, 1.1

class ELDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    # coco box: upper-left corner + width and height
    # bounding box: upper-left corner and lower-right corner
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
      # When image is too large or too small
      minimum = 1.0 if min(img.shape[0], img.shape[1]) < 384 else 0.6
      maximum = 1.0 if min(img.shape[0], img.shape[1]) > 1024 else 1.1
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        io_scale = np.random.choice(np.arange(minimum, maximum, 0.1))
        s = s * io_scale
        w_border = self._get_border(384, img.shape[1])
        h_border = self._get_border(384, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :] 
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    if self.opt.gaussian_noise:
      do_transform = np.random.random() < 0.1
      if do_transform:
        noise_std = np.random.uniform(0, 0.1)
        noise = np.random.normal(0, noise_std, size=inp.shape)
        inp += noise
    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32) # heat map of center point
    wh = np.zeros((self.max_objs, 2), dtype=np.float32) # width and height
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32) # dense map of w and h
    reg = np.zeros((self.max_objs, 2), dtype=np.float32) # regression of center point due to the float->int error
    ind = np.zeros((self.max_objs), dtype=np.int64) # index of the center point
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32) # category specific w,h and mask
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    # parameters for ellipse regression
    l = np.zeros((self.max_objs, 1), dtype=np.float32) # length of the extended square
    a = np.zeros((self.max_objs, 1), dtype=np.float32) # length of the long axis
    b = np.zeros((self.max_objs, 1), dtype=np.float32) # length of the short axis
    ratio_al = np.zeros((self.max_objs, 1), dtype=np.float32) # ratio of the long axis to the length of square
    ratio_bl = np.zeros((self.max_objs, 1), dtype=np.float32) # ratio of the short axis to the length of square
    # ratio_ba = np.zeros((self.max_objs, 1), dtype=np.float32) # ratio of the short axis to the long axis
    theta = np.zeros((self.max_objs, 1), dtype=np.float32) # angle of the ellipse divided by pi (0, 1]
    sincos = np.zeros((self.max_objs, 2), dtype=np.float32) # encoded angle of the ellipse, sin and cos

    # Generate ground truth label
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      ellipse = ann['ellipse']
      ct = np.array(ellipse[:2], dtype=np.float32)
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        ct[0] = width - ct[0] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      ct = affine_transform(ct, trans_output)
      ct[0] = np.clip(ct[0], 0, output_w - 1)
      ct[1] = np.clip(ct[1], 0, output_h - 1)
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        # ct = np.array(
        #   [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

      # parameters for ellipse regression
      if ellipse[2] >= ellipse[3]:
        ell_a = ellipse[2]
        ell_b = ellipse[3]
        ell_alpha = ellipse[4]
      else:
        ell_a = ellipse[3]
        ell_b = ellipse[2]
        ell_alpha = ellipse[4] + 0.5 * math.pi
      a[k] = trans_output[0][0] * ell_a # length of the long axis
      b[k] = trans_output[0][0] * ell_b # length of the short axis
      l[k] = 2 * math.sqrt(a[k] * a[k] + b[k] * b[k]) # length of the extended square
      ratio_al[k] = 2 * a[k] / l[k] # ratio of the long axis to the length of square
      ratio_bl[k] = 2 * b[k] / l[k] # ratio of the short axis to the long axis
      # ratio_ba[k] = b[k] / a[k] # ratio of the short axis to the long axis
      angle = ell_alpha / math.pi # angle is in [-0.5, 0.5)
      angle = angle - 1 if angle > 0.5 else angle + 1 if angle < -0.5 else angle
      # angle = angle if angle >= 0 else 1 + angle
      theta[k] = -angle if flipped else angle
      sincos[k] = [np.sin(theta[k] * math.pi), np.cos(theta[k] * math.pi)] # sin in [-1, 1], cos in [0, 1]
    
    # ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'a': a, 'b': b, 'l': l, 'ratio_al': ratio_al, 'ratio_bl': ratio_bl, 'theta': theta, 'sincos': sincos}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret