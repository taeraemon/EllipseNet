from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys

lib_path = os.path.join('../src/lib')
if not lib_path in sys.path:
    sys.path.append(lib_path)
    
lib_path = os.path.join('../src/')
if not lib_path in sys.path:
    sys.path.append(lib_path)
    
lib_path = '../../Rotated_IoU/'
if not lib_path in sys.path:
    sys.path.append(lib_path)
    
import json
import cv2
from glob import glob
import math
import numpy as np
import time
from progress.bar import Bar
import torch

import matplotlib.pyplot as plt
from PIL import Image

from external.nms import soft_nms
from fhd_opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from models.losses import _transpose_and_gather_feat
from models.Rotated_IoU.oriented_iou_loss import cal_diou, cal_giou, cal_iou

# Calculate Metrics: Dice and difference of angles
# input 2 ellipse ell1, ell2 in [cx, cy, a, b, angle], image shape [h, w], num_classes

def compute_dice(pred, label):
    
    ints = np.sum((pred == 1) * (label == 1))
    sums = np.sum(pred == 1) + np.sum(label == 1) + 1e-5
    
    return (2.0 * ints) / sums

def compute_iou(pred, label):
    
    inter = np.sum((pred == 1) * (label == 1))
    union = np.sum((pred == 1) + (label == 1)) + 1e-5
    
    return inter / union

def compute_rotated_bbox_iou(ell1, ell2):
    cx1, cy1, a1, b1, angle1 = ell1
    cx2, cy2, a2, b2, angle2 = ell2
    theta1 = angle1 / 180.0 * math.pi
    theta2 = angle2 / 180.0 * math.pi
    bbox1 = torch.tensor([[[cx1, cy1, 2 * a1, 2 * b1, theta1]]], dtype=torch.float).cuda()
    bbox2 = torch.tensor([[[cx2, cy2, 2 * a2, 2 * b2, theta2]]], dtype=torch.float).cuda()
    return cal_diou(bbox1, bbox2) # iou_loss, iou

def compute_single_class_metric(ell1, ell2, shape):
    cx1, cy1, a1, b1, angle1 = ell1
    cx2, cy2, a2, b2, angle2 = ell2
    h, w = shape
    pred = np.zeros([h, w], np.uint8)
    pred = cv2.ellipse(pred, (cx1, cy1), (int(a1), int(b1)), int(angle1), 0.0, 360.0, (1), thickness=-1)
    label = np.zeros([h, w], np.uint8)
    label = cv2.ellipse(label, (cx2, cy2), (int(a2), int(b2)), int(angle2), 0.0, 360.0, (1), thickness=-1)
    angle_error = abs(angle1 - angle2)
    angle_error = angle_error if angle_error <= 90 else (180 - angle_error)
    bbox_iou_loss, bbox_iou = compute_rotated_bbox_iou(ell1, ell2)
    return compute_dice(pred, label), compute_iou(pred, label), bbox_iou.item(), angle_error

if __name__ == '__main__':
    opt = opts().parse()
    # opt.data_dir = '/data/cc/Data/CHD/fake_data/'
    # opt.load_model = '/data/cc/workspace/Repository/CenterNet/exp/eldet/fake_0214_eldet_fhd_dla_iouw5_nocrop_896x608/model_last.pth'
    # opt.load_model = '/data/cc/workspace/Repository/CenterNet/exp/eldet/fake_0209_eldet_fhd_dla_rl_nocrop_896x608/model_best.pth'
    # opt.data_dir = '/data/cc/Data/CHD/detection/'
    # opt.load_model = '/data/cc/workspace/Repository/CenterNet/exp/eldet/0218_exp4_iou5_cond_896x608/model_last.pth'
    # opt.num_classes = 2
    # opt.dataset = 'coco_fhd'
    # opt.resume = True
    # opt.gpus_str = '5'

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    exp_name = opt.load_model.split('/')[-2]
    save_pred_folder = os.path.join('../exp/prediction/', exp_name)
    if not os.path.exists(save_pred_folder):
        os.makedirs(save_pred_folder)
        
    print(opt)

    Detector = detector_factory[opt.task]
    split = 'val'
    print(opt.mean, opt.std)
    dataset = Dataset(opt, split)
    detector = Detector(opt)


    angle_errors = []
    dices = []
    ious = []
    bbox_ious = []
    ratios_pred = []
    ratios_gt = []
    ctr_precisions = []

    for ind in range(len(dataset)):
        # Load Image
        
        cls_dice = []
        cls_angle_error = []
        cls_iou = []
        cls_bbox_iou = []
        diameter_gt = [1, 1]
        diameter_pred = [1, 1]
        
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        shape = image.shape[:2]

        # Load annotations
        ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
        anns = dataset.coco.loadAnns(ids=ann_ids)

        # Inference
        color_maps = [(0, 0, 255), (0, 255, 255), (0, 128, 0), (0, 255, 0)]
        ret = detector.run(img_path)

        for class_idx in range(opt.num_classes):
            try:
                class_id = class_idx + 1
                pt1 = (int(ret['results'][class_id][0, 0]), int(ret['results'][class_id][0, 1]))
                pt2 = (int(ret['results'][class_id][0, 2]), int(ret['results'][class_id][0, 3]))
                cx, cy = int(ret['results'][class_id][0, 4]), int(ret['results'][class_id][0, 5])
                l = int(ret['results'][class_id][0, 6]) # !attention here!
                ratio_al = ret['results'][class_id][0, 7]
                ratio_bl = ret['results'][class_id][0, 8]
                a = ratio_al * l / 2
                b = ratio_bl * l / 2
                theta = ret['results'][class_id][0, 9]
                angle = theta * 180
                print("Predict:     ", pt1, pt2, (cx, cy), l, ratio_al, ratio_bl, "({})'".format(theta), int(a), int(b), int(angle))

                # Annotations
                ann = anns[class_idx]
                ellipse = ann['ellipse']
                cx_gt = ellipse[0]
                cy_gt = ellipse[1]
                if ellipse[2] >= ellipse[3]:
                    a_gt = ellipse[2]
                    b_gt = ellipse[3]
                    theta_gt = ellipse[4]
                else:
                    a_gt = ellipse[3]
                    b_gt = ellipse[2]
                    theta_gt = ellipse[4] + 0.5 * math.pi
                theta_gt = theta_gt / math.pi
                theta_gt = theta_gt - 1 if theta_gt > 0.5 else theta_gt + 1 if theta_gt < -0.5 else theta_gt
                l_gt = 2 * math.sqrt(a_gt ** 2 + b_gt ** 2)
                pt1 = (int(cx_gt - l_gt / 2), int(cy_gt - l_gt / 2))
                pt2 = (int(cx_gt + l_gt / 2), int(cy_gt + l_gt / 2))
                ratio_al, ratio_bl = 2 * a_gt / l_gt, 2 * b_gt / l_gt
                angle_gt = theta_gt * 180
                print("Ground Truth:", pt1, pt2, (cx_gt, cy_gt), l_gt, ratio_al, ratio_bl, "({})'".format(theta_gt), int(a_gt), int(b_gt), int(angle_gt))
                img = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=(125, 0, 0), thickness=1)
                img = cv2.ellipse(img, (cx_gt, cy_gt), (int(a_gt), int(b_gt)), angle_gt, 0.0, 360.0, (255, 0, 0), thickness=2)
            #     img = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color_maps[class_idx * 2 + 1], thickness=2)
            #     img = cv2.ellipse(image, (cx_gt, cy_gt), (int(a_gt), int(b_gt)), angle_gt, 0.0, 360.0, color_maps[class_idx * 2 + 1], thickness=2)


                pt1 = (int(cx - l / 2), int(cy - l / 2))
                pt2 = (int(cx + l / 2), int(cy + l / 2))
                img = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color_maps[class_idx * 2], thickness=2)
                img = cv2.ellipse(img, (cx, cy), (int(a), int(b)), angle, 0.0, 360.0, color_maps[class_idx * 2 + 1], thickness=3)

                pred_ell = [cx, cy, a, b, angle]
                gt_ell = [cx_gt, cy_gt, a_gt, b_gt, angle_gt]
                error = compute_single_class_metric(pred_ell, gt_ell, shape)
                print("dice: {}  iou:{}  bbox_iou: {}  angle_error: {}".format(error[0], error[1], error[2], error[3]))
                diameter_pred[class_idx] = b
                diameter_gt[class_idx] = b_gt
                cls_dice.append(error[0])
                cls_iou.append(error[1])
                cls_bbox_iou.append(error[2])
                cls_angle_error.append(error[3])
                
            except Exception as e:
                print(e)
                cls_dice.append(0)
                cls_iou.append(0)
                cls_bbox_iou.append(0)
                cls_angle_error.append(90)
                pass

        if opt.save_image:
            img = Image.fromarray(image)
            img.save(os.path.join(save_pred_folder, img_path.split('/')[-1]))
            print("Saving", img_path, "to", os.path.join(save_pred_folder, img_path.split('/')[-1]))
        dices.append(cls_dice)
        ious.append(cls_iou)
        bbox_ious.append(cls_bbox_iou)
        angle_errors.append(cls_angle_error)
        try:
            ratio_p, ratio_g = float(diameter_pred[1]) / diameter_pred[0], float(diameter_gt[1]) / diameter_gt[0]
            ctr_precision = 1 - abs(ratio_p - ratio_g) / ratio_g
        except Exception as e:
            ctr_precision = 0
            ratio_p = 0
        print("predicted ratio: {}, truth ratio: {}, error: {}, precision: {}".format(ratio_p, ratio_g, abs(ratio_p - ratio_g), ctr_precision))
        ratios_pred.append(ratio_p)
        ratios_gt.append(ratio_g)
        ctr_precisions.append(ctr_precision)
        # break

    print("Evaluation Done!")
    print("Mean Dice: ", np.mean(dices, axis=0))
    print("Mean IoU: ", np.mean(ious, axis=0))
    print("Mean BBox IoU: ", np.mean(bbox_ious, axis=0))
    print("Mean Angle_err: ", np.mean(angle_errors, axis=0))
    print("Mean Ratio_err: ", np.mean(np.abs(np.array(ratios_pred) - np.array(ratios_gt))))
    print("Mean CTR precision: ", np.mean(ctr_precisions))
