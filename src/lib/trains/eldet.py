from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, L1LossRegression, RegL1Loss4Angle, IoULoss
from models.decode import eldet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import eldet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

from torch.utils.tensorboard import SummaryWriter


class EldetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(EldetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.crit_l1_loss = torch.nn.L1Loss(reduction='elementwise_mean')
    self.crit_mesloss = torch.nn.MSELoss()
    self.crit_angle = RegL1Loss4Angle(use_smooth_l1=(opt.reg_loss=='sl1'))
    self.iou_loss = IoULoss() 
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss, ellipse_loss, iou_loss = 0, 0, 0, 0, 0
    ratio_al_loss, ratio_bl_loss, theta_loss, sincos_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm']) # default

      # Use ground truth label for validation?
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(
          output['l'], batch['reg_mask'],
          batch['ind'], batch['l']) / opt.num_stacks
        # if opt.dense_wh:
        #   mask_weight = batch['dense_wh_mask'].sum() + 1e-4
        #   wh_loss += (
        #     self.crit_wh(output['wh'] * batch['dense_wh_mask'],
        #     batch['dense_wh'] * batch['dense_wh_mask']) / 
        #     mask_weight) / opt.num_stacks
        # elif opt.cat_spec_wh:
        #   wh_loss += self.crit_wh(
        #     output['wh'], batch['cat_spec_mask'],
        #     batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        # else:
        #   wh_loss += self.crit_reg(
        #     output['wh'], batch['reg_mask'],
        #     batch['ind'], batch['wh']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks

      # ellipse loss
      # if opt.ellipse_weight > 0:
        # ellipse_loss += self.crit_l1_loss(output['ratio_al'], batch['ratio_al']) / opt.num_stacks # ratio_al
        # ellipse_loss += self.crit_l1_loss(output['ratio_ba'], batch['ratio_ba']) / opt.num_stacks # ratio_ba
        # ellipse_loss += self.crit_l1_loss(output['theta'], batch['theta']) / opt.num_stacks # theta
      ratio_al_loss = self.crit_reg(output['ratio_al'], batch['reg_mask'],
                            batch['ind'], batch['ratio_al']) / opt.num_stacks # ratio_al
      ratio_bl_loss = self.crit_reg(output['ratio_bl'], batch['reg_mask'],
                            batch['ind'], batch['ratio_bl']) / opt.num_stacks # ratio_bl
      theta_loss = opt.theta_weight * self.crit_angle(output['theta'], batch['reg_mask'],
                            batch['ind'], batch['theta']) / opt.num_stacks # theta (where output['theta'] is (1, 1, 128, 128))
      sincos_loss = opt.sincos_weight * self.crit_reg(output['sincos'], batch['reg_mask'],
                            batch['ind'], batch['sincos']) / opt.num_stacks # sincos
      ellipse_loss = ellipse_loss + ratio_al_loss + ratio_bl_loss + theta_loss + sincos_loss

      # if opt.ellipse_reg_weight > 0:
      # cons = output['ratio_al'] ** 2 * (1 + output['ratio_ba'] ** 2)
      cons = output['ratio_al'] ** 2 + output['ratio_bl'] ** 2
      ellipse_loss += opt.ellipse_reg_weight * self.crit_mesloss(cons, torch.tensor(1.0).to(opt.device))

      # rotated iou loss
      iou_loss, iou = self.iou_loss(output, batch)
      # if not opt.iou_weight > 0:
      #   iou_loss = 0


        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.iou_weight * iou_loss + \
           opt.off_weight * off_loss + opt.ellipse_weight * ellipse_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'l_loss': wh_loss, 
                  'off_loss': off_loss, 'ellipse_loss': ellipse_loss,
                  'ratio_al_loss': ratio_al_loss, 'ratio_bl_loss': ratio_bl_loss, 'theta_loss': theta_loss, 'sincos_loss': sincos_loss, 'iou_loss': iou_loss, 'iou': iou}
    return loss, loss_stats

class EldetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(EldetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'l_loss', 'off_loss', 'ellipse_loss', 'ratio_al_loss', 
                    'ratio_bl_loss', 'theta_loss', 'sincos_loss', 'iou_loss', 'iou']
    loss = EldetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = eldet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = eldet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = eldet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]