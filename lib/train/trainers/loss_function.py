import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import uv_volumes
import torch.nn.functional as F
import math
import numpy as np
import cv2
from lib.networks.perceptual_loss import Perceptual_loss

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = uv_volumes.Renderer(self.net)

        self.mse = lambda x, y : torch.mean((x - y) ** 2)
        self.entroy= torch.nn.CrossEntropyLoss()
        self.iLoss_weight = ExponentialAnnealingWeight(cfg.iLossMax, cfg.iLossMin, cfg.exp_k)
        self.uvLoss_weight = ExponentialAnnealingWeight(cfg.uvLossMax, cfg.uvLossMin, cfg.exp_k)
        self.vgg_loss = Perceptual_loss()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))

    def forward(self, batch, is_train=True):
        batch['is_train'] = is_train
        ret = self.renderer.render(batch)
        epoch = batch['epoch']
        mask_at_dp, mask_at_body, mask_at_bg = \
          batch['mask_at_dp'], batch['mask_at_body'], batch['mask_at_bg']

        rgb_pred, delta_rgb_pred = ret['rgb_map'][..., :3], ret['rgb_map'][..., 3:]
        rgb_gt = batch['rgb']
        i_map, uv_map = ret['iuv_map'][..., :24], ret['iuv_map'][..., 24:]
        i_gt, uv_gt = batch['iuv'][..., :24], batch['iuv'][..., 24:]
        scalar_stats = {}
        loss = 0

        iLoss_weight = self.iLoss_weight.getWeight(epoch)
        uvLoss_weight = self.uvLoss_weight.getWeight(epoch)

        if is_train:
          rgb_loss = self.mse(rgb_pred[0] + delta_rgb_pred[0], rgb_gt[mask_at_body])
          i_at_dp_loss = self.entroy(i_map[0], i_gt.max(-1)[1][mask_at_dp]) \
                          * iLoss_weight
          uv_at_dp_loss = self.mse(uv_map[0], uv_gt[mask_at_dp]) \
                          * uvLoss_weight

        else:
          rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
          rgb_padding[ret['T_last'] < cfg.T_threshold] = rgb_pred + delta_rgb_pred
          ret['rgb_map'] = rgb_padding
          rgb_loss = self.mse(rgb_padding, rgb_gt)

          i_at_dp_loss = self.entroy(i_map[mask_at_dp], i_gt.max(-1)[1][mask_at_dp]) \
                          * iLoss_weight
          uv_at_dp_loss = self.mse(uv_map[mask_at_dp], uv_gt[mask_at_dp]) \
                          * uvLoss_weight

        scalar_stats.update({'rgb_loss': rgb_loss, 
                              'i_at_dp_loss': i_at_dp_loss,
                              'uv_at_dp_loss': uv_at_dp_loss})

        loss += i_at_dp_loss + uv_at_dp_loss + rgb_loss

        if cfg.use_TL2Loss :
            TL2Loss_weight = cfg.TLoss_weight
            TL2_loss = 0
            if mask_at_bg.sum() != 0:
                TL2_loss += torch.mean((1. - ret['T_last'][mask_at_bg]) ** 2) * TL2Loss_weight
            if (mask_at_body).sum() != 0:
                TL2_loss += torch.mean((ret['T_last'][mask_at_body]) ** 2) * TL2Loss_weight

            scalar_stats['TL2_loss'] = TL2_loss
            loss += TL2_loss
 
        if cfg.use_vggLoss:
            mask_at_box = batch['mask_at_box'][0]
            H, W = batch['H'][0], batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            sh = mask_at_box.sum()
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            
            # crop rgb gt
            rgb_gt_box = torch.cuda.FloatTensor(sh, 3).fill_(0.)
            rgb_gt_box[mask_at_body[0]] = rgb_gt[mask_at_body]
            rgb_gt_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_gt_crop[mask_at_box] = rgb_gt_box

            rgb_gt_crop = rgb_gt_crop[y:y + h, x:x + w]
            rgb_gt_crop = rgb_gt_crop.permute(2,0,1)[None].detach()

            if is_train:
                # crop rgb pred at body
                rgb_padding_box = torch.cuda.FloatTensor(sh, 3).fill_(0.).detach()
                rgb_padding_box[mask_at_body[0]] = rgb_pred[0] + delta_rgb_pred[0]
                rgb_padding = torch.cuda.FloatTensor(H, W, 3).fill_(0.).detach()
                rgb_padding[mask_at_box] = rgb_padding_box
                rgb_pred_crop = rgb_padding[y:y + h, x:x + w]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1)[None]
            else:
                # crop rgb pred at box
                rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
                rgb_pred_crop[mask_at_box] = rgb_padding[0]
                rgb_pred_crop = rgb_pred_crop[y:y + h, x:x + w]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1)[None]

            vgg_loss = self.vgg_loss(rgb_pred_crop*2-1, rgb_gt_crop*2-1).squeeze() * cfg.vggLoss_weight
            scalar_stats.update({'vgg_loss': vgg_loss})                      
            loss += vgg_loss

        scalar_stats.update({'loss': loss})

        scalar_stats.update({'iLoss_weight': torch.tensor(iLoss_weight), 
                            'uvLoss_weight': torch.tensor(uvLoss_weight)})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


class CosineAnnealingWeight():
    def __init__(self, max, min, Tmax):
        super().__init__()
        self.max = max
        self.min = min
        self.Tmax = Tmax

    def getWeight(self, Tcur):
        return self.min + (self.max - self.min) * (1 + math.cos(math.pi * Tcur / self.Tmax)) / 2


class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))