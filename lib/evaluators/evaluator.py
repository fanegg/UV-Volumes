import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
import os
import cv2
import torch
# from lpips import lpips
import lpips

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.frame_ls = []
        if cfg.use_lpips:
            self.lpips = []
            self.lpips_metric = lpips.LPIPS(net='alex')

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def iuv2iuvim(self, iuv, T_mask=None):
        val, idx = torch.max(iuv[:,:,:-2], -1)
        i_pred = idx[..., None] + 1.
        u_pred = iuv[..., -2:-1]
        v_pred = iuv[...,-1:]

        i_pred_mask = val < 0.5 if T_mask is None else T_mask
        i_pred[i_pred_mask] = 0.
        iuvim = torch.cat((i_pred.float(), u_pred*255, v_pred*255), -1).numpy()

        i_pred_img = (i_pred.repeat(1,1,3) / 24. * 255).numpy().astype(np.uint8)
        i_pred_img = cv2.applyColorMap(i_pred_img, cv2.COLORMAP_HOT)

        return iuvim, i_pred_img

    def ssim_metric(self, img_pred, img_gt, iuv_pred, iuv_gt, \
                    batch, epoch, T_mask, depth_pred=None):
                    
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'][0], batch['W'][0]
        mask_at_box = mask_at_box.reshape(H, W)
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        iuv_pred = iuv_pred[y:y + h, x:x + w]
        iuv_gt = iuv_gt[y:y + h, x:x + w]
        T_mask = T_mask[y:y + h, x:x + w]

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()

        iuv_pred, i_pred_img = self.iuv2iuvim(iuv_pred, T_mask)
        iuv_gt, i_gt_img = self.iuv2iuvim(iuv_gt)

        cv2.imwrite(
            '{}/rgb_frame{:04d}_view{:04d}_epoch{:04d}.png'.format(result_dir, frame_index,
                                                   view_index, epoch),
            (img_pred[..., [2, 1, 0]] * 255))

        cv2.imwrite(
            '{}/rgb_gt_frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        cv2.imwrite(
            '{}/iuv_frame{:04d}_view{:04d}_epoch{:04d}.png'.format(result_dir, frame_index,
                                                   view_index, epoch),
            (iuv_pred))
        cv2.imwrite(
            '{}/iuv_gt_frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                      view_index),
            (iuv_gt))

        cv2.imwrite(
            '{}/seg_frame{:04d}_view{:04d}_epoch{:04d}.png'.format(result_dir, frame_index,
                                                   view_index, epoch),
            (i_pred_img))
        cv2.imwrite(
            '{}/seg_gt_frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                      view_index),
            (i_gt_img))

        if cfg.output_depth:
            cv2.imwrite(
                '{}/dep_frame{:04d}_view{:04d}_epoch{:04d}.png'.format(
                    result_dir, frame_index,view_index, epoch),
            (depth_pred[y:y + h, x:x + w]))

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def evaluate(self, output, batch, epoch):
        rgb_pred = torch.clamp((output['rgb_map']),0,1)[0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        iuvmap_pred = output['iuv_map'][0].detach().cpu()   # 12978, 26
        iuvmap_gt = batch['iuv'][0].detach().cpu()  # 36735, 26

        Tmap_mask = output['T_last'][0].detach().cpu() > cfg.T_threshold
        mask_at_box = batch['mask_at_box'][0].detach().cpu()
        H, W = batch['H'][0], batch['W'][0]
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        iuv_pred = torch.zeros((H, W, 26))
        iuv_pred[mask_at_box] = iuvmap_pred
        iuv_gt = torch.zeros((H, W, 26))
        iuv_gt[mask_at_box] = iuvmap_gt
        T_mask = (torch.ones((H, W))).bool()
        T_mask[mask_at_box] = Tmap_mask

        depth_pred = None
        if cfg.output_depth:
            depth_pred = np.zeros((H, W))
            depth_pred[mask_at_box] = output['depth_map'][0].detach().cpu().numpy()
            x = np.nan_to_num(depth_pred) # change nan to 0
            mi = np.min(x) # get minimum depth
            ma = np.max(x)
            x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
            x = (255*x).astype(np.uint8)
            depth_pred = cv2.applyColorMap(x, cv2.COLORMAP_JET)

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(img_pred, img_gt, iuv_pred, iuv_gt, \
            batch, epoch, T_mask, depth_pred)
        self.ssim.append(ssim)



    def evaluate_metrics(self, output, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        T_last = output['T_last'][0].detach().cpu()
        white_bkgd = int(cfg.white_bkgd)
        
        rgb_pred = np.zeros((mask_at_box.sum(), 3)) + white_bkgd
        rgb_map = output['rgb_map'][0].detach().cpu().numpy()
        rgb_pred[T_last < cfg.T_threshold] = np.clip((rgb_map[...,:3] + rgb_map[...,3:]), 0,1)
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        H, W = batch['H'][0], batch['W'][0]
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)  
        self.psnr.append(psnr)

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        self.ssim.append(ssim)

        if cfg.use_lpips:
            lpips = self.lpips_metric(torch.from_numpy(img_pred).permute(2,0,1)[None].to(torch.float32)*2-1, \
            torch.from_numpy(img_gt).permute(2,0,1)[None].to(torch.float32)*2-1)
            self.lpips.append(lpips.numpy())

        result_dir = 'data/{}/{}/{}'.format(cfg.evaluate, cfg.task, cfg.exp_name)

        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        self.frame_ls.append(frame_index)
        new_fls = list(set(self.frame_ls))
        if len(new_fls) % cfg.save_frame == 0:
            cv2.imwrite(
                '{}/rgb_frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                        view_index),
                (img_pred[..., [2, 1, 0]] * 255))

            cv2.imwrite(
                '{}/rgb_gt_frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                            view_index),
                (img_gt[..., [2, 1, 0]] * 255))

    def summarize(self):
        metrics = {'zEval_mse': torch.tensor(np.mean(self.mse)), 
                'zEval_psnr': torch.tensor(np.mean(self.psnr)), 
                'zEval_ssim': torch.tensor(np.mean(self.ssim)),
                }
        self.mse = []
        self.psnr = []
        self.ssim = []
        if cfg.use_lpips:
          metrics['zEval_lpips'] = torch.tensor(np.mean(self.lpips))
          self.lpips = []
        return metrics