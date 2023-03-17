import torch
from lib.config import cfg, args
import torch.nn.functional as F

class Renderer:
    def __init__(self, net):
        self.net = net
        self.bg = int(cfg.use_bg)

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples, device="cuda")
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals
 
        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.cuda.FloatTensor(z_vals.shape).uniform_()
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]
        return pts, z_vals

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']
        sp_input['epoch'] = batch['epoch']

        sp_input['pose'] = batch['poses']
        sp_input['ratio'] = batch['ratio']
        sp_input['ray_d_center'] = batch['ray_d_center']
        
        return sp_input

    def get_density_color(self, wpts, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        xyzc_features, alpha = raw_decoder(wpts)
        return xyzc_features, alpha

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume, 
                        mask_at_dp, mask_at_body, i_mask, sp_input):
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point: self.net.calculate_density_iuv(
            x_point, feature_volume, sp_input)
        
        # compute the color and density
        wpts_xyzc_features, wpts_alpha = self.get_density_color(wpts, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw_xyzc_features = wpts_xyzc_features.reshape(-1, n_sample, wpts_xyzc_features.shape[2])
        raw_wpts_alpha = wpts_alpha.reshape(-1, n_sample, wpts_alpha.shape[2])
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)

        output = self.iuv_feature_weighting(
                raw_xyzc_features, raw_wpts_alpha, z_vals, ray_d, mask_at_dp, mask_at_body, i_mask, sp_input['ratio'], cfg.raw_noise_std)

        T_last = output['T_last']
        mask = mask_at_body[0] if self.net.training else T_last < cfg.T_threshold

        viewdir = viewdir[0][mask].contiguous()

        rgb_map = self.net.get_implicit_rgb_pose_once(output['iuv_body'], viewdir, sp_input['pose'], sp_input['epoch'], sp_input['latent_index'])

        ret = {
            'iuv_map': output['iuv_dp'][None],
            'rgb_map': rgb_map[None],
            'T_last' : T_last[None]
        }

        if not self.net.training and cfg.output_depth:
            ret['depth_map'] = output['depth_map'].view(n_batch, n_pixel)

        return ret


    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        mask_at_dp = batch['mask_at_dp']
        mask_at_body = batch['mask_at_body']
        i_mask = batch['iuv'][...,self.bg:24+self.bg].bool()    # 1, 1024, 24

        # encode neural body
        sp_input = self.prepare_sp_input(batch)

        feature_volume = self.net.encode_sparse_voxels(sp_input)

        if args.type == 'evaluate':
            n_batch, n_pixel = ray_o.shape[:2]
            chunk = n_pixel // cfg.batch_rays
            ret_list = []
            for i in range(0, n_pixel, chunk):
                ray_o_chunk = ray_o[:, i:i + chunk]
                ray_d_chunk = ray_d[:, i:i + chunk]
                near_chunk = near[:, i:i + chunk]
                far_chunk = far[:, i:i + chunk]
                mask_at_dp_chunk = mask_at_dp[:, i:i + chunk]
                mask_at_body_chunk = mask_at_body[:, i:i + chunk]
                i_mask_chunk = i_mask[:, i:i + chunk]
                pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                          near_chunk, far_chunk,
                                          feature_volume, mask_at_dp_chunk,
                                          mask_at_body_chunk, i_mask_chunk, 
                                          sp_input)
                ret_list.append(pixel_value)
            keys = ret_list[0].keys()
            ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
        else:            
            ret = self.get_pixel_value(ray_o, ray_d,
                                    near, far,
                                    feature_volume, mask_at_dp,
                                    mask_at_body, i_mask, 
                                    sp_input)

        return ret


    def iuv_feature_weighting(self, raw_xyzc_features, raw_wpts_alpha, z_vals, rays_d, mask_at_dp, mask_at_body, i_mask, ratio, raw_noise_std=0):
        raw2alpha = lambda sigma, dists: 1. - torch.exp(-sigma * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists,
            torch.cuda.FloatTensor(1).fill_(1e10).expand(dists[..., :1].shape)
            ],
            -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.cuda.FloatTensor(raw_wpts_alpha[...,0].shape).normal_() * raw_noise_std
  
        sigma = F.relu(raw_wpts_alpha[...,0])
        alpha = raw2alpha(sigma + noise, dists)  # [N_rays, N_samples]
        T_i = torch.cumprod(
                        torch.cat(
                            [torch.cuda.FloatTensor(alpha.shape[0], 1).fill_(1.), 1. - alpha + 1e-10],
                            -1), -1)[:, :-1]
        weights = alpha * T_i  # [1024, 64]
        T_last = T_i[:,-1]
        ret = {}

        if self.net.training:
            mask_at_body = mask_at_body[0]

            iuv_body_in = raw_xyzc_features[mask_at_body]  #[1024, 64, 72+1]
            iuv_body_feature = torch.sum(weights[mask_at_body,:,None] * iuv_body_in, -2)  # [mask_at_body, 72+1]
            iuv_body = self.net.feature2iuv(iuv_body_feature)
            iuv_whole = torch.cuda.FloatTensor(raw_xyzc_features.shape[0], iuv_body.shape[1]).fill_(0.) # 1024, 72
            iuv_whole[mask_at_body] = iuv_body
            i_mask = i_mask[mask_at_dp]
            mask_at_dp = mask_at_dp[0]
            iuv_dp = iuv_whole[mask_at_dp]
            u_map = iuv_dp[..., 24:48][i_mask][:,None]
            v_map = iuv_dp[..., 48:][i_mask][:,None]
            iuv_dp = torch.cat((iuv_dp[..., :24], u_map, v_map), -1)  #[mask_at_dp, 26+1]


        else:
            T_mask = T_last < cfg.T_threshold
            iuv_body_in = raw_xyzc_features[T_mask]  #[T_mask, 64, 72+1]
            iuv_body_feature = torch.sum(weights[T_mask,:,None] * iuv_body_in, -2)  # [T_mask, 72+1]
            iuv_body = self.net.feature2iuv(iuv_body_feature)
            if T_mask.sum() == 0:
                iuv_dp = torch.cuda.FloatTensor(raw_xyzc_features.shape[0], 26).fill_(0.)
            else:
                i_sum = iuv_body[..., :24]  # [T_mask, 24+1]
                i_out = F.softmax(i_sum, dim=-1) # [T_mask, 24+1]
                i_mask = torch.cuda.FloatTensor(i_out.shape).fill_(0.) # [T_mask, 24+1]
                _, i_idex = i_out.max(-1)
                i_mask.scatter_(1,i_idex.unsqueeze(1),1)
                i_mask = i_mask == 1

                u_map = (iuv_body[..., 24:48][i_mask])[:,None]
                v_map = (iuv_body[..., 48:][i_mask])[:,None]
                iuv_dp_raw = torch.cat((i_sum, u_map, v_map), -1)  #[T_mask, 26+1]
                iuv_dp = torch.cuda.FloatTensor(raw_xyzc_features.shape[0], 26).fill_(0.)
                iuv_dp[T_mask] = iuv_dp_raw

            if cfg.output_depth:
                depth_map = torch.sum(weights * z_vals, -1)
                ret['depth_map'] = depth_map

        ret.update({
            'iuv_body': iuv_body,
            'iuv_dp': iuv_dp,
            'T_last' : T_last
        })

        return ret
