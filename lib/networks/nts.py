# from curses import window
import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from . import embedder

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.c = nn.Embedding(cfg.nv, cfg.code_dim)
        self.actvn = nn.ReLU()
        self.xyzc_net = SparseConvNet_64()
        self.latent_fc1 = nn.Conv1d(64, 64, 1)
        self.latent_fc2 = nn.Conv1d(64, 64, 1)
        self.latent_out = nn.Conv1d(64, 1, 1)
        self.feature2iuvmlp = nn.Sequential(
                        nn.Linear(64, 256), nn.ReLU(True),
                        nn.Linear(256, 256), nn.ReLU(True),
                        nn.Linear(256, 256), nn.ReLU(True),
                        nn.Linear(256, 256), nn.ReLU(True),
                        nn.Linear(256, 72))

        self.implicit_tex_model = hyper_implicit_texture_dynamicMLP_once()


    def encode_sparse_voxels(self, sp_input):
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        code = self.c(torch.arange(0, cfg.nv).to(coord.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
        feature_volume = self.xyzc_net(xyzc)

        return feature_volume

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_grid_coords(self, pts, sp_input):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.cuda.FloatTensor([cfg.voxel_size])

        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.cuda.FloatTensor(sp_input['out_sh'])
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def calculate_density(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)
        grid_coords = self.get_grid_coords(ppts, sp_input)
        grid_coords = grid_coords[:, None, None]
        xyzc_features = self.interpolate_features(grid_coords, feature_volume)

        # calculate density
        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)
        alpha = alpha.transpose(1, 2)

        return alpha

    def calculate_density_iuv(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)
        grid_coords = self.get_grid_coords(ppts, sp_input)
        grid_coords = grid_coords[:, None, None]
        xyzc_features = self.interpolate_features(grid_coords, feature_volume)

        alpha = self.latent_out( \
            self.actvn(self.latent_fc2(\
                self.actvn(self.latent_fc1(xyzc_features)))))

        return xyzc_features.transpose(1, 2), alpha.transpose(1, 2)

    def feature2iuv(self, feature):
        iuv = self.feature2iuvmlp(feature)
        return torch.cat((iuv[..., :24], torch.sigmoid(iuv[..., 24:72])), -1)

    def get_implicit_rgb_pose_once(self, iuv_map, viewdir, pose, epoch, latent_index):
        i_map = F.softmax(iuv_map[..., :24], dim=-1)
        u_map = iuv_map[..., 24:48]
        v_map = iuv_map[..., 48:]
        i_onehot = torch.eye(24)[torch.arange(24)].cuda().unsqueeze(1).expand(-1,i_map.shape[0],-1).detach()  #24,mask,24
        uv_map = torch.stack((u_map, v_map), -1)  # mask, 24, 2

        uv_encoding = embedder.uv_embedder(uv_map.view(1,-1,2)) # 1, mask*24, 42
        uv_encoding = uv_encoding.view(-1,24,42).transpose(1,0)  #  24, mask, 42
        iuv_encoding = torch.cat((i_onehot, uv_encoding), -1)  # 24, mask, 24 + 42
        iuv_encoding = iuv_encoding.view(-1, iuv_encoding.shape[-1]) # 24 * mask, 24+42

        if uv_map.requires_grad:
            viewdir = viewdir + (torch.randn_like(viewdir)*cfg.view_noise_weight)
        viewdirs_encoding = embedder.view_embedder(viewdir[None].expand(24,-1,-1).contiguous().view(-1, 3))   # 24 * mask, 27

        rgb = self.implicit_tex_model.get_rgb(iuv_encoding, pose, viewdirs_encoding)
        rgb_pred = (i_map.permute(1,0).unsqueeze(2) * rgb.view(24,-1,3)).sum(0)
        delta_rgb_pred = torch.cuda.FloatTensor(rgb_pred.shape).fill_(0.)
        rgb_pred_out = torch.cat((rgb_pred, delta_rgb_pred), -1) # 714, 6
        return rgb_pred_out


class SparseConvNet_64(nn.Module):
    def __init__(self):
        super(SparseConvNet_64, self).__init__()

        self.conv0 = double_conv(cfg.code_dim, 
                                 cfg.code_dim, 'subm0')
        self.down0 = stride_conv(cfg.code_dim, 16, 'down0')

        self.conv1 = double_conv(16, 16, 'subm1')
        self.down1 = stride_conv(16, 16, 'down1')

        self.conv2 = triple_conv(16, 16, 'subm2')
        self.down2 = stride_conv(16, 16, 'down2')

        self.conv3 = triple_conv(16, 16, 'subm3')
        self.down3 = stride_conv(16, 16, 'down3')

        self.conv4 = triple_conv(16, 16, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes

def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class implicit_texture_dynamicMLP(nn.Module):
    def __init__(self, W=256, channels_i=24, channels_uv=2, channels_uv_high=42, \
        channels_view=3, channels_pose=128, channels_latent=128):
        super().__init__()

        self.channels_pose = channels_pose

        # uv encoding layers
        self.rgb_mapping = nn.Sequential(
                            nn.Linear(channels_i + channels_uv_high + channels_latent, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Sequential(nn.Linear(W, 3), nn.Sigmoid()))

        self.viewin_layer = nn.Sequential(nn.Linear(channels_i + channels_uv + channels_view + channels_latent//2, W), nn.ReLU(True),
                                          nn.Linear(W, W), nn.ReLU(True),
                                          nn.Linear(W, 3), nn.Tanh())

        self.latent_square = 2
        self.pose2latent = nn.Linear(channels_i + channels_pose, 512 * self.latent_square * self.latent_square)
        self.latent_decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                    padding=1, output_padding=1), nn.LeakyReLU(),# 24, 256, 4, 4
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                    padding=1, output_padding=1), nn.LeakyReLU(),# 24, 128, 8, 8
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1), nn.LeakyReLU(), # 24, 128, 16, 16
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1), nn.LeakyReLU(), # 24, 128, 32, 32
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1)) # 24, 128, 64, 64

    def get_pose_rgb(self, uv, slatent):
        i_onehot = torch.eye(24)[torch.arange(24)].cuda().detach()  #24, 24
        latent_theta = self.pose2latent(torch.cat((i_onehot, slatent.repeat(24,1)), -1))   #24, 24+128 -> 24, 512 * 32 * 32
        decode_latent = self.latent_decoder(latent_theta.view(-1, 512, self.latent_square, self.latent_square)) # 24, 512, 32, 32 -> 24, 128, 256, 256

        decode_latent_gridsample = nn.functional.grid_sample(decode_latent,
                            (2*uv.transpose(1,0).unsqueeze(1)-1), 
                            mode='bilinear', align_corners=False) # 24, 1, mask, 2 - > 24, 128, 1, mask
        decode_latent_gridsample = decode_latent_gridsample.squeeze(2).transpose(2,1)   # 24, mask, 128

        i = i_onehot.unsqueeze(1).repeat(1,uv.shape[0],1)  # 24,mask,24
        uv_encoding = embedder.xyz_embedder(uv.view(1,-1,2)) #1, mask*24, 42
        uv_encoding = uv_encoding.view(-1,24,42).transpose(1,0)  #24, mask, 42

        iuvl_encoding = torch.cat((i, uv_encoding, decode_latent_gridsample), -1)  # 24, mask, 24+42+128
        rgb = self.rgb_mapping(iuvl_encoding.view(-1,iuvl_encoding.shape[-1]))   # 24 * mask, 24+42+128
        return rgb

    def get_view_rgb(self, x):
        delta_rgb = self.viewin_layer(x)  #24*mask, 24+2+3+64
        return delta_rgb


class hyper_implicit_texture_dynamicMLP_once(nn.Module):
    def __init__(self, channels_i=24, channels_uv=2, channels_uv_high=42, \
        channels_view=27, channels_latent_hyper=cfg.pose_dim, channels_latent_app=128, hyper_width=64, rgb_width=256):        # channels_view_high=27
        super().__init__()

        self.latent_square = 2
        self.pose2latent = nn.Linear(channels_i + channels_latent_hyper, 512 * self.latent_square * self.latent_square)
        self.latent_decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                    padding=1, output_padding=1), nn.LeakyReLU(),# 24, 256, 4, 4
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                    padding=1, output_padding=1), nn.LeakyReLU(),# 24, 128, 8, 8
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1), nn.LeakyReLU(), # 24, 128, 16, 16
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1), nn.LeakyReLU(), # 24, 128, 32, 32
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                                        padding=1, output_padding=1)) # 24, 128, 64, 64

        # uv layers
        self.rgb_mapping_1 = nn.Sequential(
                            nn.Linear(channels_i + channels_uv_high + 128, rgb_width), nn.ReLU(True),
                            nn.Linear(rgb_width, rgb_width), nn.ReLU(True),
                            nn.Linear(rgb_width, rgb_width), nn.ReLU(True),
                            nn.Linear(rgb_width, rgb_width), nn.ReLU(True))
        self.rgb_mapping_2 = nn.Sequential(
                            nn.Linear(channels_i + channels_uv_high + 128 + rgb_width, rgb_width), nn.ReLU(True))
        self.rgb_mapping_3 = nn.Sequential(
                            nn.Linear(channels_view + rgb_width, rgb_width//2), nn.ReLU(True),
                            nn.Linear(rgb_width//2, 3))

    def get_rgb(self, iuv_encoding, pose, view):
        iuv_view = iuv_encoding.view(24, -1, 24 + 42)
        if iuv_view.shape[1] > 0:
            i_onehot = iuv_view[:,0,:24].detach()  #24, 24
        else:
            i_onehot = torch.eye(24)[torch.arange(24)].cuda().detach()  #24, 24
        uv = iuv_view[...,24:26]  # 24, mask, 2

        latent_theta = self.pose2latent(torch.cat((i_onehot, pose.expand(24,-1)), -1))   #24, 24+128 -> 24, 512 * 32 * 32
        decode_latent = self.latent_decoder(latent_theta.view(-1, 512, self.latent_square, self.latent_square)) # 24, 512, 32, 32 -> 24, 16, 256, 256

        decode_latent_gridsample = nn.functional.grid_sample(decode_latent,
                            (2*uv.unsqueeze(1)-1), 
                            mode='bilinear', align_corners=False) # 24, 1, mask, 2 - > 24, 128, 1, mask
        hyper = (decode_latent_gridsample.squeeze(2).transpose(2,1)).contiguous().view(-1, 128)   # 24 * mask, 128

        feature = self.rgb_mapping_1(torch.cat((iuv_encoding, hyper), -1))
        feature = self.rgb_mapping_2(torch.cat((iuv_encoding, hyper, feature), -1))
        rgb = self.rgb_mapping_3(torch.cat((view, feature), -1))

        return torch.sigmoid(rgb)

import math
class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))