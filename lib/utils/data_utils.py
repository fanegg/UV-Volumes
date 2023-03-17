import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


def get_near_far_with_sample_dilate(bounds, ray_o, ray_d, H, W, mask_at_body_dilate):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far

    mask_at_box = mask_at_box & mask_at_body_dilate.reshape(-1)

    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


def sample_ray_h36m_whole(img, msk, K, R, T, bounds, iuv):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    ray_d_center = ray_d[H//2, W//2, :]

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    dp_mask = (iuv[..., :24].sum(-1) != 0).astype(np.uint8)
    dp_mask = dp_mask * (msk==1)

    rgb = img.reshape(-1, 3).astype(np.float32)
    mask_at_dp = dp_mask.reshape(-1)
    mask_at_body = (msk == 1).reshape(-1)
    masked_iuv = iuv.reshape(-1, 26+int(cfg.use_bg)).astype(np.float32)
    mask_at_bbx_sub_body = ((bound_mask == 1) & (msk == 0)).reshape(-1)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    rgb = rgb[mask_at_box]
    masked_iuv = masked_iuv[mask_at_box]
    mask_at_dp = mask_at_dp[mask_at_box]
    mask_at_body = mask_at_body[mask_at_box]
    mask_at_bbx_sub_body = mask_at_bbx_sub_body[mask_at_box]
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, masked_iuv, \
           mask_at_dp.astype(np.bool), mask_at_body.astype(np.bool), mask_at_bbx_sub_body.astype(np.bool), ray_d_center

def sample_ray_h36m_whole_dilate(img, msk, K, R, T, bounds, iuv, split):
    border = 20
    kernel = np.ones((border, border), np.uint8)
    msk_dilate = msk.copy()
    msk_dilate[msk_dilate != 0] = 1
    msk_dilate = cv2.dilate(msk_dilate, kernel)
    mask_at_body_dilate = (msk_dilate==1)
    while(split == 'train' and mask_at_body_dilate.sum() > 80000):
        mask_at_body_dilate_sample = np.zeros_like(mask_at_body_dilate)
        H_shift = np.random.randint(0, ((mask_at_body_dilate.shape[0]*1)/3))
        H_size = int((mask_at_body_dilate.shape[0]*2)/3)
        W_shift = np.random.randint(0, ((mask_at_body_dilate.shape[1]*1)/3))
        W_size = int((mask_at_body_dilate.shape[1]*2)/3)        
        mask_at_body_dilate_sample[H_shift:H_shift+H_size, W_shift:W_shift+W_size] = True
        mask_at_body_dilate = mask_at_body_dilate & mask_at_body_dilate_sample

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    ray_d_center = ray_d[H//2, W//2, :]

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    dp_mask = (iuv[..., :24].sum(-1) != 0).astype(np.uint8)
    dp_mask = dp_mask * (msk==1)

    rgb = img.reshape(-1, 3).astype(np.float32)
    mask_at_dp = dp_mask.reshape(-1)

    mask_at_body_box = (msk == 1)
    mask_at_body = mask_at_body_box.reshape(-1)
    masked_iuv = iuv.reshape(-1, 26+int(cfg.use_bg)).astype(np.float32)
    mask_at_bbx_sub_body = ((bound_mask == 1) & (msk == 0)).reshape(-1)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far_with_sample_dilate(bounds, ray_o, ray_d, H, W, mask_at_body_dilate)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    rgb = rgb[mask_at_box]
    masked_iuv = masked_iuv[mask_at_box]
    mask_at_dp = mask_at_dp[mask_at_box]
    mask_at_body = mask_at_body[mask_at_box]
    mask_at_bbx_sub_body = mask_at_bbx_sub_body[mask_at_box]
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, masked_iuv, \
           mask_at_dp.astype(np.bool), mask_at_body.astype(np.bool), mask_at_bbx_sub_body.astype(np.bool), ray_d_center

