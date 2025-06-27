import yaml
import json
import numpy as np
import torch
import cv2

def config_to_str(config):
    return yaml.dump(yaml.safe_load(json.dumps(config)))  # fuck yeah


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def KL_regular(mu_1,logvar_1,mu_2,logvar_2):
    var_1=torch.exp(logvar_1)
    var_2=torch.exp(logvar_2)
    KL_loss=logvar_2-logvar_1+((var_1.pow(2)+(mu_1-mu_2).pow(2))/(2*var_2.pow(2)))-0.5
    KL_loss=KL_loss.sum(dim=1).mean()
    return KL_loss

def pixel_to_normalized(coords_pixel: torch.Tensor, 
                        H: int, W: int, 
                        align_corners: bool = False) -> torch.Tensor:
    """
    将像素坐标转换为grid_sample所需的归一化坐标
    输入：
        coords_pixel: 像素坐标张量 [..., 2] (x,y格式)
        H: 特征图高度
        W: 特征图宽度
    返回：
        coords_normalized: 归一化坐标 [-1,1]
    """
    x, y = coords_pixel[..., 0], coords_pixel[..., 1]
    
    if align_corners:
        x = (x / (W-1)) * 2 - 1
        y = (y / (H-1)) * 2 - 1
    else:
        x = (x + 0.5) / W * 2 - 1
        y = (y + 0.5) / H * 2 - 1
    
    return torch.stack([x, y], dim=-1)

def normalize_coordinates(x, y, feat_w=48, feat_h=64, align_corners=False):
    """
    将像素坐标转换为grid_sample的归一化坐标
    参数：
        x: 目标点的x坐标（支持浮点数，范围0~feat_w-1）
        y: 目标点的y坐标（范围0~feat_h-1）
    """
    if align_corners:
        # 对齐角点模式（坐标对应像素角点）
        norm_x = (x / (feat_w-1)) * 2 - 1
        norm_y = (y / (feat_h-1)) * 2 - 1
    else:
        # 对齐像素中心模式（推荐）
        norm_x = (x + 0.5) / feat_w * 2 - 1
        norm_y = (y + 0.5) / feat_h * 2 - 1
    return norm_x, norm_y


def calc_gradient_norm(named_parameters):
    total_norm = 0.0
    for name, p in named_parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    center = np.array(center)
    scale = np.array(scale)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # rot_rad = np.pi * rot / 180

    # src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def crop_image(image, center, scale, output_size):
	"""Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
	Args:
		image numpy array of shape (height, width, 3): input image
		bbox tuple of size 4: input bbox (left, upper, right, lower)

	Returns:
		cropped_image numpy array of shape (height, width, 3): resulting cropped image

	"""

	trans = get_affine_transform(center, scale, 0, output_size)
	image = cv2.warpAffine(
		image,
		trans,
		(output_size),
		flags=cv2.INTER_LINEAR)

	return image