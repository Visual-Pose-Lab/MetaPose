import numpy as np
import torch
import hashlib
from utils.img import image_batch_to_torch

import os
import zipfile
import cv2
import random
import torch.nn as nn
import torch.distributed as dist

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

class data_prefetcher():
    def __init__(self, loader, device, is_train, flip_test, backbone):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.is_train = is_train
        # self.flip = False
        self.flip_test = flip_test
        self.backbone = backbone

        if backbone in ['hrnet_32', 'hrnet_48']:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device)
        elif backbone == 'cpn':
            self.mean = torch.tensor([122.7717, 115.9465, 102.9801]).cuda().to(device).view(1, 1, 1, 3)
            self.mean /= 255.

        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)-4):
                self.next_batch[i] = self.next_batch[i].cuda(non_blocking=True).to(self.device)

            images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, subject, action, cam, images_path = self.next_batch

            images_batch = torch.flip(images_batch, [-1])  #! for cv2

            if self.backbone in ['hrnet_32', 'hrnet_48']:
                images_batch = (images_batch / 255.0 - self.mean) / self.std
            elif self.backbone == 'cpn':
                images_batch = images_batch / 255.0 - self.mean  # for CPN
                
            keypoints_3d_gt[..., 1:, :] -= keypoints_3d_gt[..., :1, :]
            keypoints_3d_gt[..., 0, :] = 0

            if random.random() <= 0.5 and self.is_train and self.flip_test:
                images_batch = torch.flip(images_batch, [-2])

                keypoints_2d_batch_cpn[..., 0] *= -1
                keypoints_2d_batch_cpn[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn[..., joints_right + joints_left, :]

                keypoints_2d_batch_cpn_crop[..., 0] = 192 - keypoints_2d_batch_cpn_crop[..., 0] - 1
                keypoints_2d_batch_cpn_crop[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn_crop[..., joints_right + joints_left, :]

                keypoints_3d_gt[..., 0] *= -1
                keypoints_3d_gt[..., joints_left + joints_right, :] = keypoints_3d_gt[..., joints_right + joints_left, :]

            if (not self.is_train) and self.flip_test:
                # import pdb; pdb.set_trace()
                images_batch = torch.stack([images_batch, torch.flip(images_batch, [-2])], dim=1)

                keypoints_2d_batch_cpn_flip = keypoints_2d_batch_cpn.clone()
                keypoints_2d_batch_cpn_flip[..., 0] *= -1
                keypoints_2d_batch_cpn_flip[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn_flip[..., joints_right + joints_left, :]
                keypoints_2d_batch_cpn = torch.stack([keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_flip], dim=1)

                keypoints_2d_batch_cpn_crop_flip = keypoints_2d_batch_cpn_crop.clone()
                keypoints_2d_batch_cpn_crop_flip[..., 0] = 192 - keypoints_2d_batch_cpn_crop_flip[..., 0] - 1
                keypoints_2d_batch_cpn_crop_flip[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn_crop_flip[..., joints_right + joints_left, :]
                keypoints_2d_batch_cpn_crop = torch.stack([keypoints_2d_batch_cpn_crop, keypoints_2d_batch_cpn_crop_flip], dim=1)

                del keypoints_2d_batch_cpn_flip, keypoints_2d_batch_cpn_crop_flip

            self.next_batch = [images_batch.float(), keypoints_3d_gt.float(), keypoints_2d_batch_cpn.float(), keypoints_2d_batch_cpn_crop.float(), subject, action, cam, images_path]


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def all_gather_strings(string_list, device):
    """
    在分布式环境中收集多个进程的字符串列表。
    假设当前已经调用了 dist.init_process_group，并且各进程都已同步。
    """
    # 将字符串列表先拼接为一个以特殊分隔符分隔的字符串
    delimiter = "<SEP>"
    joined_str = delimiter.join(string_list)
    
    # 编码为字节并转换为 Tensor
    byte_tensor = torch.ByteTensor(list(joined_str.encode("utf-8"))).to(device)
    length_tensor = torch.LongTensor([byte_tensor.size(0)]).to(device)
    
    # 所有进程先收集长度信息
    gathered_lengths = [torch.LongTensor([0]).to(device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_lengths, length_tensor)
    
    # 找出所有进程的 max_length，方便统一张量形状
    max_length = max([l.item() for l in gathered_lengths])
    padded_byte_tensor = torch.zeros(max_length, dtype=torch.uint8).to(device)
    padded_byte_tensor[:byte_tensor.size(0)] = byte_tensor
    
    # 收集每个进程的字节数据
    gathered_bytes = [torch.zeros(max_length, dtype=torch.uint8).to(device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_bytes, padded_byte_tensor)
    
    # 解析每个进程的字符串列表
    all_str_lists = []
    for rank in range(dist.get_world_size()):
        # 获取实际长度
        data_len = gathered_lengths[rank].item()
        # 根据实际长度截取并解码为字符串
        raw_str = gathered_bytes[rank][:data_len].tolist()
        decoded_str = bytes(raw_str).decode("utf-8")
        # 分割为原来的字符串列表
        str_list = decoded_str.split(delimiter)
        all_str_lists.extend(str_list)
    
    return all_str_lists

# for camera projection
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    
    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)
    
def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def mpjpe_cal_l1(predicted, target):
    assert predicted.shape == target.shape
    criterion = nn.L1Loss(reduction='mean')
   # return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    return criterion(predicted,target)

def test_calculation(predicted, target, actions):
    error_sum = define_error_list(actions)
    error_sum = mpjpe_by_action_p1(predicted, target, actions, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, actions, error_sum)
    # import pdb; pdb.set_trace()

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item()*frame_num, frame_num)
            
    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    assert frame_num==1
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist).item() * batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist[i]).item(), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","TakingPhoto","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def log_error(logger, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = log_error_action(logger, action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def log_error_action(log, action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        log.info("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            log.info("{0:<12} {1:>6.2f} {2:>10.2f}".format(action, mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        log.info("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg

