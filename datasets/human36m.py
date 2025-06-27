import os
import collections
from collections import defaultdict
import pickle

import numpy as np
np.set_printoptions(suppress=True)
import cv2
import os.path as osp
from torch.utils.data import Dataset

from datasets.utils import deterministic_random, normalize_screen_coordinates
from datasets.generator_tds import SequenceGenerator

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

class Human36MAllDataset(Dataset):
    def __init__(self, config=None, train=True, downsample=1, subset=1, crop=False):
        self.image_root = config.image_root
        self.chunk_len = config.out3d_frames
        self.stride = downsample
        self.pad_2d = (config.input2d_frames - 1) // 2
        self.pad_img = (config.image_frames - 1) // 2
        self.tds_2d = config.input2d_tds
        self.tds_img = config.image_tds
        actions = config.actions
        self.action_filter = None if actions == '*' else actions
        self.subjects = config.train_subjects if train else config.test_subjects
        self.crop = crop

        # 数据长度匹配，筛人
        # print('INFO: Preparing data...')
        poses_3d, poses_2d, poses_2d_crop = self.prepare_data(config.labels_3d_path,
                                                              config.labels_2d_path,
                                                              config.labels_2d_crop_path,
                                                              self.subjects)
        # 重新建立键值对
        # print('INFO: Fetching data...')
        self.poses_3d, self.poses_2d, self.poses_2d_crop = self.fetch(poses_3d, 
                                                                      poses_2d, 
                                                                      poses_2d_crop, 
                                                                      self.subjects, subset)
        
        self.generator = SequenceGenerator(self.poses_3d, self.poses_2d, self.poses_2d_crop, 
                                           chunk_length=self.chunk_len, 
                                           pad_img=self.pad_img, pad_2d=self.pad_2d,
                                           tds_img=self.tds_img, tds_2d=self.tds_2d,
                                           out_all=False)
        
        print('INFO: {} on {} frames'.format('Training' if train else 'Testing',
                                                self.generator.num_frames()))
        self.key_index = self.generator.saved_index


    def prepare_data(self, labels_3d_path, labels_2d_path, labels_2d_crop_path, folder_list):
        kps_3d = np.load(labels_3d_path, allow_pickle=True)['positions_3d'].item()
        kps_2d = np.load(labels_2d_path, allow_pickle=True)['positions_2d'].item()
        kps_2d_crop = np.load(labels_2d_crop_path, allow_pickle=True)['positions_2d'].item()

        for subject in folder_list:
            assert subject in kps_2d, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            assert subject in kps_2d_crop, 'Subject {} is missing from the 2D detections cropped dataset'.format(subject)
            for action in kps_3d[subject].keys():
                assert action in kps_2d[subject], \
                    'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
                assert action in kps_2d_crop[subject], \
                    'Action {} of subject {} is missing from the 2D detections cropped dataset'.format(action, subject)
                for cam_idx in range(len(kps_2d[subject][action])):

                    mocap_length = kps_3d[subject][action][cam_idx].shape[0]

                    assert kps_2d[subject][action][cam_idx].shape[0] >= mocap_length
                    assert kps_2d_crop[subject][action][cam_idx].shape[0] >= mocap_length

                    if kps_2d[subject][action][cam_idx].shape[0] > mocap_length:
                        kps_2d[subject][action][cam_idx] = kps_2d[subject][action][cam_idx][:mocap_length]

                    if kps_2d_crop[subject][action][cam_idx].shape[0] > mocap_length:
                        kps_2d_crop[subject][action][cam_idx] = kps_2d_crop[subject][action][cam_idx][:mocap_length]

        
        return kps_3d, kps_2d, kps_2d_crop

    def fetch(self, poses_3d, poses_2d, poses_2d_crop, subjects, subset=1):
        out_poses_3d = {}
        out_poses_2d = {}
        out_poses_2d_crop = {}

        for subject in subjects:
            for action in poses_3d[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                pose_2d = poses_2d[subject][action]
                for i in range(len(pose_2d)):
                    out_poses_2d[(subject, action, i)] = pose_2d[i][..., :2]
                
                pose_2d_crop = poses_2d_crop[subject][action]
                assert len(pose_2d) == len(pose_2d_crop), 'Camera count mismatch'
                for i in range(len(pose_2d_crop)):
                    assert len(pose_2d[i]) == len(pose_2d_crop[i]), 'frames count mismatch'
                    out_poses_2d_crop[(subject, action, i)] = pose_2d_crop[i][..., :2]
                
                if poses_3d[subject][action]:
                    pose_3d = poses_3d[subject][action]
                    assert len(pose_3d) == len(pose_2d), 'Camera count mismatch'
                    for i in range(len(pose_3d)): 
                        out_poses_3d[(subject, action, i)] = pose_3d[i]

        if len(out_poses_3d) == 0:
            out_poses_3d = None

        # subset     随机取其中一个n等分分组
        # downsample 降采样
        stride = self.stride
        if subset < 1:          # 
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                out_poses_2d_crop[key] = out_poses_2d_crop[key][start:start + n_frames:stride]
                out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:        
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                out_poses_2d_crop[key] = out_poses_2d_crop[key][::stride]
                out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_poses_3d, out_poses_2d, out_poses_2d_crop

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d = self.generator.pairs[index]

        gt_3D, input_2D, input_2D_crop, subject, action, cam_ind = self.generator.get_sequence(seq_name, start_3d, end_3d)
        img_frame_index = self.generator.get_sequence_index(seq_name, start_3d, end_3d) + 1

        # load image
        subdir = osp.join(self.image_root, subject, action, str(cam_ind))
        image_seq = []
        images_path = []
        for idx, img_frame in enumerate(img_frame_index):
            imagename = '{}_{}_{}_{:06d}.jpg'.format(subject, action, cam_ind, img_frame)
            image_path = os.path.join(subdir, imagename)

            image = cv2.imread(
                os.path.join(subdir, image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )#[..., ::-1]#.astype('float32')

            image_seq.append(np.expand_dims(image, axis=0))
            images_path.append(image_path)
            
        image_seq = np.concatenate(image_seq, axis=0)

        return image_seq, gt_3D, input_2D, input_2D_crop, subject, action, cam_ind, images_path