import numpy as np

class SequenceGenerator:
    '''
    改自ChunkedGenerator，只能生成many-to-one 的 2d 3d 动作序列，即 2D:(f,p,c) -> 3D:(p,c)
    poses_2d_crop与图像特征保证同步

    chunk_length:   主要和3D序列长度相关
    pad_img:        主要和2D_crop序列长度相关
    pad_2d:         主要和2D序列相关
    tds_img:        img 序列的步长
    tds_2d:         2d  序列的步长

    当前可用功能： 
    1. tds     序列降采样
    2. causal  因果序列生成  ->  未实现
    '''
    def __init__(self, poses_3d, poses_2d, poses_2d_crop, chunk_length=1, pad_img=0, pad_2d=0, tds_img=1, tds_2d=1, out_all=False):
        assert len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))

        pairs = []
        self.saved_index = {}
        start_index = 0
        for key in poses_2d.keys():
            assert poses_2d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            keys = np.tile(np.array(key).reshape([1,3]),(len(bounds - 1),1))  
            
            # seqname, start, end -> (start,end) is the range of the chunk
            pairs += list(zip(keys, bounds[:-1], bounds[1:])) 

            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index, end_index]
            start_index = end_index

        self.frames_len = start_index
        self.pairs = pairs

        self.chunk_length = chunk_length
        self.pad_2d = pad_2d
        self.pad_img = pad_img
        self.tds_2d = tds_2d
        self.tds_img = tds_img

        self.state = None
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.poses_2d_crop = poses_2d_crop

        self.out_all = out_all

    def num_frames(self):
        return self.frames_len
    
    def sequence_padding(self, seq, start, end, pad, tds):
        start = start - pad * tds 
        end = end + pad * tds
        low = max(start, 0)
        high = min(end, len(seq))
        pad_left = low - start
        pad_right = end - high
        if pad_left != 0:
            data_pad = np.repeat(seq[0:1], pad_left, axis=0)
            new_data = np.concatenate((data_pad, seq[low:high]), axis=0)
            seq_pad = new_data[::tds]
        elif pad_right != 0:
            data_pad = np.repeat(seq[len(seq)-1:len(seq)], pad_right, axis=0)
            new_data = np.concatenate((seq[low:high], data_pad), axis=0)
            seq_pad = new_data[::tds]
        else:
            seq_pad = seq[low:high:tds]
        return seq_pad
        
    def get_sequence(self, seq_i, start_3d, end_3d):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))

        # for 2D sequence
        seq_2d = self.poses_2d[seq_name].copy()
        seq_2d_pad = self.sequence_padding(seq_2d, start_3d, end_3d, self.pad_2d, self.tds_2d)
       
        # for Img sequence
        seq_2d_crop = self.poses_2d_crop[seq_name].copy()
        seq_2d_crop_pad = self.sequence_padding(seq_2d_crop, start_3d, end_3d, self.pad_img, self.tds_img)

        # for 3D sequence
        seq_3d = self.poses_3d[seq_name].copy()
        if self.out_all:
            seq_3d_pad = self.sequence_padding(seq_3d, start_3d, end_3d, self.pad_2d, self.tds_2d)
        else:
            seq_3d_pad = self.sequence_padding(seq_3d, start_3d, end_3d, 0, 1)

        return seq_3d_pad, seq_2d_pad, seq_2d_crop_pad, subject, action, int(cam_index)
    

    def get_sequence_index(self, seq_i, start_3d, end_3d):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))

        seq_2d_crop = self.poses_2d_crop[seq_name].copy()
        seq_index = np.arange(len(seq_2d_crop))
        seq_index_pad = self.sequence_padding(seq_index, start_3d, end_3d, self.pad_img, self.tds_img)

        return seq_index_pad





            

