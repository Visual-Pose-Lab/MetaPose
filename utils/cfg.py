import yaml
from easydict import EasyDict as edict
import os

config = edict()

config.title = "human36m_vol_softmax_single"
config.kind = "human36m"
config.azureroot = ""
config.logdir = "logs"
config.batch_output = False
config.id = 600
config.frame = 1

# model definition
config.model = edict()
config.model.image_shape = [192, 256]
config.model.init_weights = True
config.model.checkpoint = None

config.model.text_encoder = edict()
config.model.text_encoder.type = 'ViT-B/32'

config.model.image_encoder = edict()
config.model.image_encoder.type = 'hrnet_32'
config.model.image_encoder.num_final_layer_channel = 17
config.model.image_encoder.num_joints = 17
config.model.image_encoder.num_layers = 152
config.model.image_encoder.init_weights = True
config.model.image_encoder.fix_weights = False
config.model.image_encoder.checkpoint = "data/pretrained/human36m/pose_hrnet_w32_256x192.pth"

# pose_hrnet related params
# config.model.image_encoder = edict()
config.model.image_encoder.NUM_JOINTS = 17
config.model.image_encoder.PRETRAINED_LAYERS = ['*']
config.model.image_encoder.STEM_INPLANES = 64
config.model.image_encoder.FINAL_CONV_KERNEL = 1

config.model.image_encoder.STAGE2 = edict()
config.model.image_encoder.STAGE2.NUM_MODULES = 1
config.model.image_encoder.STAGE2.NUM_BRANCHES = 2
config.model.image_encoder.STAGE2.NUM_BLOCKS = [4, 4]
config.model.image_encoder.STAGE2.NUM_CHANNELS = [32, 64]
# config.model.image_encoder.STAGE2.NUM_CHANNELS = [48, 96]
config.model.image_encoder.STAGE2.BLOCK = 'BASIC'
config.model.image_encoder.STAGE2.FUSE_METHOD = 'SUM'

config.model.image_encoder.STAGE3 = edict()
# config.model.image_encoder.STAGE3.NUM_MODULES = 1
config.model.image_encoder.STAGE3.NUM_MODULES = 4
config.model.image_encoder.STAGE3.NUM_BRANCHES = 3
config.model.image_encoder.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.model.image_encoder.STAGE3.NUM_CHANNELS = [32, 64, 128]
# config.model.image_encoder.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.model.image_encoder.STAGE3.BLOCK = 'BASIC'
config.model.image_encoder.STAGE3.FUSE_METHOD = 'SUM'

config.model.image_encoder.STAGE4 = edict()
# config.model.image_encoder.STAGE4.NUM_MODULES = 1
config.model.image_encoder.STAGE4.NUM_MODULES = 3
config.model.image_encoder.STAGE4.NUM_BRANCHES = 4
config.model.image_encoder.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.model.image_encoder.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
# config.model.image_encoder.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.model.image_encoder.STAGE4.BLOCK = 'BASIC'
config.model.image_encoder.STAGE4.FUSE_METHOD = 'SUM'

# pose_resnet related params
config.model.image_encoder.NUM_LAYERS = 50
config.model.image_encoder.DECONV_WITH_BIAS = False
config.model.image_encoder.NUM_DECONV_LAYERS = 3
config.model.image_encoder.NUM_DECONV_FILTERS = [256, 256, 256]
config.model.image_encoder.NUM_DECONV_KERNELS = [4, 4, 4]
config.model.image_encoder.FINAL_CONV_KERNEL = 1
config.model.image_encoder.PRETRAINED_LAYERS = ['*']

config.model.pose_net = edict()
config.model.pose_net.use_gt_pelvis = False
config.model.pose_net.cuboid_size = 2500.0
config.model.pose_net.use_feature_v2v = True
config.model.pose_net.att_channels = 51
config.model.pose_net.temperature = 1500
config.model.pose_net.clip_image_feature = False

config.model.pose_net.base_dim = 32
config.model.pose_net.embed_dim_ratio = 128
config.model.pose_net.depth = 4
config.model.pose_net.levels = 4
config.model.pose_net.save_inter_feat = False

# loss related params
config.loss = edict()
config.loss.criterion = "MAE"
config.loss.mse_smooth_threshold = 0
config.loss.grad_clip = 0
config.loss.scale_keypoints_3d = 0.1
config.loss.use_volumetric_ce_loss = True
config.loss.volumetric_ce_loss_weight = 0.01
config.loss.use_global_attention_loss = True
config.loss.global_attention_loss_weight = 1000000

# dataset related params
config.dataset = edict()
config.dataset.kind = "human36m"
config.dataset.data_format = ''
config.dataset.transfer_cmu_to_human36m = False
config.dataset.image_root = "../H36M-Toolbox/images/"
config.dataset.train_dataset = "allhuman36m"
config.dataset.val_dataset = "allhuman36m"
config.dataset.points_only = False
config.dataset.labels_3d_path = ""
config.dataset.labels_2d_path = ""
config.dataset.labels_2d_crop_path = ""
config.dataset.train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
config.dataset.test_subjects = ['S9', 'S11']
config.dataset.actions = '*'
config.dataset.downsample = 1
config.dataset.out3d_frames = 1
config.dataset.input2d_frames = 1
config.dataset.image_frames = 1
config.dataset.input2d_tds = 1
config.dataset.image_tds = 1

# train related params
config.train = edict()
config.train.n_epochs = 9999
config.train.n_iters_per_epoch = 5000
config.train.batch_size = 16
config.train.optimizer = 'Adam'
config.train.image_encoder_lr = 0.0001
config.train.image_encoder_lr_step = [1000]
config.train.image_encoder_lr_factor = 0.1
config.train.process_features_lr = 0.001
config.train.pose_net_lr = 0.001
config.train.pose_net_lr_decay = 0.99
config.train.pose_net_lr_step = [1000]
config.train.pose_net_lr_factor = 0.5
config.train.shuffle = True
config.train.downsample = 1
config.train.subset = 1
config.train.num_workers = 8

# val related params
config.val = edict()
config.val.flip_test = True
config.val.batch_size = 6
config.val.shuffle = False
config.val.downsample = 1
config.val.subset = 1
config.val.num_workers = 10



def update_dict(v, cfg):
    for kk, vv in v.items():
        if kk in cfg:
            if isinstance(vv, dict):
                update_dict(vv, cfg[kk])
            else:
                cfg[kk] = vv
        else:
            raise ValueError("{} not exist in cfg.py".format(kk))


def update_config(path):
    exp_config = None
    with open(path) as fin:
        exp_config = edict(yaml.safe_load(fin))
        update_dict(exp_config, config)


def handle_azureroot(config_dict, azureroot):
    for key in config_dict.keys():
        if isinstance(config_dict[key], str):
            if config_dict[key].startswith('data/'):
                config_dict[key] = os.path.join(azureroot, config_dict[key])
        elif isinstance(config_dict[key], dict):
            handle_azureroot(config_dict[key], azureroot)


def update_dir(azureroot, logdir):
    config.azureroot = azureroot
    config.logdir = os.path.join(config.azureroot, logdir)
    if config.model.checkpoint != None and not config.model.checkpoint.startswith('data/'):
        config.model.checkpoint = os.path.join(config.azureroot, config.model.checkpoint)
    handle_azureroot(config, config.azureroot)   

   