import os
import shutil
import argparse
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from tensorboardX import SummaryWriter
from utils.logger import Logger
from tqdm import tqdm
import datasets

from models.metapose import MetaPose
from models.loss import MPJPE, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
import clip
import glob

from utils import misc
from utils.cfg import config, update_config, update_dir
from datasets import utils as dataset_utils

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--debug', action='store_true', help="If set, debug mode will be enabled")
	
	parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
	parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
	parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

	parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument('--sync_bn', action='store_true', help="If set, then utilize pytorch convert_syncbn_model")

	parser.add_argument("--logdir", type=str, default="logs/", help="Path, where logs will be stored")
	parser.add_argument("--azureroot", type=str, default="", help="Root path, where codes are stored")

	parser.add_argument("--frame", type=int, default=1, help="Frame number to be used.")
	parser.add_argument("--image_encoder", type=str, default='hrnet_32', choices=['hrnet_32', 'hrnet_48'], help="Tmage encoder.")
	parser.add_argument("--text_encoder", type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-L/14'], help="Text encoder.")
	args = parser.parse_args()
	# update config
	update_config(args.config)
	update_dir(args.azureroot, args.logdir)
	config.model.image_encoder.type = args.image_encoder
	config.model.text_encoder.type = args.text_encoder
	return args


def setup_human36m_dataloaders(config, is_train, distributed):
	train_dataloader = None
	train_sampler = None
	if is_train:
		# train
		train_dataset = eval('datasets.' + config.dataset.train_dataset)(
			config=config.dataset,
			train=True,
			downsample=config.train.downsample,
			subset=config.train.subset)

		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
																  shuffle=config.train.shuffle) if distributed else None

		train_dataloader = DataLoader(
			train_dataset,
			batch_size=config.train.batch_size,
			shuffle=config.train.shuffle and (train_sampler is None), # debatable
			sampler=train_sampler,
			num_workers=config.train.num_workers,
			worker_init_fn=dataset_utils.worker_init_fn,
			pin_memory=True)

	# val
	val_dataset = eval('datasets.' + config.dataset.val_dataset)(
		config=config.dataset,
		train=False,
		downsample=config.val.downsample,
		subset=config.val.subset)

	val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, 
															   shuffle=config.val.shuffle) if distributed else None

	val_dataloader = DataLoader(
		val_dataset,
		batch_size=config.val.batch_size,
		shuffle=config.val.shuffle and (val_sampler is None),
		sampler=val_sampler,
		num_workers=config.val.num_workers,
		worker_init_fn=dataset_utils.worker_init_fn,
		pin_memory=True)

	return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed=False):
	if config.dataset.kind == 'human36m':
		train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed)
	else:
		raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))
	
	return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True):
	prefix = "" if is_train else "eval_"

	if config.title:
		experiment_title = config.title + "_" + model_name
	else:
		experiment_title = model_name

	experiment_title = "MetaPose"
	experiment_title = prefix + experiment_title

	experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
	print("Experiment name: {}".format(experiment_name))

	experiment_dir = os.path.join(config.logdir, experiment_name)
	os.makedirs(experiment_dir, exist_ok=True)

	checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
	os.makedirs(checkpoints_dir, exist_ok=True)

	shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

	# tensorboard
	writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

	# dump config to tensorboard
	writer.add_text(misc.config_to_str(config), "config", 0)

	return experiment_dir, writer

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def one_epoch_full(model, criterion, optimizer, config, dataloader, device, epoch, master, is_train=True):
	name = "train" if is_train else "val"

	if is_train:
		epoch_loss_3d_train = 0
		epoch_kl_loss = 0
		N = 0

		if config.model.image_encoder.fix_weights:
			model.module.image_encoder.eval()
			model.module.pose_net.train()
		else:
			model.train()
	else:
		model.eval()

	metric_dict = defaultdict(list)

	results = defaultdict(list)
	
	template = "The {joint} joint located on the {part} part is engaged in an action where the person is {action} characterized by {description}"

	pre_text_joint = ['pelvis', 'right hip', 'right knee', 'right ankle', 
			'left hip', 'left knee', 'left ankle', 'belly', 'neck', 
			'nose', 'head', 'left shoulder', 'left elbow', 
			'left wrist', 'right shoulder', 'right elbow', 'right wrist']
	
	pre_text_part = ['body', 'leg', 'leg', 'leg', 
			'leg', 'leg', 'leg', 'body', 'body', 
			'head', 'head', 'arm', 'arm', 
			'arm', 'arm', 'arm', 'arm']

	pre_text_action = {
		'Directions': 'The person stands or sits while using gestures or body movements to indicate a direction. This might involve pointing to a location, waving, or turning the body towards a target',
		'Discussion': 'The person performs actions associated with conversations, using hand gestures, head movements, and eye contact. This may include gesturing while talking, nodding, or adopting thinking or explaining poses',
		'Eating': 'The person mimics eating behaviors, usually while seated. This could involve picking up food, using utensils, chewing, or drinking, all related to mealtime activities',
		'Greeting': 'The person performs gestures of greeting, such as waving, nodding, shaking hands, or using other polite gestures to welcome or acknowledge someone',
		'Phoning': 'The person acts as if holding a phone, simulating phone-related actions like placing a hand to the ear, talking, and listening during a phone call',
		'Posing': 'The person adopts specific poses, usually remaining still. These poses might be exaggerated or stylized, simulating scenarios where the person is intentionally posing, such as for a photograph',
		'Purchases': 'The person performs actions related to shopping, such as selecting items, examining goods, or mimicking the action of paying, often involving reaching out and pretending to hold objects',
		'Sitting': 'The person sits in a chair or another surface. The actions might include adjusting the posture, crossing legs, stretching, or resting while seated',
		'SittingDown': 'The person transitions from standing to sitting. This process includes scanning for a seat, slowly bending the knees, and sitting down on a surface',
		'Smoking': 'The person mimics smoking, including holding a cigarette or e-cigarette, bringing it to the lips, inhaling, and exhaling',
		'TakingPhoto': 'The person simulates the role of a photographer, usually holding an imaginary camera with both hands, pointing it at a person, pressing the shutter, or adjusting the angle for a shot',
		'Waiting': 'The person performs actions related to waiting. This often includes standing or sitting with slight movements, such as shifting weight, checking a watch, or looking around, conveying impatience or boredom',
		'Walking': 'The person walks naturally in the scene, with steady steps and relaxed posture. This action involves typical arm swings and regular walking motion',
		'WalkDog': 'The person simulates walking a dog, which may involve holding a leash in one hand, occasionally glancing at the "dog" and adjusting the pace based on its movement',
		'WalkTogether': 'The person performs actions that represent walking alongside another person, syncing steps, occasionally exchanging glances or engaging in side conversations while walking together'
	}

	# just joint now
	pre_text_tensor = clip.tokenize(pre_text_joint)        # [token, 77] 
	joint_token = pre_text_tensor
	if torch.cuda.is_available():
		joint_token = joint_token.cuda()


	# used to turn on/off gradients
	grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
	with grad_context():
		prefetcher = dataset_utils.data_prefetcher(dataloader, device, is_train, 
											 	   config.val.flip_test, 
												   config.model.image_encoder.type)

		# for iter_i, batch in iterator:
		batch = prefetcher.next()
		data_len = len(dataloader)
		pbar = tqdm(total=data_len, desc=f"[{name} epoch {epoch+1}]", disable=(not master))

		while batch is not None:
			# measure data loading time

			images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, subjects, actions, cam, images_path = batch
			actions = [action.split(' ')[0] for action in actions]
			actions = ['TakingPhoto' if action=='Photo'else action for action in actions ]
			# keypoints_3d_gt = keypoints_3d_gt[:, 0, :, :].clone() if config.dataset.input2d_frames==1 else keypoints_3d_gt# (b, 17, 3)
			# import pdb; pdb.set_trace()
			# text prompt generation
			action_token_batch = []
			action_set = list(set(actions))
			for i in range(len(action_set)):
				action_token = []
				for j in range(len(pre_text_joint)):
					if action_set[i] in pre_text_action.keys():
						joint =  pre_text_joint[j]
						part = pre_text_part[j]
						action_description = pre_text_action[action_set[i]]
						pre_text = template.format(joint=joint, part=part, action=actions[i], description=action_description)
						action_token.append(clip.tokenize(pre_text).cuda())
					else:
						print('Action not found!')
						break
				action_token = torch.cat(action_token, dim=0).unsqueeze(0)
				action_token_batch.append(action_token)
			action_token_batch = torch.cat(action_token_batch, dim=0)


			kl = True
			if (not is_train) and config.val.flip_test:
				keypoints_3d_pred = model(images_batch[:, 0, 0], keypoints_2d_batch_cpn[:, 0, 0], keypoints_2d_batch_cpn_crop[:, 0, 0].clone(), keypoints_3d_gt, joint_token, action_token_batch, actions)
				keypoints_3d_pred_flip = model(images_batch[:, 1, 0], keypoints_2d_batch_cpn[:, 1, 0], keypoints_2d_batch_cpn_crop[:, 1, 0].clone(), keypoints_3d_gt, joint_token, action_token_batch, actions)
				keypoints_3d_pred_flip[..., 0] *= -1
				keypoints_3d_pred_flip[..., joints_left + joints_right, :] = keypoints_3d_pred_flip[..., joints_right + joints_left, :]
				keypoints_3d_pred = torch.mean(torch.cat((keypoints_3d_pred, keypoints_3d_pred_flip), dim=1), dim=1,
											   keepdim=True)

				del keypoints_3d_pred_flip

			else:
				keypoints_3d_pred = model(
										images_batch[:,0], 
										keypoints_2d_batch_cpn[:,0], 
										keypoints_2d_batch_cpn_crop[:,0], 
										keypoints_3d_gt[:,0], 
										joint_token, 
										action_token_batch, actions)

			
			if config.model.pose_net.save_inter_feat:
				model.module.pose_net.inter_feat['imagename'] += images_path
				print('imagename', len(model.module.pose_net.inter_feat['imagename']))

				model.module.pose_net.inter_feat['action'] += actions
				print('action', len(model.module.pose_net.inter_feat['action']))

			
			# calculate loss
			loss = criterion(keypoints_3d_pred, keypoints_3d_gt)

			if is_train:
				epoch_loss_3d_train += keypoints_3d_gt.shape[0] * loss.item()
				N += keypoints_3d_gt.shape[0]

				if not torch.isnan(loss):
					optimizer.zero_grad()
					loss.backward()

					if config.loss.grad_clip:
						torch.nn.utils.clip_grad_norm_(
							model.parameters(), 
							config.loss.grad_clip / config.train.pose_net_lr)

					optimizer.step()

			# # save answers for evalulation
			if not is_train:
				results['keypoints_gt'].append(keypoints_3d_gt.detach())    # (b, 17, 3)
				results['keypoints_3d'].append(keypoints_3d_pred.detach())    # (b, 17, 3)
				results['action'].append(actions)

			pbar.set_postfix(loss=loss.item())
			pbar.update(1)
			batch = prefetcher.next()

		pbar.close()
	
	# save intermediate features
	if config.model.pose_net.save_inter_feat:
		for k in model.module.pose_net.inter_feat.keys():
			if k not in ['imagename', 'action']:
				model.module.pose_net.inter_feat[k] = np.concatenate(model.module.pose_net.inter_feat[k], axis=0)
				print(k, model.module.pose_net.inter_feat[k].shape)	
		np.savez_compressed(f'inter_feat_hrnet.npz', data=model.module.pose_net.inter_feat)

	if is_train:
		return epoch_loss_3d_train / max(1, N), 0


	# Evaluate
	if not is_train:
		for term in ['keypoints_gt', 'keypoints_3d']:
			world_size = dist.get_world_size()
			results[term] = torch.cat(results[term])
			gathered_term = [torch.zeros_like(results[term]) for _ in range(world_size)]
			dist.all_gather(gathered_term, results[term])

		all_actions = []
		for action in results['action']:
			all_actions.extend(action)
		results['action'] = all_actions

		results['action'] = dataset_utils.all_gather_strings(results['action'], device)

	if master:
		if not is_train:
			result = dataset_utils.test_calculation(results['keypoints_gt'], 
													results['keypoints_3d'], 
													results['action'])

			return result


def init_distributed(args):
	if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
		return False

	torch.cuda.set_device(args.local_rank)

	assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
	assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	torch.manual_seed(args.seed)
	dist.init_process_group(backend="nccl", init_method="env://")

	return True


def match_name_keywords(n, name_keywords):
	out = False
	for b in name_keywords:
		if b in n:
			out = True
			break
	return out

def main(args):
	is_distributed = init_distributed(args)
	if args.debug:
		import debugpy
		rank = int(os.getenv('RANK', '0'))
		port = 8765 + rank
		debugpy.listen(port)
		print(f"Process {rank} waiting for debugger to attach on port {port}...")
		debugpy.wait_for_client()
	else:
		pass

	master = True
	if is_distributed and os.environ["RANK"]:
		master = int(os.environ["RANK"]) == 0
		rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
	else:
		rank = world_size = None

	if is_distributed:
		device = torch.device(args.local_rank)
	else:
		device = torch.device(0)

	config.train.n_iters_per_epoch = None   

	# image_encoder-specific configurations
	if args.image_encoder == 'hrnet_32':
		# Default setting
		config.model.pose_net.base_dim = 32

	elif args.image_encoder == 'hrnet_48':
		# Override the default setting
		config.model.image_encoder.checkpoint = 'data/pretrained/coco/pose_hrnet_w48_256x192.pth'
		config.model.image_encoder.STAGE2.NUM_CHANNELS = [48, 96]
		config.model.image_encoder.STAGE3.NUM_CHANNELS = [48, 96, 192]
		config.model.image_encoder.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
		config.model.pose_net.base_dim = 48

	model = MetaPose(config, device)

	# experiment
	experiment_dir, writer = None, None
	if master:
		experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
		[shutil.copy(src, experiment_dir) for src in glob.glob('models/*.py', recursive=True)]
		shutil.copy('train.py', experiment_dir)

		logger = Logger(log_path=experiment_dir).logger
		logger.parent = None
		logger.info("args: {}".format(args))
		logger.info("Number of available GPUs: {}".format(torch.cuda.device_count()))


	if config.model.image_encoder.init_weights:
		if args.image_encoder in ['hrnet_32', 'hrnet_48']:
			# Load HRNet
			ret = model.image_encoder.load_state_dict(torch.load(config.model.image_encoder.checkpoint, map_location=device), strict=False)

		# For HRNet, expected to see "_IncompatibleKeys(missing_keys=[], unexpected_keys=['final_layer.weight', 'final_layer.bias'])"
		print(ret)
		print(f"Loading {args.image_encoder} image_encoder from {config.model.image_encoder.checkpoint}")

	if args.eval:
		if args.image_encoder == 'hrnet_32':
			ckpt_path = 'checkpoint/best_epoch_hrnet_32.bin'

		elif args.image_encoder == 'hrnet_48':
			ckpt_path = 'checkpoint/best_epoch_hrnet_48.bin'

		checkpoint = torch.load(ckpt_path, map_location=device)['model']

		for key in list(checkpoint.keys()):
			new_key = key.replace("module.", "")
			checkpoint[new_key] = checkpoint.pop(key)

		ret = model.load_state_dict(checkpoint, strict=True)
		# Expected to see "<All keys matched successfully>"
		print(ret)
		print(f"Loading checkpoint from {ckpt_path}")

	elif config.model.init_weights :       # resume
		ckpt_path = config.model.checkpoint
		if ckpt_path != None and os.path.isfile(ckpt_path):
			state_dict = torch.load(ckpt_path, map_location=device)
			state_dict = state_dict['model']   #!
			for key in list(state_dict.keys()):
				new_key = key.replace("module.", "")
				state_dict[new_key] = state_dict.pop(key)
			try:
				model.load_state_dict(state_dict, strict=True)
			except:
				print('Warning: Final layer do not match!')
				for key in list(state_dict.keys()):
					if 'final_layer' in key:
						state_dict.pop(key)
				model.load_state_dict(state_dict, strict=True)

			if master:
				logger.info("Successfully loaded weights from {}".format(ckpt_path))
			del state_dict
		else:
			print("Failed loading weights as no checkpoint found at {}".format(ckpt_path))

	# sync bn in multi-gpus
	if args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model = model.to(device)

	# criterion
	criterion_class = {
		"MPJPE": MPJPE,
		"MSE": KeypointsMSELoss,
		"MSESmooth": KeypointsMSESmoothLoss,
		"MAE": KeypointsMAELoss
	}[config.loss.criterion]

	if config.loss.criterion == "MSESmooth":
		criterion = criterion_class(config.loss.mse_smooth_threshold).to(device)
	else:
		criterion = criterion_class().to(device)

	# optimizer
	lr = config.train.pose_net_lr
	lr_decay = config.train.pose_net_lr_decay

	if not args.eval:
		param_dicts = [
		{
			"params":
				[p for n, p in model.pose_net.named_parameters() if p.requires_grad],
			"lr": config.train.pose_net_lr,
		},
		]
		optimizer = optim.AdamW(param_dicts, weight_decay=0.1)
	if config.model.init_weights and ckpt_path != None:
		optimizer_path = ckpt_path
		if os.path.isfile(optimizer_path):
			optimizer_dict = torch.load(optimizer_path, map_location=device)
			lr = optimizer_dict['lr']
			epoch = optimizer_dict['epoch']
			if master:
				logger.info(f'start as epoch {epoch} now...')
			optimizer_dict = optimizer_dict['optimizer']
			optimizer.load_state_dict(optimizer_dict)

	# datasets
	if master:
		print("Loading data...")
	train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed=is_distributed, 
																	    is_train=not args.eval)

	if master:
		model_params = 0
		for parameter in model.pose_net.parameters():
			model_params += parameter.numel()
		logger.info("Trainable parameter count: " + str(model_params))

	# multi-gpu
	if is_distributed:
		model = DistributedDataParallel(model, device_ids=[device], output_device=args.local_rank) # , find_unused_parameters=True

	if not args.eval:
		# train loop
		min_loss = 100000
		
		for epoch in range(config.train.n_epochs):

			start_time = time.time()
			if train_sampler is not None:
				train_sampler.set_epoch(epoch)

			loss_3d_train = one_epoch_full(model, criterion, optimizer, config, train_dataloader, device, epoch, is_train=True, master=master)
			result = one_epoch_full(model, criterion, None, config, val_dataloader, device, epoch, is_train=False, master=master)

			if master:
				error_p1, error_p2 = dataset_utils.log_error(logger, result, is_train=True)
				loss_3d_train = loss_3d_train * 1000        

			if master:
				logger.info('[%d] time %.2f lr %.6f mse %.3f 3d_test_p1 %.1f 3d_test_p2 %.1f' % (
					epoch + 1,
					(time.time()-start_time) / 60.,
					lr,
					loss_3d_train,
					error_p1,
					error_p2))

				if error_p1 < min_loss:
					min_loss = error_p1
					logger.info("save best checkpoint")
					torch.save({
						'epoch': epoch + 1,
						'lr': lr,
						'optimizer': optimizer.state_dict(),
						'model': model.state_dict(),
					}, os.path.join(experiment_dir, "checkpoints/best_epoch.bin"))

			lr *= lr_decay
			for param_group in optimizer.param_groups:
				param_group['lr'] *= lr_decay

	else:
		result = one_epoch_full(model, criterion, None, config, val_dataloader, device, 0, is_train=False, master=master)

		if master:
			error_p1, error_p2 = dataset_utils.log_error(logger, result, is_train=False)
			logger.info(f"avg, p1:{error_p1}, p2:{error_p2}")
			logger.info("Done.")

if __name__ == '__main__':
	args = parse_args()
	main(args)