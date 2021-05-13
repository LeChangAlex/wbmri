
from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from prob_utils import normal_parse_params, GaussianLoss

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae, wb_mask
from VAEAC import VAE
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image

from source.wbmri_crops import BodyPartDataset
from skimage import exposure, restoration, transform

import math
# from opacus import PrivacyEngine
# from opacus.utils import module_modification

# from opacus.dp_model_inspector import DPModelInspector

import os

import skimage.io
import matplotlib.pyplot as plt
# from skimage.transform import resize
# from train_utils import fast_auprc, fast_auc2, dice, post_proc_vol, get_validation_score

def L1L(batch, target, reduction="none"):
    return  nn.L1Loss(reduction=reduction)(batch, target)

def L2L(batch, target, reduction="none"):
    return  nn.MSELoss(reduction=reduction)(batch, target)
    

def resize(tensor, dim):


	out = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=dim)

	return out.squeeze(0)


parser = ArgumentParser(description='Train VAEAC to inpaint.')

parser.add_argument('--model_dir', type=str, action='store', default="vae_models",
					help='Directory with model.py. ' +
						 'It must be a directory in the root ' +
						 'of this repository. ' +
						 'The checkpoints are saved ' +
						 'in this directory as well. ' +
						 'If there are already checkpoints ' +
						 'in the directory, the training procedure ' +
						 'is resumed from the last checkpoint ' +
						 '(last_checkpoint.tar).')
parser.add_argument('--chest_model', type=str, default="6")
parser.add_argument('--legs_model', type=str, default="6")

parser.add_argument('--channels', type=int, action='store', default=5)
parser.add_argument('--z_dim', type=int, action='store', default=64)

parser.add_argument('--chest_ckpt', type=str, default="best_checkpoint_flch5b1.tar") #"condyzswfixedy_180.tar")
parser.add_argument('--legs_ckpt', type=str, default="legs_180.tar")
parser.add_argument('--mode', type=str, default="save")

parser.add_argument('--discriminator', action='store_true')
parser.add_argument('--gradcam', action='store_true')

args = parser.parse_args()


model_module = import_module(args.model_dir + '.model')

# chest_cond = ["z"]
chest_cond = []

real_labels = True
save_conv_features = True
n_outc = args.channels

chop = False

if args.chest_model == "6":
	proposal_network, generative_network, discriminator = model_module.get_vae_networks6(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(chest_cond),
						discriminator=args.discriminator)
elif args.chest_model == "7":
	proposal_network, generative_network, discriminator = model_module.get_vae_networks7(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(chest_cond),
						discriminator=args.discriminator)


	
chest_cond_method = "input"
if args.chest_model in ["7"]:
	chest_cond_method = "resblock"


# chest model

chest_model = VAE(
	L2L,
	proposal_network,
	generative_network,
	channels=args.channels,
	cond=chest_cond,
	cond_method=chest_cond_method,
	# save_conv_features=save_conv_features
)

checkpoint = torch.load(join(args.model_dir, args.chest_ckpt),
							map_location="cuda" if torch.cuda.is_available() else "cpu")

# chest_model.load_state_dict(checkpoint['model_state_dict'])
chest_model.load_state_dict(checkpoint['model_state_dict'])

# if torch.cuda.is_available():
# 	chest_model = chest_model.cuda()
# else:
# 	print("running on cpu")


# Legs model
legs_cond = []

if args.legs_model == "6":
	proposal_network, generative_network, discriminator = model_module.get_vae_networks6(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(legs_cond),
						discriminator=args.discriminator)
elif args.legs_model == "7":
	proposal_network, generative_network, discriminator = model_module.get_vae_networks7(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(legs_cond),
						discriminator=args.discriminator)

# legs_cond_method = "input"
# if args.legs_model in ["7"]:
# 	legs_cond_method = "resblock"


# legs_model = VAE(
# 	L2L,
# 	proposal_network,
# 	generative_network,
# 	channels=args.channels,
# 	cond=legs_cond,
# 	cond_method=legs_cond_method
# )

# checkpoint = torch.load(join(args.model_dir, args.legs_ckpt),
# 							map_location="cuda")

# # chest_model.load_state_dict(checkpoint['model_state_dict'])
# legs_model.load_state_dict(checkpoint['model_swatate_dict'])

# legs_model = legs_model.cuda()

pp = True

if args.mode == "mask":
	# chest_dataset = BodyPartDataset(split="test", 
	# all_slices=True, 
	# n_slices=5, 
	# return_n=True, 
	# cond=chest_cond,
	# nodule=True,
	# body_part="chest",
	# real_labels=real_labels,
	# store_full_volume=True,
	# sliding_window=True,
	# store_full_labels=True,
	# n_volume_slices=40)

	chest_dataset = BodyPartDataset(split="test", 
	all_slices=True, 
	n_slices=5, 
	return_n=True, 
	cond=chest_cond,
	nodule=True,
	body_part="chest",
	real_labels=True,
	store_full_volume=True,
	sliding_window=True,
	store_full_labels=True,
	n_volume_slices=40,
	chop=chop)

	plot = True
	flat_losses = []
	flat_labels = []


	for i in tqdm(range(4, len(chest_dataset))):

		chest_model.reset_conv_features()
		full_im, full_label, chest_mask, (l_x, u_x, l_y, u_y), label, nodule_vol, recs, vn = wb_mask(chest_dataset, i, chest_model, full_res_mask=False, gradcam=args.gradcam)
		chest_model.merge_batch_conv_features()

		print(chest_model.conv_features[-1].shape)
		fig, ax = plt.subplots(8, 8)
		

		latent_map = chest_model.conv_features[-1].detach().cpu().numpy()
		# for each batch element/ordered window
		for j in range(68, 88):#latent_map.shape[0]):
			
			# print(len(chest_model.conv_features))

			# for each feature map
			for k in range(latent_map.shape[1]):

				ax[k // 8][k % 8].imshow(latent_map[j][k], vmin=-3, vmax=3)

			plt.savefig("feature_maps/{}_{}_{}.png".format(i, j, k))

		# if vn in [540]:
		# 	continue	
		# if pp:
			# chest_mask = post_proc(chest_mask, recs, nodule_vol)

		# print(chest_mask.shape, label.shape, nodule_vol.shape, recs.shape, "===========")
		# skimage.io.imsave("wb_masks/nodule_im_{}.png".format(i), nodule_vol.reshape(-1, 256))



		red_label = nodule_vol.clone().unsqueeze(0).repeat(3, 1, 1, 1)

		red_label[0, label > 0] = 1

		red_label = red_label.permute(1, 2, 3, 0)

		# nodule_idx = np.argwhere(label.sum(-1).sum(-1)).reshape(-1)
		nodule_idx = range(label.shape[0])

		if plot:
			for n in nodule_idx:

				fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 15))
				ax1.imshow(nodule_vol[n], cmap="gray")
				ax2.imshow(red_label[n])
				ax3.imshow(chest_mask[n], cmap="gray")
				ax4.imshow(recs[n], cmap="gray")

				fig.savefig('wb_masks/chest_mask_{}_{}.png'.format(vn, n))

				fig.clf()
				plt.clf()
				plt.close()

		flat_losses.append(chest_mask.reshape(-1))
		flat_labels.append(label.reshape(-1))



	losses = torch.cat(flat_losses)
	labels = torch.cat(flat_labels)
	print("auprc", fast_auprc(losses, labels))
	print("auroc", fast_auc2(losses, labels))

	for i in range(100):
		t = i / 100
		d = dice(losses, labels)
		print("dice", t, dice)


	# full res
	# anomaly_mask = torch.zeros_like(full_im)
	# anomaly_mask[:, :anomaly_mask.shape[1] // 2, l_x: u_x] += chest_mask

	# _, _, legs_mask, _, _, _, recs, vn = wb_mask(legs_dataset, i, legs_model, full_res_mask=True)


	# anomaly_mask[:, anomaly_mask.shape[1] // 2:, :] += legs_mask * 0.5 ############# to remove


	# red_label = full_im.clone()
	# red_label = full_im.unsqueeze(0).repeat(3, 1, 1, 1)
	# # print(red_label.shape, (full_label>0).shape)


	# red_label[0, full_label > 0] = 1 

	# red_label = red_label.permute(1, 2, 3, 0)

	# nodule_idx = np.argwhere(full_label.sum(-1).sum(-1)).reshape(-1)
elif args.mode == "validate":
	chest_dataset = BodyPartDataset(split="test", 
	all_slices=True, 
	n_slices=5, 
	return_n=True, 
	cond=chest_cond,
	nodule=True,
	body_part="chest",
	real_labels=real_labels,
	store_full_volume=True,
	sliding_window=True,
	store_full_labels=True,
	n_volume_slices=40)


	# legs_dataset = BodyPartDataset(split="test", 
	# 	all_slices=True, 
	# 	n_slices=5, 
	# 	return_n=True, 
	# 	cond=legs_cond,
	# 	nodule=True,
	# 	body_part="legs",
	# 	real_labels=real_labels,
	# 	store_full_volume=True,
	# 	sliding_window=True,
	# 	store_full_labels=True)

	chest_model.eval()
	metrics_dict = get_validation_score(chest_dataset,
			chest_model,
			post_proc=True,
			plot_fn="",
			return_losses=True)

	print("mdice", metrics_dict["mdice"])

	print(args.chest_ckpt, "auprc", metrics_dict["auprc"])
	print(metrics_dict["losses"].shape, metrics_dict["labels"].shape)



	for j in range(100):
	
		d = dice(metrics_dict["mdice"][j])

		print("dice", t, d)


elif args.mode == "sample":
	bs = 1
	
	z = torch.normal(torch.zeros((bs, 32, 16, 16)), torch.ones((bs, 32, 16, 16))).cuda()

	for h in range(50):

		wh = -h / 50 
		metadata = torch.zeros((bs, 2)).cuda() 

		metadata[:, 1] += wh
		radius = args.channels // 2


		sample = chest_model.sample(metadata, z).clip(0, 1).detach().cpu()[:, radius]

		for i in range(bs):

			skimage.io.imsave("samples/sample_{}_{}.png".format(wh, i), sample[i])

elif args.mode == "save":

	chest_dataset = BodyPartDataset(split="test", 
	all_slices=True, 
	n_slices=5, 
	return_n=True, 
	cond=chest_cond,
	nodule=True,
	body_part="chest",
	real_labels=True,
	store_full_volume=True,
	sliding_window=True,
	store_full_labels=True,
	n_volume_slices=40,
	data_dir="chop")

	legs_dataset = BodyPartDataset(split="test", 
	all_slices=True, 
	n_slices=5, 
	return_n=True, 
	cond=chest_cond,
	nodule=True,
	body_part="chest",
	real_labels=True,
	store_full_volume=True,
	sliding_window=True,
	store_full_labels=True,
	n_volume_slices=40,
	data_dir="chop")


	plot = True
	flat_losses = []
	flat_labels = []


	for i in tqdm(range(4, len(chest_dataset))):

		full_im, full_label, chest_mask, (l_x, u_x, l_y, u_y), label, nodule_vol, recs, vn = wb_mask(chest_dataset, i, chest_model, full_res_mask=True)#, gradcam=args.gradcam)



		# resized_chest_mask = resize(chest_mask, (full_im.shape[0], full_im.shape[1] // 2, u_x - l_x))


		os.makedirs("masks", exist_ok=True)


		# red_label = nodule_vol.clone().unsqueeze(0).repeat(3, 1, 1, 1)

		# red_label[0, label > 0] = 1

		# red_label = red_label.permute(1, 2, 3, 0)

		# nodule_idx = np.argwhere(label.sum(-1).sum(-1)).reshape(-1)
		# nodule_idx = range(label.shape[0])



		# full res
		anomaly_mask = torch.zeros_like(full_im)
		anomaly_mask[:, :anomaly_mask.shape[1] // 2, l_x: u_x] += chest_mask

		# _, _, legs_mask, _, _, _, recs, vn = wb_mask(legs_dataset, i, legs_model, full_res_mask=True)

		# anomaly_mask[:, anomaly_mask.shape[1] // 2:, :] += legs_mask * 0.5 ############# to remove
	


		# resized_anomaly_mask = resize(anomaly_mask, (int(anomaly_mask.shape[1] * 256 / anomaly_mask.shape[2]), 256))
		os.makedirs("masks/volume_{}".format(vn), exist_ok=True)

		for j in range(full_im.shape[0]):
			skimage.io.imsave("masks/volume_{}/slice_{}.png".format(vn, j), anomaly_mask[j])



