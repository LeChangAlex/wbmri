
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
import wandb
from opacus import PrivacyEngine
from opacus.utils import module_modification

from opacus.dp_model_inspector import DPModelInspector

import skimage.io
import matplotlib.pyplot as plt

from train_utils import fast_auprc, fast_auc2, post_proc

def L1L(batch, target, reduction="none"):
    return  nn.L1Loss(reduction=reduction)(batch, target)

def L2L(batch, target, reduction="none"):
    return  nn.MSELoss(reduction=reduction)(batch, target)




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
parser.add_argument('--chest_model', type=str, default="7")
parser.add_argument('--legs_model', type=str, default="6")

parser.add_argument('--channels', type=int, action='store', default=5)
parser.add_argument('--z_dim', type=int, action='store', default=64)

parser.add_argument('--chest_ckpt', type=str, default="condyzswfixedy_180.tar")
parser.add_argument('--legs_ckpt', type=str, default="legs_180.tar")

parser.add_argument('--discriminator', action='store_true')

args = parser.parse_args()


model_module = import_module(args.model_dir + '.model')

chest_cond = ["z", "y"]
real_labels = True
n_outc = args.channels

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
	cond_method=chest_cond_method
)

checkpoint = torch.load(join(args.model_dir, args.chest_ckpt),
							map_location="cuda")

# chest_model.load_state_dict(checkpoint['model_state_dict'])
chest_model.load_state_dict(checkpoint['model_swatate_dict'])

chest_model = chest_model.cuda()

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

legs_cond_method = "input"
if args.legs_model in ["7"]:
	legs_cond_method = "resblock"


legs_model = VAE(
	L2L,
	proposal_network,
	generative_network,
	channels=args.channels,
	cond=legs_cond,
	cond_method=legs_cond_method
)

checkpoint = torch.load(join(args.model_dir, args.legs_ckpt),
							map_location="cuda")

# chest_model.load_state_dict(checkpoint['model_state_dict'])
legs_model.load_state_dict(checkpoint['model_swatate_dict'])

legs_model = legs_model.cuda()

legs_dataset = BodyPartDataset(split="test", 
	all_slices=True, 
	n_slices=5, 
	return_n=True, 
	cond=legs_cond,
	nodule=True,
	body_part="legs",
	real_labels=real_labels,
	store_full_volume=True,
	sliding_window=True,
	store_full_labels=True)



# def validate():
plot = False
flat_losses = []
flat_labels = []


for i in tqdm(range(len(chest_dataset))):


	full_im, full_label, chest_mask, (l_x, u_x, l_y, u_y), label, nodule_vol, recs, vn = wb_mask(chest_dataset, i, chest_model, full_res_mask=False)
	# if vn in [540]:
	# 	continue	
	if post_proc:
		chest_mask = post_proc(chest_mask, recs, nodule_vol)

	# print(chest_mask.shape, label.shape, nodule_vol.shape, recs.shape, "===========")
	# skimage.io.imsave("wb_masks/nodule_im_{}.png".format(i), nodule_vol.reshape(-1, 256))



	red_label = nodule_vol.clone().unsqueeze(0).repeat(3, 1, 1, 1)

	red_label[0, label > 0] = 1 

	red_label = red_label.permute(1, 2, 3, 0)

	nodule_idx = np.argwhere(label.sum(-1).sum(-1)).reshape(-1)


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

	print(label.max(), "label max")

print("auprc", fast_auprc(torch.cat(flat_losses), torch.cat(flat_labels)))
print("auroc", fast_auc2(torch.cat(flat_losses), torch.cat(flat_labels)))


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