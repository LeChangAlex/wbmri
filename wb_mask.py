
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

def L1L(batch, target, reduction="none"):
    return  nn.L1Loss(reduction=reduction)(batch, target)

def L2L(batch, target, reduction="none"):
    return  nn.MSELoss(reduction=reduction)(batch, target)




parser = ArgumentParser(description='Train VAEAC to inpaint.')

parser.add_argument('--model_dir', type=str, action='store', default="vaeac_models",
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

parser.add_argument('--chest_ckpt', type=str, default="-1")
parser.add_argument('--legs_ckpt', type=str, default="-1")

parser.add_argument('--discriminator', action='store_true')

args = parser.parse_args()


model_module = import_module(args.model_dir + '.model')

chest_cond = ["z"]
real_labels = False
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

checkpoint = torch.load(join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.chest_ckpt)),
							map_location="cuda")

chest_model.load_state_dict(checkpoint['model_state_dict'])

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
	sliding_window=True)


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

checkpoint = torch.load(join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.legs_ckpt)),
							map_location="cuda")

legs_model.load_state_dict(checkpoint['model_state_dict'])

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
	sliding_window=True)




for i in range(len(chest_dataset)):
	full_im, chest_mask, (l_x, u_x, l_y, u_y), label, nodule_vol = wb_mask(chest_dataset, i, chest_model)

	skimage.io.imsave("wb_masks/nodule_im_{}.png".format(i), nodule_vol.reshape(-1, 256))

	anomaly_mask = torch.zeros_like(full_im)
	anomaly_mask[:, :anomaly_mask.shape[1] // 2, l_x: u_x] += chest_mask

	full_im_label = torch.zeros_like(full_im)
	full_im_label[:, :anomaly_mask.shape[1] // 2, l_x: u_x] = label

	_, legs_mask, _, _, _ = wb_mask(legs_dataset, i, legs_model)



	anomaly_mask[:, anomaly_mask.shape[1] // 2:, :] += legs_mask * 0.5 ############# to remove


	heatmap = full_im.clone().unsqueeze(0).repeat(3, 1, 1, 1)
	heatmap[0] += anomaly_mask	

	# canvas = torch.cat((full_im.reshape(-1, full_im.shape[2]), anomaly_mask.reshape(-1, full_im.shape[2])


	canvas = torch.cat((full_im.reshape(-1, full_im.shape[2]), 
		anomaly_mask.reshape(-1, full_im.shape[2]), 
		full_im_label.reshape(-1, full_im.shape[2])), 1)

	# anomaly_mask = anomaly_mask.reshape()



	# anomaly_mask = anomaly_mask.permute(1, 2, 3, 0)


	skimage.io.imsave("wb_masks/test.png", canvas)

	# using tuple unpacking for multiple Axes
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(50,50))
	ax1.imshow(full_im.reshape(-1, full_im.shape[2]), cmap="gray")
	ax2.imshow(chest_mask[25] * 5, cmap="gray")
	ax3.imshow(anomaly_mask[25])

	fig.savefig('wb_masks/mask_{}.png'.format(i))
