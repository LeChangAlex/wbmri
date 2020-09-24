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

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC, VAE
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image

from source.wbmri_crops import BodyPartDataset

import wandb

class MyDataParallel(torch.nn.DataParallel):
	def __init__(self, model, device_ids):
		super(MyDataParallel, self).__init__(model, device_ids)

	def __getattr__(self, name):
		try:
			return super(MyDataParallel, self).__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)


class FairDataset(Dataset):
	# def __init__(self, path, transform, resolution=256):

	# # def __init__(self,csv_file,root_dir,transform=None):
	#     self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
	#     self.root_dir = path 
	#     self.transform = transform
	
	# def __len__(self):
	#     return (len(self.annotations))

	# def __getitem__(self,index):
	#     volume_name = os.path.join(self.root_dir,
	#     self.annotations.iloc[index,0])
	#     np_volume = np.load(volume_name)
	#     volume = Image.fromarray(np_volume)
	#     # annotations = self.annotations.iloc[index,0].as_matrix()
	#     # annotations = annotations.astype('float').reshape(-1,2)
	#     sample = volume#[np.newaxis, ...]

	#     if self.transform:
	#         sample = self.transform(sample)
		
	#     return sample
	def __init__(self, path, transform, reg, resolution=512, split="train", run=0, intensity=1, size=10, nodule_mask=0):


		self.nodule_mask = abs(nodule_mask)
		self.metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")
		if split == "train":
			self.metadata = self.metadata[self.metadata["train"] == 1]
		elif split == "test":
			self.metadata = self.metadata[self.metadata["train"] == 0]
		else:
			raise Exception("Invalid data split")



		

		data_mean = 0.175
		data_std = 0.17

		adjusted_intensity = intensity / data_std
		adjusted_max = (1 - data_mean) / data_std
		
		self.data = np.load("../mri_gan_cancer/data/chest_data.npy")
		# self.min_val = np.amin(self.data)
		# self.max_val = np.amax(self.data)
		# # print("mean:", np.mean(self.data.flatten()))
		# # print("std:", np.std(self.data.flatten()))
		# # print("max:", np.amax(self.data))



		# self.masks = np.zeros_like(self.data)

		# self.positions = pd.read_csv("../mri_gan_cancer/data/nodule_positions.csv")["run_{}".format(run)]

		# for i in range(self.data.shape[0]):
		#     positions = [int(n) for n in self.positions[i].split(",")]

		#     self.data[i] = self.insert_nodule(self.data[i], adjusted_intensity, size, positions)
		#     self.masks[i, positions[1] - self.nodule_mask: positions[1] + self.nodule_mask, positions[0] - self.nodule_mask: positions[0] + self.nodule_mask] = 1

		#     # print("nodules inserted")
		# self.data[self.data > adjusted_max] = adjusted_max
		# # print("clipped at", adjusted_max)
		# self.data = self.data[self.metadata["npy_idx"]]
		# cv2.imwrite("test/diseased_{}.png".format(run), (self.data[0] * data_std + data_mean) * 255) 
		# cv2.imwrite("test/diseased_{}_mask.png".format(run), (self.masks[0] * 255) )

		self.transform = transform
		self.reg = reg

	
	def insert_nodule(self, im, intensity, sigma, position):
		
		x, y = np.meshgrid(np.linspace(-25, 25, 50), np.linspace(-25, 25, 50))
		d = np.sqrt(x * x + y * y)
		nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

		nodule_x, nodule_y = position[0], position[1]

		im[nodule_y - 25: nodule_y + 25, nodule_x - 25: nodule_x + 25] += nodule * intensity



		return im



	def __len__(self):
		if self.reg:
			return self.metadata["patient_n"].unique().shape[0]
		return self.metadata.shape[0]


	def __getitem__(self,index):

		if not self.reg:

			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
			im = self.data[int(npy_idx)]
		else:
			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


			# print(patient_rows, index)
			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

			im = self.data[npy_idx - 1]
			

		volume = Image.fromarray(im)
		# annotations = self.annotations.iloc[index,0].as_matrix()
		# annotations = annotations.astype('float').reshape(-1,2)
		sample = volume#[np.newaxis, ...]

		if self.transform:
			sample = self.transform(sample)

		return sample

	def get_nodule_mask(self, index):

		if not self.reg:

			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
			mask = self.masks[int(npy_idx)]
		else:
			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


			# prior_network(patient_rows, index)
			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

			mask = self.masks[npy_idx - 1]
		return mask


parser = ArgumentParser(description='Train VAEAC to inpaint.')

parser.add_argument('--model_dir', type=str, action='store', required=True,
					help='Directory with model.py. ' +
						 'It must be a directory in the root ' +
						 'of this repository. ' +
						 'The checkpoints are saved ' +
						 'in this directory as well. ' +
						 'If there are already checkpoints ' +
						 'in the directory, the training procedure ' +
						 'is resumed from the last checkpoint ' +
						 '(last_checkpoint.tar).')

parser.add_argument('--exp', type=str)
parser.add_argument('--body_part', type=str)
parser.add_argument('--scale_factor', type=float, default=1)
parser.add_argument('--lr', type=float)
parser.add_argument('--vaeac', action='store_true')
parser.add_argument('--vae', action='store_true')

parser.add_argument('--translate', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--scale', action='store_true')
parser.add_argument('--shear', action='store_true')

parser.add_argument('--brightness', action='store_true')
parser.add_argument('--contrast', action='store_true')
parser.add_argument('--saturation', action='store_true')
parser.add_argument('--hue', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--hm', action='store_true')

parser.add_argument('--dense', action='store_true')


parser.add_argument('--epochs', type=int, action='store', required=True,
					help='Number epochs to train VAEAC.')
parser.add_argument('--z_dim', type=int, action='store', required=True)
parser.add_argument('--channels', type=int, action='store')

# parser.add_argument('--train_dataset', type=str, action='store',
#                     required=True,
#                     help='Dataset of images for training VAEAC to inpaint ' +
#                          '(see load_datasets function in datasets.py).')

# parser.add_argument('--validation_dataset', type=str, action='store',
#                     required=True,
#                     help='Dataset of validation images for VAEAC ' +
#                          'log-likelihood IWAE estimate ' +
#                          '(see load_datasets function in datasets.py).')

parser.add_argument('--validation_iwae_num_samples', type=int, action='store',
					default=25,
					help='Number of samples per object to estimate IWAE ' +
						 'on the validation set. Default: 25.')

parser.add_argument('--validations_per_epoch', type=int, action='store',
					default=1,
					help='Number of IWAE estimations on the validation set ' +
						 'per one epoch on the training set. Default: 5.')


args = parser.parse_args()
wandb.init(project='VAEAC head3d', dir="/scratch/hdd001/home/lechang", name=args.exp)

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, so maybe it is time to set it
# to the number of CPU cores in the system.
num_workers = 32
kl_start = 100 if args.vae else 0
kl_anneal = 100 if args.vae else 1
# import the module with the model networks definitions,
# optimization settings, and a mask generator
model_module = import_module(args.model_dir + '.model')


# import mask generator
mask_generator = model_module.mask_generator


# build VAEAC on top of the imported networks
if args.vaeac:
	if args.dense:
		proposal_network, prior_network, generative_network = model_module.get_dense_networks(args.channels, args.z_dim)
	else:
		proposal_network, prior_network, generative_network = model_module.get_networks(args.channels, args.z_dim)

	model = VAEAC(
		model_module.reconstruction_log_prob,
		proposal_network, 
		prior_network,
		generative_network,
		mask_generator,
		channels=args.channels
	)

else:
	print("TRAINING VAE =======================")
	if args.hm:
		model = VAE(
			model_module.reconstruction_log_prob,
			model_module.encoder_network_hm,
			model_module.generative_network_hm,
			channels=args.channels
		)
	else:
		if args.dense:
			proposal_network, generative_network = model_module.get_dense_vae_networks(args.channels, args.z_dim)
		else:
			proposal_network, generative_network = model_module.get_vae_networks(args.channels, args.z_dim)

		model = VAE(
			model_module.reconstruction_log_prob,
			proposal_network,
			generative_network,
			channels=args.channels
		)





# wandb.watch(model.proposal_network)
# wandb.watch(model.generative_network)
# wandb.watch(model.prior_network)

if use_cuda:
	model = model.cuda()

# build optimizer and import its parameters
optimizer = model_module.optimizer(model.parameters(), lr=args.lr)
batch_size = model_module.batch_size
# vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)
vlb_scale_factor = args.scale_factor ** 2


# load train and validation datasets
# train_dataset = load_dataset(args.train_dataset)
# validation_dataset = load_dataset(args.validation_dataset)

# AUGMENTATIONS =================
translate, scale, shear, rotate = None, None, None, 0
brightness, contrast, saturation, hue = 0, 0, 0, 0


if args.translate:
    translate = (0.2, 0.2)
    # translate = (0.1, 0.1)
if args.scale:
    scale = (0.8, 1.2)
if args.shear:
    shear = (-20, 20)
if args.rotate:
    rotate = (-20, 20)
if args.brightness:
    brightness = (0.1, 0.1)
if args.contrast:
    contrast = (0.9, 1.1)
if args.saturation:
    saturation = (0.9, 1.1)
if args.hue:
    hue = 0.1
# translate, scale, shear, rotate = (0.1, 0.1), (0.9, 1.1), (-15, 15), (-15, 15)
# brightness, contrast, saturation, hue = (0.1, 0.1), (0.9, 1.1), (0.9, 1.1), 0.1

tl = [transforms.ToPILImage()]

if args.flip:
	tl.append(transforms.RandomHorizontalFlip(p=0.5))

tl += [
	transforms.RandomAffine(rotate, translate, scale, shear),
	transforms.ColorJitter(brightness, contrast, saturation, hue),
	transforms.ToTensor()
]
transform = transforms.Compose(tl)

# ===========================


# translate, scale, shear, rotate = None, None, None, 0

# if args.translate:
# 	translate, scale, shear, rotate = (0.1, 0.1), None, None, 0



# transform = transforms.Compose([
# 		transforms.ToPILImage(),
# 		transforms.RandomAffine(rotate, translate, scale, shear),
# 		transforms.ToTensor()
# ])

train_dataset = BodyPartDataset(body_part=args.body_part, split="train", transform=transform, n_slices=args.channels)


validation_dataset = BodyPartDataset(body_part=args.body_part, split="test", n_slices=args.channels)

# transform = transforms.Compose([
# 		transforms.RandomAffine(rotate, translate, scale, shear),
# 		transforms.ToTensor()
# ])

# train_dataset = FairDataset("", transform, True, split="train")

# val_transform = transforms.Compose([
# 		transforms.ToTensor()
# ])
# validation_dataset = FairDataset("", val_transform, True, split="test")


# build dataloaders on top of datasets
dataloader = DataLoader(train_dataset, batch_size=batch_size,
						shuffle=True, drop_last=True,
						num_workers=num_workers)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
							shuffle=False, drop_last=False,
							num_workers=num_workers)

# number of batches after which it is time to do validation
validation_batches = ceil(len(dataloader) / args.validations_per_epoch)

start_epoch = 0
# a list of validation IWAE estimates
validation_iwae = []
# a list of running variational lower bounds on the train set
train_vlb = []
# the length of two lists above is the same because the new
# values are inserted into them at the validation checkpoints only

# load the last checkpoint, if it exists
if exists(join(args.model_dir, 'last_checkpoint_{}.tar', args.exp)):
	print("loading checkpoint ============================")
	location = 'cuda' if use_cuda else 'cpu'

	checkpoint = torch.load(join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp)),
							map_location=location)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	validation_iwae = checkpoint['validation_iwae']
	train_vlb = checkpoint['train_vlb']
	start_epoch = checkpoint['epoch']

# Makes checkpoint of the current state.
# The checkpoint contains current epoch (in the current run),
# VAEAC and optimizer parameters, learning history.
# The function writes checkpoint to a temporary file,
# and then replaces last_checkpoint.tar with it, because
# the replacement operation is much more atomic than
# the writing the state to the disk operation.
# So if the function is interrupted, last checkpoint should be
# consistent.
def make_checkpoint():
	filename = join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp))
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'validation_iwae': validation_iwae,
		'train_vlb': train_vlb,
	}, filename + '.bak')

	replace(filename + '.bak', filename)


# main train loop
for epoch in range(start_epoch, args.epochs):

	iterator = dataloader
	avg_vlb = 0
	if verbose:
		print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
		iterator = tqdm(iterator)

	# one epoch
	for i, batch in enumerate(iterator):

		# the time to do a checkpoint is at start and end of the training
		# and after processing validation_batches batches


		# if batch size is less than batch_size, extend it with objects
		# from the beginning of the dataset
		batch = extend_batch(batch, dataloader, batch_size)
		# generate mask and do an optimizer step over the mask and the batch
		# mask = mask_generator(batch)



		optimizer.zero_grad()
		if use_cuda:
			batch = batch.cuda()
			# mask = mask.cuda()
		# vlb = model.batch_vlb(batch, mask).mean()
		# rec_params, mask, proposal, prior = model(batch)
		# r = model(batch)
		# print(r)
		# print(len(r))
		# vlb = model.compute_loss(rec_params, mask, proposal, prior).mean()

		# print("b")
		beta = 0
		if epoch >= kl_start:
			beta = (epoch - kl_start) / kl_anneal

			beta = min(beta, 1)


		vlb, rec_loss, kl = nn.parallel.data_parallel(model, (batch, args.hm, beta), device_ids=range(2))

		vlb, rec_loss, kl = vlb.mean(), rec_loss.mean(), kl.mean()
		# print("c")
		print("vlb", vlb)

		# (-vlb / vlb_scale_factor).backward()
		(-vlb).backward()

		# print("d")

		optimizer.step()
		# update running variational lower bound average
		avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

		# print("e")


	
            
	if verbose:
		iterator.set_description('Train VLB: %g' % avg_vlb)
	with torch.no_grad():
		val_iwae, val_rec_loss, val_kl, recs, masks, imgs = get_validation_iwae(val_dataloader, mask_generator,
									   batch_size, model,
									   args.validation_iwae_num_samples,
									   epoch, verbose, middle_mask=args.hm)
		val_rec_loss, val_kl = val_rec_loss.mean(), val_kl.mean()

		validation_iwae.append(val_iwae)
		train_vlb.append(avg_vlb)
		# print(recs.shape, masks.shape, imgs.shape, "=========")

		make_checkpoint()

		# if current model validation IWAE is the best validation IWAE
		# over the history of training, the current checkpoint is copied
		# to best_checkpoint.tar
		# copying is done through a temporary file, i. e. firstly last
		# checkpoint is copied to temporary file, and then temporary file
		# replaces best checkpoint, so even best checkpoint should be
		# consistent even if the script is interrupted
		# in the middle of copying

		if max(validation_iwae[::-1]) <= val_iwae:
			src_filename = join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp))
			dst_filename = join(args.model_dir, 'best_checkpoint_{}.tar'.format(args.exp))
			copy(src_filename, dst_filename + '.bak')
			replace(dst_filename + '.bak', dst_filename)

		if epoch % 100 == 0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'validation_iwae': validation_iwae,
				'train_vlb': train_vlb,
			},  'vaeac_models/{}_{}.tar'.format(args.exp, epoch))
			

		if verbose:
			print(file=stderr)
			print(file=stderr)

	# pred_slice = 0 if args.hm else 1
	# if epoch % 20 == 0:
	images = []
	for i in range(10):
		canvas = torch.cat((imgs[i][0], masks[i][0], recs[i][0]), 1)

		images.append(canvas)


	images = torch.cat(images, 0)

	images = images.unsqueeze(0).repeat(3, 1, 1).cpu().numpy() * 255
	images = np.rollaxis(images.astype(np.uint8), 0, 3)
	images = Image.fromarray(images)#.save("recs/{}_{}.png".format(args.exp, epoch))
	
	# canvas = torch.cat((imgs[i][0], masks[i][0], recs[i][0]), 1)
	wandb.log({
		"train_vlb": avg_vlb,
		"train_rl": rec_loss.item(),
		"train_kl": kl.item(),
		"val_iwae": val_iwae,
		"val_rl": val_rec_loss,
		"val_kl": val_kl,
        "reconstructions":[wandb.Image(images)]

	})
