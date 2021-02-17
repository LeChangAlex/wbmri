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
from train_utils import extend_batch, get_validation_iwae
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


print(wandb.__version__)

np.random.seed(0)


def L1L(batch, target, reduction="none"):
    return  nn.L1Loss(reduction=reduction)(batch, target)

def L2L(batch, target, reduction="none"):
    return  nn.MSELoss(reduction=reduction)(batch, target)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class MyDataParallel(torch.nn.DataParallel):
	def __init__(self, model, device_ids):
		super(MyDataParallel, self).__init__(model, device_ids)

	def __getattr__(self, name):
		try:
			return super(MyDataParallel, self).__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)
# 

# class FairDataset(Dataset):
# 	# def __init__(self, path, transform, resolution=256):

# 	# # def __init__(self,csv_file,root_dir,transform=None):
# 	#     self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
# 	#     self.root_dir = path 
# 	#     self.transform = transform
	
# 	# def __len__(self):
# 	#     return (len(self.annotations))

# 	# def __getitem__(self,index):
# 	#     volume_name = os.path.join(self.root_dir,
# 	#     self.annotations.iloc[index,0])
# 	#     np_volume = np.load(volume_name)
# 	#     volume = Image.fromarray(np_volume)
# 	#     # annotations = self.annotations.iloc[index,0].as_matrix()
# 	#     # annotations = annotations.astype('float').reshape(-1,2)
# 	#     sample = volume#[np.newaxis, ...]

# 	#     if self.transform:
# 	#         sample = self.transform(sample)
		
# 	#     return sample
# 	def __init__(self, path, transform, reg, resolution=512, split="train", run=0, intensity=1, size=10, nodule_mask=0):


# 		self.nodule_mask = abs(nodule_mask)
# 		self.metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")
# 		if split == "train":
# 			self.metadata = self.metadata[self.metadata["train"] == 1]
# 		elif split == "test":
# 			self.metadata = self.metadata[self.metadata["train"] == 0]
# 		else:
# 			raise Exception("Invalid data split")



		

# 		data_mean = 0.175
# 		data_std = 0.17

# 		adjusted_intensity = intensity / data_std
# 		adjusted_max = (1 - data_mean) / data_std
		
# 		self.data = np.load("../mri_gan_cancer/data/chest_data.npy")
# 		# self.min_val = np.amin(self.data)
# 		# self.max_val = np.amax(self.data)
# 		# # print("mean:", np.mean(self.data.flatten()))
# 		# # print("std:", np.std(self.data.flatten()))
# 		# # print("max:", np.amax(self.data))



# 		# self.masks = np.zeros_like(self.data)

# 		# self.positions = pd.read_csv("../mri_gan_cancer/data/nodule_positions.csv")["run_{}".format(run)]

# 		# for i in range(self.data.shape[0]):
# 		#     positions = [int(n) for n in self.positions[i].split(",")]

# 		#     self.data[i] = self.insert_nodule(self.data[i], adjusted_intensity, size, positions)
# 		#     self.masks[i, positions[1] - self.nodule_mask: positions[1] + self.nodule_mask, positions[0] - self.nodule_mask: positions[0] + self.nodule_mask] = 1

# 		#     # print("nodules inserted")
# 		# self.data[self.data > adjusted_max] = adjusted_max
# 		# # print("clipped at", adjusted_max)
# 		# self.data = self.data[self.metadata["npy_idx"]]
# 		# cv2.imwrite("test/diseased_{}.png".format(run), (self.data[0] * data_std + data_mean) * 255) 
# 		# cv2.imwrite("test/diseased_{}_mask.png".format(run), (self.masks[0] * 255) )

# 		self.transform = transform
# 		self.reg = reg

	
# 	def insert_nodule(self, im, intensity, sigma, position):
		
# 		x, y = np.meshgrid(np.linspace(-25, 25, 50), np.linspace(-25, 25, 50))
# 		d = np.sqrt(x * x + y * y)
# 		nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

# 		nodule_x, nodule_y = position[0], position[1]

# 		im[nodule_y - 25: nodule_y + 25, nodule_x - 25: nodule_x + 25] += nodule * intensity



# 		return im



# 	def __len__(self):
# 		if self.reg:
# 			return self.metadata["patient_n"].unique().shape[0]
# 		return self.metadata.shape[0]


# 	def __getitem__(self,index):

# 		if not self.reg:

# 			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
# 			im = self.data[int(npy_idx)]
# 		else:
# 			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


# 			# print(patient_rows, index)
# 			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

# 			im = self.data[npy_idx - 1]
			

# 		volume = Image.fromarray(im)
# 		# annotations = self.annotations.iloc[index,0].as_matrix()
# 		# annotations = annotations.astype('float').reshape(-1,2)
# 		sample = volume#[np.newaxis, ...]

# 		if self.transform:
# 			sample = self.transform(sample)

# 		return sample

# 	def get_nodule_mask(self, index):

# 		if not self.reg:

# 			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
# 			mask = self.masks[int(npy_idx)]
# 		else:
# 			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


# 			# prior_network(patient_rows, index)
# 			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

# 			mask = self.masks[npy_idx - 1]
# 		return mask


# def train_discriminator(discriminator, ):



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

parser.add_argument('--exp', type=str)
parser.add_argument('--ckpt', type=str, default="-1")
parser.add_argument('--wandb', action='store_true')

parser.add_argument('--body_part', type=str, default="chest")
parser.add_argument('--scale_factor', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--d_lr', type=float, default=0.0001)

parser.add_argument('--delta', type=float, default=0)
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument('--adv_lambda', type=float, default=1)
parser.add_argument('--vaeac', action='store_true')
parser.add_argument('--vae', action='store_true')


parser.add_argument('--cvae', action='store_true')
parser.add_argument('--cond_noise', action='store_true')
parser.add_argument('--cond', action='store_true')
parser.add_argument('--discriminator', action='store_true')


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
parser.add_argument('--blur', action='store_true')


parser.add_argument('--dense', action='store_true')
parser.add_argument('--model', type=str, default="3")
parser.add_argument('--test_batch', action='store_true')
parser.add_argument('--validate_only', action='store_true')
parser.add_argument('--sliding_window', action='store_true')

parser.add_argument('--beta', type=float, default=5)
parser.add_argument('--loss', type=str, default="l2")

parser.add_argument('--real_labels', action='store_true')

parser.add_argument('--epochs', type=int, action='store', default=250,
					help='Number epochs to train VAEAC.')
parser.add_argument('--d_start', type=int, action='store', default=0)
parser.add_argument('--batch_size', type=int, action='store', default=64)

parser.add_argument('--z_dim', type=int, action='store', default=64)
parser.add_argument('--channels', type=int, action='store', default=1)
parser.add_argument('--out_channels', type=int, action='store', default=1)
parser.add_argument('--hw', type=int, action='store', default=16)

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
					default=1,
					help='Number of samples per object to estimate IWAE ' +
						 'on the validation set. Default: 25.')

parser.add_argument('--validations_per_epoch', type=int, action='store',
					default=1,
					help='Number of IWAE estimations on the validation set ' +
						 'per one epoch on the training set. Default: 5.')

parser.add_argument('--cond_y', action='store_true')
parser.add_argument('--cond_z', action='store_true')
parser.add_argument('--cond_age', action='store_true')
parser.add_argument('--cond_weight', action='store_true')
parser.add_argument('--cond_sex', action='store_true')

args = parser.parse_args()
wandb.init(project='VAEAC head3d', dir="/scratch/ssd001/home/lechang/GANomaly-PyTorch/vaeac/wandb", name=args.exp, mode="online" if args.wandb else "disabled")

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, so maybe it is time to set it
# to the number of CPU cores in the system.
num_workers = 32
kl_start = 0 
kl_anneal = 20
# import the module with the model networks definitions,
# optimization settings, and a mask generator
model_module = import_module(args.model_dir + '.model')


# import mask generator
mask_generator = model_module.mask_generator

if args.loss == "gaussian_loss":
	reconstruction_log_prob = GaussianLoss()
elif args.loss == "l1":
	reconstruction_log_prob = L1L
elif args.loss =="l2":
	reconstruction_log_prob = L2L




# CONDITION FEATURES
cond = []
if args.cond_age:
    cond.append("age")
if args.cond_sex:
    cond.append("sex")
if args.cond_weight:
    cond.append("weight")
if args.cond_z:
    cond.append("z")
if args.cond_y:
	cond.append("y")


# build VAEAC on top of the imported networks
# if args.vaeac:
# 	if args.dense:
# 		proposal_network, prior_network, generative_network = model_module.get_dense_networks(args.channels * 2, args.z_dim)
# 	else:
# 		proposal_network, prior_network, generative_network = model_module.get_networks(args.channels * 2, args.z_dim)

# 	model = VAEAC(
# 		model_module.reconstruction_log_prob,
# 		proposal_network, 
# 		prior_network,
# 		generative_network,
# 		mask_generator,
# 		channels=args.channels
# 	)
# elif args.cvae:
# 	print("TRAINING CVAE =======================")
# 	if args.dense:
# 		proposal_network, prior_network, generative_network = model_module.get_dense_networks(args.channels, args.z_dim)
# 	else:
# 		proposal_network, prior_network, generative_network = model_module.get_networks(args.channels, args.z_dim)

# 	model = CVAE(
# 		model_module.reconstruction_log_prob,
# 		proposal_network, 
# 		prior_network,
# 		generative_network,
# 		mask_generator,
# 		channels=args.channels
# 	)
# else:
# 	print("TRAINING VAE =======================")
# 	if args.hm:
# 		model = VAE(
# 			model_module.reconstruction_log_prob,
# 			model_module.encoder_network_hm,
# 			model_module.generative_network_hm,
# 			channels=args.channels
# 		)
# 	else:
# 		if args.dense:
# 			proposal_network, generative_network = model_module.get_dense_vae_networks(args.channels, args.z_dim)
# 		else:
# 			if args.hw == 16:
# 				n_outc = args.channels
# 				if args.loss == "gaussian_loss":
# 					n_outc *= 2

# 				proposal_network, generative_network, discriminator = model_module.get_vae_networks3(args.channels + len(cond), 
# 					n_outc, 
# 					args.z_dim, 
# 					metadata_channels=len(cond),
# 					discriminator=args.discriminator)

# 			elif args.hw == 32:
# 				proposal_network, generative_network = model_module.get_vae_networks4(args.channels + len(cond), args.out_channels, args.z_dim, metadata_channels=len(cond))

# 			elif args.hw == 64:
# 				proposal_network, generative_network = model_module.get_vae_networks5(args.channels + len(cond), args.out_channels, args.z_dim, metadata_channels=len(cond))

# 		print(args.hw)



# 		model = VAE(
# 			reconstruction_log_prob,
# 			proposal_network,
# 			generative_network,
# 			channels=args.channels
# 		)
n_outc = args.channels
if args.loss == "gaussian_loss":
	n_outc *= 2

if args.model == "sota":
	proposal_network, generative_network, discriminator = model_module.get_vae_networks(args.channels + len(cond), 
					n_outc, 
					args.z_dim, 
					metadata_channels=len(cond),
					discriminator=args.discriminator)

elif args.model == "3":
	print(args.channels, cond)
	proposal_network, generative_network, discriminator = model_module.get_vae_networks3(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(cond),
						discriminator=args.discriminator)
elif args.model == "4":
	print(args.channels, cond)
	proposal_network, generative_network, discriminator = model_module.get_vae_networks4(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(cond),
						discriminator=args.discriminator)
elif args.model == "5":
	print(args.channels, cond)
	proposal_network, generative_network, discriminator = model_module.get_vae_networks5(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(cond),
						discriminator=args.discriminator)
elif args.model == "6":
	print(args.channels, cond)
	proposal_network, generative_network, discriminator = model_module.get_vae_networks6(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(cond),
						discriminator=args.discriminator)
elif args.model == "7":
	print(args.channels, cond)
	proposal_network, generative_network, discriminator = model_module.get_vae_networks7(args.channels, 
						n_outc, 
						args.z_dim, 
						metadata_channels=len(cond),
						discriminator=args.discriminator)

	
cond_method = "input"
if args.model in ["7"]:
	cond_method = "resblock"

model = VAE(
	reconstruction_log_prob,
	proposal_network,
	generative_network,
	channels=args.channels,
	cond=cond,
	cond_method=cond_method
)

wandb.watch(model.encoder_network)
wandb.watch(model.generative_network)

if use_cuda:

	model = model.cuda()
	if args.discriminator:
		discriminator = discriminator.cuda()
print(args.lr)

# build optimizer and import its parameters
optimizer = model_module.optimizer(model.parameters(), lr=args.lr)

if args.discriminator:
	d_optimizer = model_module.optimizer(discriminator.parameters(), lr=args.d_lr)

batch_size = args.batch_size









# vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)
vlb_scale_factor = args.scale_factor ** 2


# load train and validation datasets
# train_dataset = load_dataset(args.train_dataset)
# validation_dataset = load_dataset(args.validation_dataset)

# AUGMENTATIONS =================
translate, scale, shear, rotate = None, None, None, 0
brightness, contrast, saturation, hue = 0, 0, 0, 0


if args.translate:
    translate = (0.1, 0.1)
    # translate = (0.1, 0.1)
if args.scale:
    scale = (0.9, 1.1)
if args.shear:
    shear = (-10, 10)
if args.rotate:
    rotate = (-10, 10)
if args.brightness:
    brightness = (0.8, 1.2)
if args.contrast:
    contrast = (0.8, 1.2)
if args.saturation:
    saturation = (0.9, 1.1)
if args.hue:
    hue = 0.1


# translate, scale, shear, rotate = (0.1, 0.1), (0.9, 1.1), (-15, 15), (-15, 15)
# brightness, contrast, saturation, hue = (0.1, 0.1), (0.9, 1.1), (0.9, 1.1), 0.1

tl = [transforms.ToPILImage()]

if args.flip:
	tl.append(transforms.RandomHorizontalFlip(p=0.5))
if args.blur:
	tl.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1, 2))], p=0.5))

tl += [
	transforms.RandomAffine(rotate, translate, scale, shear),
	transforms.ColorJitter(brightness, contrast, saturation, hue),
	transforms.Resize((256, 256)),
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

train_dataset = BodyPartDataset(body_part=args.body_part, split="train", transform=transform, n_slices=args.channels, cond_noise=args.cond_noise, cond=cond, 
						test_batch=args.test_batch,
						sliding_window=args.sliding_window)

val_tl = [
	transforms.ToPILImage(),
	transforms.Resize((256, 256)),
	transforms.ToTensor()
]

val_transform = transforms.Compose(val_tl)

validation_dataset = BodyPartDataset(split="train" if args.test_batch else "test", 
	all_slices=True, 
	n_slices=args.channels, 
	transform=val_transform, 
	body_part=args.body_part, 
	return_n=True, 
	cond=cond,
	nodule=True,
	test_batch=args.test_batch, 
	real_labels=args.real_labels)




# validation_dataset = BodyPartDataset(body_part=args.body_part, split="test", transform=val_transform, n_slices=args.channels, cond=cond)

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
						shuffle=True, drop_last=not args.test_batch,
						num_workers=num_workers)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size * 8,
							shuffle=False, drop_last=False,
							num_workers=num_workers)




# number of batches after which it is time to do validation
validation_batches = ceil(len(dataloader) / args.validations_per_epoch)


print(f"Using sigma={args.sigma} and C={args.max_grad_norm}")

if args.delta > 0:

	privacy_engine = PrivacyEngine(
	    model,
	    batch_size=batch_size,
	    sample_size=len(train_dataset),
	    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
	    noise_multiplier=args.sigma,
	    max_grad_norm=args.max_grad_norm,
	)
	model = module_modification.convert_batchnorm_modules(model).cuda()
	inspector = DPModelInspector()
	print(inspector.validate(model),"--------------")

	privacy_engine.attach(optimizer)


start_epoch = 0
# a list of validation IWAE estimates
validation_iwae = []
# a list of running variational lower bounds on the train set
train_vlb = []
# the length of two lists above is the same because the new
# values are inserted into them at the validation checkpoints only

# load the last checkpoint, if it exists
if exists(join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp))):
# if exists(join(args.model_dir, args.ckpt)):
	
	print("loading checkpoint ============================")
	location = 'cuda' if use_cuda else 'cpu'

	checkpoint = torch.load(join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp)),
							map_location=location)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	validation_iwae = checkpoint['validation_iwae']
	train_vlb = checkpoint['train_vlb']
	start_epoch = checkpoint['epoch']



	print("loaded")
# enc = list(model.encoder_network.children())
# enc.insert(22, nn.AvgPool2d(2, 2))
# model.encoder_network = nn.Sequential(*enc)

# dec = list(model.generative_network.children())
# dec.insert(21, nn.Upsample(scale_factor=2))
# model.generative_network = nn.Sequential(*dec)# feats.insert(8, nn.Identity())
# model.features = nn.Sequential(feats)



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
step = 0


for epoch in range(start_epoch, args.epochs):


	channel_loss = torch.zeros(args.channels)

	model.train()
	iterator = dataloader
	avg_vlb = 0
	if verbose:
		print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
		iterator = tqdm(iterator)

	# one epoch

	if not args.validate_only:
		for i, (batch, metadata) in enumerate(iterator):

			# if i > 1:#
			# 	continue
			step += 1
			# the time to do a checkpoint is at start and end of the training
			# and after processing validation_batches batches

			# if batch size is less than batch_size, extend it with objects
			# from the beginning of the dataset
			# batch = extend_batch(batch, dataloader, batch_size)
			# generate mask and do an optimizer step over the mask and the batch
			# mask = mask_generator(batch)
			

				


			optimizer.zero_grad()
			if use_cuda:
				batch, metadata = batch.cuda(), metadata.cuda()

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
				beta = (epoch - kl_start) / kl_anneal * args.beta

				beta = min(beta, args.beta)
				print("beta:", beta)


			# print(batch.shape)

			vlb, rec_loss, kl, recs = nn.parallel.data_parallel(model, (batch, args.hm, beta, metadata), device_ids=range(1))

			vlb, rec_loss, kl, recs = vlb.mean(), rec_loss, kl.mean(), recs.detach()
			# print("c")
			# print("vlb", vlb)


			iter_rec_loss = rec_loss.sum(-1).sum(-1).sum(0).cpu()
			# print(rec_loss.shape, iter_rec_loss.shape, "------")
			channel_loss += iter_rec_loss
			

			# (-vlb / vlb_scale_factor).backward()
			loss = vlb 
			if args.discriminator and epoch >= args.d_start:
				d_pred = nn.parallel.data_parallel(discriminator, recs, device_ids=range(1))
				adv_loss = nn.functional.softplus(-d_pred).mean()

				loss += args.adv_lambda * adv_loss
				wandb.log({
					"vae adv loss" : adv_loss.item(),
					"step_vae": step
				})
			optimizer.zero_grad()
			loss.backward()

			# print("d")

			optimizer.step()
			# update running variational lower bound average
			avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

			# print("e")

	            



			# train discriminator
			if args.discriminator:
				# real predictions
				real_d_pred = nn.parallel.data_parallel(discriminator, batch, device_ids=range(1))
				real_d_loss = nn.functional.softplus(-real_d_pred)

				rec_d_pred = nn.parallel.data_parallel(discriminator, recs[:, 0:1], device_ids=range(1))
				rec_d_loss = nn.functional.softplus(rec_d_pred)


				real_d_loss, rec_d_loss = real_d_loss.mean(), rec_d_loss.mean()
				d_loss = real_d_loss + rec_d_loss

				d_optimizer.zero_grad()
				d_loss.backward()
				d_optimizer.step()
	            
				wandb.log({
					"real d loss" : real_d_loss.item(),
					"rec d loss" : rec_d_loss.item(),
					"d loss" : d_loss.item(),
					"step_d": step
				})

		if verbose:
			iterator.set_description('Train VLB: %g' % avg_vlb)

		if args.delta > 0:
			epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)

		else:
			epsilon = 0

	# print(epsilon,"--------------------")
	with torch.no_grad():
		#### make this return a dict
		metrics_dict = get_validation_iwae(val_dataloader,
									   batch_size, 
									   model,
									   args.validation_iwae_num_samples,
									   verbose, middle_mask=args.hm,
									   compute_auroc=not args.test_batch)

									   # compute_auroc=(epoch + 1) % 20 == 0)

		validation_iwae.append(metrics_dict["val_iwae"])
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

		if max(validation_iwae[::-1]) <= metrics_dict["val_iwae"]:
			src_filename = join(args.model_dir, 'last_checkpoint_{}.tar'.format(args.exp))
			dst_filename = join(args.model_dir, 'best_checkpoint_{}.tar'.format(args.exp))
			copy(src_filename, dst_filename + '.bak')
			replace(dst_filename + '.bak', dst_filename)

		if epoch % 20 == 0:
			torch.save({
				'epoch': epoch,
				'model_swatate_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'validation_iwae': metrics_dict["val_iwae"],
				'train_vlb': train_vlb,
			},  'vaeac_models/{}_{}.tar'.format(args.exp, epoch))
			

		if verbose:
			print(file=stderr)
			print(file=stderr)

	train_images = []
	radius = args.channels // 2
	for i in range(15):
		# if metrics_dict["nodule_recs"][i].shape[radius] == 1:
		train_canvas = torch.cat((batch[i][radius].cpu(), 
			recs[i][radius].cpu(), 
			rec_loss[i][radius].cpu() * 5), 1)

		train_images.append(train_canvas)


	train_images = torch.cat(train_images, 0).clamp(0, 1)

	train_images = train_images.unsqueeze(0).repeat(3, 1, 1).cpu().detach().numpy() * 255
	train_images = np.rollaxis(train_images.astype(np.uint8), 0, 3)
	train_images = Image.fromarray(train_images)#.save("recs/{}_{}.png".format(args.exp, epoch))


	images = []
	for i in range(1 if args.test_batch else min(20, metrics_dict["batch"].shape[0])):
		# if metrics_dict["nodule_recs"][i].shape[radius] == 1:
		canvas = torch.cat((metrics_dict["batch"][i][radius].cpu(), 
			metrics_dict["recs"][i][radius].cpu(), 
			metrics_dict["rec_loss_t"][i][radius].cpu() * 5, 
			metrics_dict["batch_nodule"][i][radius].cpu(), 
			metrics_dict["batch_labels"][i][radius].cpu(), 
			metrics_dict["nodule_recs"][i][radius].cpu(), 
			metrics_dict["nodule_rec_loss_t"][i][radius].cpu() * 5), 1)

		images.append(canvas)

	print(metrics_dict["batch"][i][radius].cpu().max(),"-----------")
	images = torch.cat(images, 0).clamp(0, 1)
	images = images.unsqueeze(0).repeat(3, 1, 1).cpu().numpy() * 255
	images = np.rollaxis(images.astype(np.uint8), 0, 3)
	images = Image.fromarray(images)#.save("recs/{}_{}.png".format(args.exp, epoch))
	
	if not args.validate_only:
		# canvas = torch.cat((imgs[i][0], masks[i][0], recs[i][0]), 1)
		log_d = {
			"train_loss/train_vlb": avg_vlb,
			"train_loss/train_rl": rec_loss.sum(-1).sum(-1).sum(-1).mean().item(),
			"train_loss/train_kl": kl.item(),
			"val_loss/val_iwae": metrics_dict["val_iwae"],
			"val_loss/val_rl": metrics_dict["rec_loss"],
			"val_loss/val_kl": metrics_dict["kls"].mean(),
	        "reconstructions/validation":[wandb.Image(images)],
	        "reconstructions/train":[wandb.Image(train_images)],

	        "metrics/auroc": metrics_dict["auroc"],
	        "metrics/auprc": metrics_dict["auprc"],
	        "metrics/sensitivity": metrics_dict["sensitivity"],
	        "metrics/specificity": metrics_dict["specificity"],
	        "epsilon": epsilon,
	        "epoch": epoch
		}

		for i in range(args.channels):
			log_d["channel/train_channel_loss_{}".format(i)] = channel_loss[i].item()
			log_d["channel/val_channel_loss_{}".format(i)] = metrics_dict["channel_loss"][i].item()

	else:
		log_d = {
	        "reconstructions":[wandb.Image(images)],
	        "metrics/auroc": metrics_dict["auroc"],
	        "metrics/auprc": metrics_dict["auprc"],
			"val_loss/val_iwae": metrics_dict["val_iwae"],
	        "epoch": epoch,
			"val_loss/val_rl": metrics_dict["rec_loss"],
			"val_loss/val_kl": metrics_dict["kls"].mean(),
		}			
	wandb.log(log_d)
