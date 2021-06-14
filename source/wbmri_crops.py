	
import os
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import torchvision
import math
from PIL import Image
from tqdm import tqdm

import random
from skimage import io
import pandas as pd
from sklearn import metrics
from skimage import exposure, restoration, transform

import ast

pd.set_option('display.max_columns', None)

# class Volume():


# 	def __init__(self, volume, l_x, u_x, l_y, u_y):
# 		"""
# 		vol: full resolution volume

# 		"""

# 		self.volume = volume

# 		self.l_x = l_x
# 		self.u_x = u_x
# 		self.l_y = l_y
# 		self.u_y = u_y



# 	def get_lowres(self):
def read_slice(vn, s):
	try:

		volume_slice = io.imread("/datasets/wbmri/preproc5/volume_{}/{}.png".format(vn, s), as_gray=True)
		return volume_slice
	except:
		pass

	try:

		volume_slice = io.imread("/datasets/wbmri/preproc5_bright/volume_{}/{}.png".format(vn, s), as_gray=True)
		return volume_slice
	except:
		pass


def read_png_volume(dir, transform=None):

	vol = []
	for i in range(len(os.listdir(dir))):
		a = torch.from_numpy(io.imread(os.path.join(dir, "{}.png".format(i)), as_gray=True)).unsqueeze(0)

		# a = a[:a.shape[1]]
		# if transform:
		# 	a = transform(a)
		vol.append(a)

	return torch.cat(vol, 0)


		

def get_crop_coords(fn, return_dict=False):

	l = []

	# f = open("/datasets/wbmri/crop_coords/chest_coords_preproc5.txt".format(mod, bp))
	# f = open("/datasets/wbmri/crop_coords/{}_{}_filt.txt".format(mod, bp))

	# f = open("/datasets/wbmri/crop_coords/{}_{}.txt".format(mod, bp))
	# f = open("../../mri_gan_cancer/preprocess/chest_headless_coords.csv")
	f = open(fn)


	# coords_line = f.readline().split(" ")
	# coords_line = [tmp.strip("(),\n") for tmp in coords_line]
	# template_x_size, template_y_size = int(coords_line[1]), int(coords_line[2])
	d = {}
	for line in f:


		line = [int(float(u.strip())) for u in line.split(",")]
		# line = [tmp.strip("(),\n") for tmp in line]
		

		# i = int(line[0])

		# x = line[2]
		# y = line[1]

		# u_x = x + line[4]
		# u_y = y + line[3]

		i = int(line[0])

		x = line[1]
		y = line[2]

		u_x = x + line[3]
		u_y = y + line[4]



		# print(i, x, y, u_x, u_y)
		l.append((i, x, u_x, y, u_y))
		d[i] = (x, u_x, y, u_y)

	if return_dict:
		return d
	return l

def insert_nodule_volume(tensor, intensity=0.7, sigma=6, n_nodule=1, multi_ch=False, nodule_thickness=3):
	"""
	Takes an int tensor returns an int tensor
	"""

	vol = tensor.clone().float() / 255

	mask = torch.zeros_like(vol)


	for i in range(0, tensor.shape[0], nodule_thickness):
		vol[i:i+nodule_thickness], mask[i:i+nodule_thickness] = \
		insert_nodule_slice(vol[i:i+nodule_thickness], intensity, sigma, n_nodule, multi_ch)

	
	return (vol * 255).int(), (mask * 255).int()

def insert_nodule_slice(tensor, intensity=0.7, sigma=6, n_nodule=1, multi_ch=False):

	vol = tensor.clone()

	masks = []

	# try:
	coords = np.indices((1, vol.shape[1] - 32, vol.shape[2] - 32))

	tissue_int = 0.1

	coords[1] += 16
	coords[2] += 16

	radius = vol.shape[0] // 2

	tissue_idx1 = vol[radius, coords[1], coords[2]] > tissue_int
	tissue_idx2 = vol[radius, coords[1], coords[2]] < 0.4
	tissue_idx = torch.logical_and(tissue_idx1, tissue_idx2)
	# tissue_idx = np.logical_and(vol[coords[0], coords[1], coords[2]] > tissue_int, vol[coords[0], coords[1], coords[2]] < 0.7)
	# print(tissue_idx.shape)
	a = tissue_idx.cpu().numpy()
	tissue_coords = coords[:, a]

	mask = torch.zeros_like(vol)
	if tissue_coords.shape[-1] == 0:
		return vol, mask

	idx = np.random.randint(tissue_coords.shape[-1])

	loc = tissue_coords[:, idx]

	x, y = np.meshgrid(np.linspace(-15, 15, 32), np.linspace(-15, 15, 32))
	d = np.sqrt(x * x + y * y)
	nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))#[np.newaxis, ...]

	vol[:, loc[1] - 16: loc[1] + 16, loc[2] - 16: loc[2] + 16] += torch.from_numpy(nodule * intensity)
	mask[:, loc[1] - 16: loc[1] + 16, loc[2] - 16: loc[2] + 16] += torch.from_numpy((nodule > 0.3).astype(np.float))
	# except:
	# 	mask = torch.zeros_like(vol)
	# 	masks.append(mask.unsqueeze(0))
	
	vol[vol > 1] = 1
	# masks = torch.cat(masks, 0)
	
	return vol, mask




def insert_nodule(tensor, intensity=0.7, sigma=6, n_nodule=1, multi_ch=False):

	vol = tensor.clone()

	masks = []

	# try:
	coords = np.indices((vol.shape[0], vol.shape[1] - 32, vol.shape[2] - 32))

	tissue_int = 0.1

	coords[1] += 16
	coords[2] += 16

	tissue_idx1 = vol[coords[0], coords[1], coords[2]] > tissue_int
	tissue_idx2 = vol[coords[0], coords[1], coords[2]] < 0.4
	tissue_idx = torch.logical_and(tissue_idx1, tissue_idx2)
	# tissue_idx = np.logical_and(vol[coords[0], coords[1], coords[2]] > tissue_int, vol[coords[0], coords[1], coords[2]] < 0.7)
	# print(tissue_idx.shape)
	a = tissue_idx.cpu().numpy()
	tissue_coords = coords[:, a]

	mask = torch.zeros_like(vol)
	if tissue_coords.shape[-1] == 0:
		return vol, mask

	idx = np.random.randint(tissue_coords.shape[-1])

	loc = tissue_coords[:, idx]

	x, y = np.meshgrid(np.linspace(-15, 15, 32), np.linspace(-15, 15, 32))
	d = np.sqrt(x * x + y * y)
	nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))#[np.newaxis, ...]

	vol[loc[0] - 1: loc[0] + 2, loc[1] - 16: loc[1] + 16, loc[2] - 16: loc[2] + 16] += torch.from_numpy(nodule * intensity)
	mask[loc[0] - 1: loc[0] + 2, loc[1] - 16: loc[1] + 16, loc[2] - 16: loc[2] + 16] += torch.from_numpy((nodule > 0.3).astype(np.float))
	# except:
	# 	mask = torch.zeros_like(vol)
	# 	masks.append(mask.unsqueeze(0))
	
	vol[vol > 1] = 1
	# masks = torch.cat(masks, 0)
	
	return vol, mask


def real_label_volume(label_paths, im):

	mask = None
	if not isinstance(label_paths, list) and pd.isna(label_paths):
		return torch.zeros_like(im)

	for i in range(len(label_paths)):
		s, path = label_paths[i]
		sl = torch.from_numpy(io.imread(path))

		if mask is None:

			mask = torch.zeros((im.shape[0], sl.shape[0], sl.shape[1]))
		
		else:
			mask[s] = sl

	return mask


def get_label_df():
	return pd.read_pickle("/datasets/wbmri/preproc_label_df.pkl")

def find_vol_path(row):

	n = row["volume_n"]

	path1 = "/datasets/wbmri/preproc5/volume_{}".format(n)
	path2 = "/datasets/wbmri/preproc5_bright/volume_{}".format(n)

	if os.path.exists(path1):
		return path1
	elif os.path.exists(path2):
		return path2
	return None



class BodyPartDataset(Dataset):

	def __init__(self, split="train", body_part="head", transform=None, vol_size=None, 
		all_slices=False, n_slices=1, real_labels=False, cond_features=False, cond_noise=False, 
		return_n=False, cond=[], nodule=False, test_batch=False, multi_ch_nodule=False, sliding_window=False,
		store_full_volume=False, return_volumes=False, store_full_labels=False, n_volume_slices=40, data_dir="vector", gen_training_set=False, n_rows=None, prefix=None):

		def exists(i):
			
			return os.path.exists("/datasets/wbmri/headless_preproc_crops/chest_{}.png".format(i)) or \
			os.path.exists("/datasets/wbmri/preproc5_bright_crops/chest_{}.png".format(i))
			# return os.path.exists("im_recs_w_nodules/volume_{}".format(i))

		def try_open(fn):
			try:
				# print("aaa")
				return np.load(fn)
				# print(np.sum(a))
			except:
				print(fn)
				return
		def count_slices(self, row):
			# print(row["volume_path"])
			# return len(os.listdir(row["volume_path"]))
			# return len(os.listdir(self.volumes))

			n = row["volume_n"]
			return len(os.listdir(os.path.join(self.data_path, "volume_{}".format(n))))
		self.nodule = nodule
		self.return_n = return_n
		self.store_full_volume = store_full_volume
		self.data_dir = data_dir
		if data_dir == "default" or "chop":
			self.data_path = "volumes/"
			
		elif data_dir == "vector" or data_dir == "reader":
			self.data_path = "/datasets/wbmri/preproc5/"

		elif prefix == "recs":
			self.data_path = "recs"

		elif prefix == "nodule_im":
			self.data_path = "nodule_im"
		
		# elif prefix == "reader":
		# 	self.data_path = "reader"



		self.all_slices = all_slices

		self.split = split
		self.body_part = body_part
		self.real_labels = real_labels
		self.sliding_window = sliding_window

		self.return_volumes = return_volumes
		self.store_full_labels = store_full_labels
		# TODO: Implement function which returns df
		# with cols [volume_n, age, sex, weight]
		if data_dir == "default":
			self.metadata = pd.read_csv("metadata.csv")
			
		elif data_dir == "chop":
			self.metadata = pd.read_csv("chop_metadata.csv")
		
		elif data_dir == "reader":
			# self.metadata = pd.read_csv("reader_set.csv")
			self.metadata = pd.read_pickle("reader_set.pkl")

			print(self.metadata, "========")
		
		elif data_dir == "vector":
			# self.metadata = pd.read_csv("/datasets/wbmri/practice_set.csv")
			self.metadata = pd.read_csv("/datasets/wbmri/anonym_wbmri_metadata.csv")
			self.metadata = self.metadata.fillna("0")


			# TODO: Implement function which returns df 
			# with cols [volume_n, list of tuples of format (slice number, path to slice label)]
			self.label_df = get_label_df()



			self.metadata = self.metadata.merge(self.label_df, on="volume_n", how="left")



			self.metadata[pd.isna(self.metadata["label_paths"])]
		
			if split == "train":
				self.metadata = self.metadata[pd.isna(self.metadata["label_paths"])]
			elif split == "test":
				self.metadata = self.metadata[np.logical_not(pd.isna(self.metadata["label_paths"]))]
			

			# add volume_path column
			self.metadata["volume_path"] = self.metadata.apply(find_vol_path, 1)

			# only include volumes with > 20 slices
			self.metadata = self.metadata[self.metadata["dim1"].apply(int) > 20]
			
			# remove rows with no paths
			self.metadata = self.metadata[self.metadata["volume_path"].apply(lambda x: x is not None)]
			
			# remove rows with no coords
			self.metadata = self.metadata[self.metadata.apply(lambda row: exists(row["volume_n"]), axis=1)]

		
		if data_dir == "default" or "chop":

			# self.metadata = pd.read_csv("coords.csv")

			# self.coords_dict = get_crop_coords("chop_coords.csv", return_dict=True)
			self.coords_dict = None
	
		elif data_dir == "vector" or data_dir == "reader":
			self.coords_dict = get_crop_coords("preprocess/chest_headless_coords.csv", return_dict=True)
		

		# add n_slice column 
		
			self.metadata["n_slices"] = self.metadata.apply(self.count_slices, 1)


		if n_rows:
			self.metadata = self.metadata.reset_index()[:n_rows]

		else:

			self.metadata = self.metadata.reset_index()


		self.metadata = self.metadata[self.metadata.apply(lambda row: exists(row["volume_n"]), axis=1)]
		

		if gen_training_set:

			training_set = self.metadata[pd.isna(self.metadata["label_paths"])]
			nodule_set = self.metadata[np.logical_not(pd.isna(self.metadata["label_paths"]))]
			# print(self.metadata, "======")
			# print(training_set)
			# print(nodule_set)

			# random_set = self.metadata.sample(n=40)
			random_set = pd.concat((training_set.sample(n=40), nodule_set.sample(n=10)), 0)
			random_set = random_set.sample(frac=1)
			random_set.to_pickle("reader_set.pkl")


		self.multi_ch_nodule = multi_ch_nodule
		self.transform = transform
		self.n_slices = n_slices
		self.vol_size = vol_size
		self.cond_features = cond_features
		self.cond_noise = cond_noise
		self.cond = cond
		self.volumes = []
		self.nodule_volumes = []
		self.full_labels = []
		self.labels = []
		self.volume_n = []

		self.slices = []
		self.test_batch = test_batch
		self.radius = n_slices // 2
		self.n_volume_slices = n_volume_slices

		# correct_crop_list = os.listdir("/datasets/wbmri/headless_preproc_crops/")


		# print(self.z_mean, self.z_std)
		self.full_volumes = []
		self.masks = []

		
		for index in tqdm(range(len(self.metadata))): # to remove

			# print("b========================")

			# print(self.metadata["volume_n"])
			n = self.metadata["volume_n"][index]
			if self.data_path == "chop":
				sex = self.metadata["sex"][index]
				weight = self.metadata["weight"][index]
				age = self.metadata["age"][index]

				n_slices = self.metadata["n_slices"][index]
				spacing = float(self.metadata["slicespacing"][index]) 
				thickness = float(self.metadata["slicethickness"][index])

			sex, weight, age, n_slices, spacing, thickness = 0,0,0,0,0,0
			if spacing == 0:
				spacing = 6

			if thickness == 0:
				thickness = 6

			slice_spacing = spacing + thickness




			# slice_dir = "/datasets/wbmri/slice_png_chest_headless/volume_{}/".format(n)
			# n_slices = len([name for name in os.listdir(slice_dir) if name[-4:] == ".png"])


			# # im = read_png_volume("/datasets/wbmri/slice_png_abdomen_ni/volume_{}/".format(n), self.transform)



			# store column of possible windows
			# im = read_png_volume("/datasets/wbmri/headless_preproc/volume_{}/".format(n), self.transform)
			# im = read_png_volume("/datasets/wbmri/preproc5/volume_{}/".format(n), self.transform)
			
			# vol_path = self.metadata["volume_path"][index]
			# if vol_path != "/"
			vol_path = os.path.join(self.data_path, "volume_{}".format(n))

			im = read_png_volume(vol_path, self.transform)
			full_height = im.shape[1]
			

			if self.coords_dict:
				l_x, u_x, l_y, u_y = self.coords_dict[n]
			else:
				l_x, u_x = im.shape[2] // 4, im.shape[2] // 4 * 3


			full_im = None

			if self.store_full_volume:
				full_im = im.clone()
				self.full_volumes.append(full_im)


			if self.body_part == "legs":
				# left_im = im[:, im.shape[1] // 2:, :im.shape[2] // 2]
				# right_im = im[:, im.shape[1] // 2:, im.shape[2] // 2:]
				im =  im[:, im.shape[1] // 2:]

				# nodule_im, label = insert_nodule_volume(im)


				size_h = int(im.shape[1] * 512 / im.shape[2])

				im = im.unsqueeze(0)
				im = nn.functional.interpolate(im.float(), size=(size_h, 512), mode="bicubic", align_corners=False).squeeze(0).int()
				
				left_im = im[:, :, :256]
				right_im = im[:, :, 256:]

				# left_im = left_im.unsqueeze(0)
				# left_im = nn.functional.interpolate(left_im.float(), size=(size_h, 256), mode="bicubic", align_corners=False).squeeze(0).int()
			
				# right_im = right_im.unsqueeze(0)
				# right_im = nn.functional.interpolate(right_im.float(), size=(size_h, 256), mode="bicubic", align_corners=False).squeeze(0).int()

				# im = (left_im, right_im)

			elif self.body_part == "chest":

				if self.sliding_window:
					im = im[:, :im.shape[1] // 2, l_x: u_x]

					# nodule_im, label = insert_nodule_volume(im)


					size_h = int(im.shape[1] * 256 / im.shape[2])
					# print(size_h, im.shape, "========================================")
					
					im = im.unsqueeze(0)

					im = nn.functional.interpolate(im.float(), size=(size_h, 256), mode="bicubic", align_corners=False).squeeze(0).int()
				

				else:
					im = im[:, l_y: l_y + u_x - l_x, l_x: u_x].unsqueeze(0)


					im = nn.functional.interpolate(im.float(), size=(256, 256), mode="bicubic", align_corners=False).squeeze(0).int()


			self.volumes.append(im)

			# print(self.real_labels)


			if self.nodule:
				if self.real_labels:
					nodule_im = im

					if data_dir == "chop":
						label = torch.zeros_like(im)
					else:
						label = real_label_volume(self.metadata["label_paths"][index], im)


					# if not self.store_full_volume:					
					# 	h = 0
						
					# 	l_x, u_x, l_y, u_y = self.coords_dict[n]
						
					# 	mask = mask[:, l_x: u_x, l_y: u_y]
					# 	mask = mask[:, h :h + mask.shape[2]].unsqueeze(0)

					# 	mask = nn.Upsample(size=(256, 256), mode="nearest")(mask).squeeze(0)
					# 	mask[mask >= 1] = 1

					# store resized chest label
					if self.store_full_labels:
						full_label = label.clone()
						self.full_labels.append(full_label)

						# print(n, full_label.shape)
					if self.body_part == "chest":

						nodule_im = im

						if self.sliding_window:
							# print(label.shape, l_x, u_x, "=======================")
							label = label[:, :label.shape[1] // 2].unsqueeze(0)

							label = nn.functional.interpolate(label.float(), size=(size_h, 256), mode="nearest").squeeze(0).int()
						# else:
							# label = label[:, :label.shape[1] // 2, l_x: u_x].unsqueeze(0)

							# label = nn.functional.interpolate(label.float(), size=(size_h, 256), mode="nearest").squeeze(0).int()

					elif self.body_part == "legs":
						


						label = label[:, label.shape[1] // 2:].unsqueeze(0)
						label = nn.functional.interpolate(label.float(), size=(size_h, 512), mode="nearest").squeeze(0).int()

						# left_label = label[:, label.shape[1] // 2:, :label.shape[2] // 2]
						# right_label = label[:, label.shape[1] // 2:, label.shape[2] // 2:]
				

						# left_label = left_label.unsqueeze(0)
						# left_label = nn.functional.interpolate(left_label.float(), size=(size_h, 256), mode="nearest").squeeze(0).int()
					
						# right_label = right_label.unsqueeze(0)
						# right_label = nn.functional.interpolate(right_label.float(), size=(size_h, 256), mode="nearest").squeeze(0).int()

						# nodule_im = im
						# label = (left_label, right_label)





				else:
					# if self.body_part == "chest":
					# 	nodule_im, label = insert_nodule_volume(im)

					# elif self.body_part == "legs":
					nodule_im, label = insert_nodule_volume(im)
					print(label.sum())

						# left_nodule_im, left_mask = insert_nodule_volume(left_im)
						# right_nodule_im, right_mask = insert_nodule_volume(right_im)

						# nodule_im = (left_nodule_im, right_im)
						# mask = (left_mask, right_mask)

				self.nodule_volumes.append(nodule_im)
				self.labels.append(label)

			self.slices += [(index, n, s, sex, weight, age, n_slices, s * slice_spacing, full_height) for s in range(max(n_slices - self.n_volume_slices, 1 + self.radius), n_slices - 2 - self.radius)]


		z_list = [z_coord for index, n, s, sex, weight, age, n_slices, z_coord, full_h in self.slices]
		self.z_mean, self.z_std = np.mean(z_list), np.std(z_list)



		print(self.metadata)
		print(self.z_mean, self.z_std)
		print("aaaaa")

	def __len__(self):

		if self.split == "test":
		# if self.sliding_window:
			# print("size of dataset:", len(self.volumes))
			# return len(self.slices)
			print("cccccc")
			return len(self.volumes)

		# print("size of training dataset:", len(self.volumes))

		return len(self.slices)
		# if self.return_volumes:
		# 	return len(self.volumes)
		# if self.sliding_window:
		# 	return len(self.slices)

		# print(len(self.slices))
		# if self.test_batch:
		# 	return 1
		# return len(self.slices)
		# # print(len(self.metadata))
		# # return len(self.metadata)


	def get_features(self, sex, weight, age, s, n_slices, z_coord, height_to_chest, im_height):

		if sex == "M":
			sex = 1
		elif sex == "F":
			sex = 0
			


		# print(s, n_slices)
		age = float(age)
		weight = float(weight)
		# z = ((n_slices - s) - self.z_mean) / self.z_std  # how many slices from the back
		z = (z_coord - self.z_mean) / self.z_std  # how many slices from the back

		if self.cond_noise and self.split == "train":
			age += np.random.normal(0, 0.5)
			weight += np.random.normal(0, 2)
			z += np.random.normal(0, 0.5)

		age = (age - 12.33) / 3.7
		weight = (weight - 49.4) / 20.19


		# this gives better samples
		y = height_to_chest / im_height

		# y = height_to_chest / im_height

		# print(z, y, "=================")
		# if self.cond:


		# 	if self.return_n:
		# 		return im, (sex, age, weight, z, n)

		# 	return im, (sex, age, weight, z)

		cond_features = []
		if "sex" in self.cond:
			cond_features.append(sex)
		if "age" in self.cond:
			cond_features.append(age)
		if "weight" in self.cond:
			cond_features.append(weight)
		if "z" in self.cond:
			cond_features.append(z)

		if "y" in self.cond:
			cond_features.append(y)

		cond_features = torch.tensor(cond_features).float()

		return z, sex, age, weight, cond_features

	def __getitem__(self, index):
		

		# if self.all_slices:

		# 	for i in range(len(self.slices[i])):
		# 		(n, s, sex, weight, age, n_slices) = self.slices[index]

		# 		cond_features = self.get_features()

		# 		return im, cond_features, n, z, sex, age, weight, nodule, nodule_mask

		# 	return self.volumes[index], self.volume_n[index]
	
		(vn, n, s, sex, weight, age, n_slices, z_coord, full_h) = self.slices[index]
		height = 0
 
		# print(im.max())
		l_x, u_x, l_y, u_y = self.coords_dict[n]



		nodule = None
		nodule_mask = None
		h = 0

		if self.body_part == "chest":
			im = self.volumes[vn][s - self.radius: s + self.radius + 1].float() / 255

			# print(im.shape, "===========================")

			# h = 0
			# if self.sliding_window:

			# 	h = int(random.random() * (im.shape[1] - im.shape[2]))



			# im = im[:, h :h + im.shape[2]]


			# print(im.shape, "ttttttttt")
		elif self.body_part == "legs":
			im = self.volumes[vn]

			if random.random() > 0.5:
				side = "left"
				im = im[:, :, :im.shape[2] // 2]
			else:
				side = "right"
				im = im[:, :, im.shape[2] // 2:]

			im = im[s - self.radius: s + self.radius + 1].float() / 255

			
		h = 0
		if self.sliding_window:

			h = int(random.random() * (im.shape[1] - im.shape[2]))


		im = im[:, h :h + im.shape[2]]
			# if self.transform:
			# 	im = self.transform(im)

		# print(im.shape, "uuuuuuuuu")

		if self.sliding_window:
			h -= l_y * im.shape[1] / (full_h / 2)
			# h /= 200

		spacing = s


		z, sex, age, weight, cond_features = self.get_features(sex, weight, age, s, n_slices, z_coord, h, im.shape[1])





		if self.nodule:
			if self.body_part == "chest":
				nodule = self.nodule_volumes[vn][s - self.radius: s + self.radius + 1].float() / 255 
				nodule_mask = self.labels[vn][s - self.radius: s + self.radius + 1].float() / 255
			
			elif self.body_part == "legs":
				left_nodule, right_nodule = self.nodule_volumes[vn]
				left_nodule_mask, right_nodule_mask = self.labels[vn]
				if side == "left":
					nodule = left_nodule[s - self.radius: s + self.radius + 1].float() / 255
					nodule_mask = left_nodule_mask[s - self.radius: s + self.radius + 1].float() / 255
					
				else:
					nodule = right_nodule[s - self.radius: s + self.radius + 1].float() / 255
					nodule_mask = right_nodule_mask[s - self.radius: s + self.radius + 1].float() / 255
					
			h = int(h)
			nodule = nodule[:, h :h + im.shape[2]]
			nodule_mask = nodule_mask[:, h :h + im.shape[2]]

		# print(im.shape, nodule.shape, h, im.shape[2])
		if self.return_n:
			return im, cond_features, n, z, sex, age, weight, nodule, nodule_mask
		
		return im, cond_features



	def ordered_windows(self, index):
		
		# (vn, n, s, sex, weight, age, n_slices) = self.slices[index]


		print(index, "-----------------")
		n = self.metadata["volume_n"][index]
		vn = index

		if self.data_dir == "chop":

			sex, weight, age, spacing, thickness = 0,0,0,0,0


			sex = self.metadata["sex"][index]
			weight = self.metadata["weight"][index]
			age = self.metadata["age"][index]
			n_slices = self.metadata["n_slices"][index]

			spacing = float(self.metadata["slicespacing"][index]) 
			thickness = float(self.metadata["slicethickness"][index])

		if spacing == 0:
			spacing = 6

		if thickness == 0:
			thickness = 6

		slice_spacing = spacing + thickness

		l_x, u_x, l_y, u_y = self.coords_dict[n]




		# if self.nodule:
		# 	nodule, nodule_mask = self.nodule_volumes[vn][s - self.radius: s + self.radius + 1].float() / 255, self.labels[vn][s - self.radius: s + self.radius + 1].float() / 255



		windows = []
		window_features = []
		window_heights = []
		window_widths = []

		slice_numbers = []
	
		# label = None
		nodule_volume = None

		# label = self.labels[vn].float() / 255

		for s in range(max(n_slices - self.n_volume_slices, 1 + self.radius), n_slices - 2 - self.radius):

			im = self.nodule_volumes[vn][s - self.radius: s + self.radius + 1].float() / 255



			nodule_volume = self.nodule_volumes[vn].float() / 255 
			
			# if self.body_part == "chest":
			# 	im = self.nodule_volumes[vn][s - self.radius: s + self.radius + 1].float() / 255
			# 	nodule_volume = self.nodule_volumes[vn].float() / 255 
				

			


			# elif self.body_part == "legs":
			# 	# im = torch.cat(self.volumes[vn], 2)
			# 	im = self.volumes[vn]
			# 	nodule_volume = im[s - self.radius: s + self.radius + 1].float() / 255
				

			n_windows = im.shape[1] // 256 + 1

			z_coord = s * slice_spacing

			for w in range(n_windows):

				if self.body_part == "chest":
					window_height = int((im.shape[1] - 256) / (n_windows - 1) * w)
					window = im[:, window_height: window_height + 256]

					z, sex, age, weight, cond_features = self.get_features(sex, weight, age, s, n_slices, z_coord, window_height - l_y, nodule_volume.shape[1])

					window_heights.append(window_height)
					window_features.append(cond_features.unsqueeze(0))
					windows.append(window.unsqueeze(0))
					slice_numbers.append(s)

				elif self.body_part == "legs":

					window_height = int((im.shape[1] - 256) / (n_windows - 1) * w)
					left_window = im[:, window_height: window_height + 256, :im.shape[2] // 2]
					right_window = im[:, window_height: window_height + 256, im.shape[2] // 2:]

					z, sex, age, weight, cond_features = self.get_features(sex, weight, age, s, n_slices, z_coord, window_height - l_y, nodule_volume.shape[1])

					window_heights.append(window_height)
					window_heights.append(window_height)

					window_widths.append(0)
					window_widths.append(im.shape[2] // 2)
					
					window_features.append(cond_features.unsqueeze(0))
					window_features.append(cond_features.unsqueeze(0))

					windows.append(left_window.unsqueeze(0))
					windows.append(right_window.unsqueeze(0))

					slice_numbers.append(s)
					slice_numbers.append(s)


		full_label = None
		full_im = None

		if self.store_full_volume:
			full_im = self.full_volumes[vn].float() / 255

		if self.store_full_labels:
			full_label = self.full_labels[vn].float() / 255
		
		# label = self.labels[vn].float() / 255
		label = torch.zeros_like(nodule_volume)

		windows = torch.cat(windows, 0)
		window_features = torch.cat(window_features, 0)
		
		resized_h = im.shape[1]
		# print(full_im.shape, resized_h, im.shape, "yyyyyyyyyyyyyyyyyy")

		return windows, full_im, full_label, window_features, (l_x, u_x, l_y, u_y), window_heights, slice_numbers, resized_h, window_widths, label, nodule_volume, n
			



	def make_avg_template(self, save_im=False):
		def get_shape(row):
			a = self[row["idx"]]
			if a is None:
				return None
			return tuple(a.shape)

		# self.metadata["dim"] = self.metadata.apply(get_shape, axis=1)
		# self.metadata.to_csv("/datasets/wbmri/anonym_wbmri_metadata2.csv")


		sum_vol = np.zeros((32, 192, 192))
		count = 0
		for n in range(1600):

			fn = "/datasets/wbmri/new_wbmri_preproc4/volume_{}.npy".format(n)
			# volume = np.load(fn)[:, l_x: u_x, l_y: u_y].astype(np.float)
			try:
				volume = np.load(fn)
			except:
				print(n, "nonexist")
				continue


			label = np.zeros_like(volume)

			volume = volume[:, l_x: u_x, l_y: u_y].astype(np.float)
			volume -= np.amin(volume)
			volume /= np.amax(volume)


			volume *= 255
			volume = volume.astype(np.uint8)



			for s in range(len(volume)):
				Image.fromarray(volume[s]).save("/datasets/wbmri/slice_png/volume_{}_{}_{}.png".format(self.body_part, n, s))

				try:
					slice_label = Image.open("/datasets/wbmri/labelled_volumes/volume_{}/slice_{}_label.png".format(n, s))
					slice_label = slice_label[l_x: u_x, l_y: u_y]
					print(n, s, slice_label.shape)
				except:
					continue

				label[s] = slice_label



			vol, n = self[i]
			vol = vol.unsqueeze(0)
			# print(torch.max(vol), torch.min(vol))            
			if vol is not None and int(self.metadata["dim1"][i]) > 20:
				# print(vol.shape)

				shape = list(vol.shape[2:])

				shape[0] = 32
				
				pix_spacing = float(self.metadata["pixelspacing1"][i])

				

				shape[1] = int(shape[1] * pix_spacing)
				shape[2] = int(shape[2] * pix_spacing)

				if pix_spacing == 0:
					continue
				# print(shape, self.metadata["pixelspacing1"][i])
				resized = torch.nn.functional.interpolate(vol, mode="trilinear", size=shape).squeeze(0).squeeze(0).numpy()
				
				
				# take middle 192 columns and 192 rows starting 5mm above head

				coords = np.indices(resized.shape)

				# print(coords.shape)

				if self.body_part == "head":

					# print(np.amax(coords[1][resized > 0.3]), np.amin(coords[1][resized > 0.3]))
					highest_point = max(np.amin(coords[1][resized > 0.3]) - 5, 0)

					if shape[1] - highest_point < 192:
						print(highest_point, shape[1])
						continue

					im = resized[:, highest_point: highest_point + 192, resized.shape[2] // 2 - 96: resized.shape[2] // 2 + 96]

					# print(im.shape)

					np.save("/datasets/wbmri/head_volumes/{}.npy".format(n), (im * 255).astype(np.uint8))

				elif self.body_part == "chest":
					
					im = resized[:, :192, :]

					np.save("/datasets/wbmri/chest_volumes/{}.npy".format(n), (im * 255).astype(np.uint8))


				sum_vol += im
				count += 1
			else:
				continue

		sum_vol /= count

		# np.save("/datasets/wbmri/head_avg.npy", sum_vol)



# class CropsDataset(Dataset):

# 	def __init__(self, split="train", crop_size=64, body_part="head", channel=5, patient_n=None, transform=None, top_crop=False, diseased=False):

# 		self.data_path = "/datasets/wbmri/"

# 		self.split = split
# 		self.body_part = body_part
		
# 		self.channel = channel
# 		self.crop_size = crop_size
# 		self.radius = crop_size // 2
# 		self.slice_radius = channel // 2
# 		self.coords_dict = np.load(self.data_path + "tissue_coords2/{}_coords.npz".format(self.body_part), allow_pickle=True)["arr_0"][()]
		
# 		self.tissue_coords = pd.DataFrame.from_dict(coords_dict, orient="index")
# 		# print(self.tissue_coords)

# 		# self.idx_to_volume_n = list(self.tissue_coords.index)
# 		self.idx_to_volume_n = self.tissue_coords.index.values

# 		# self.idx_to_volume_n = np.array(list(self.tissue_coords.keys()))

# 		# print(self.idx_to_volume_n)
# 		self.metadata = pd.read_csv("/datasets/wbmri/anonym_wbmri_metadata_nodule_val.csv")

# 		if split == "train":
# 			patients = self.metadata["volume_n"][self.metadata["rand"] < 0.8].values
# 		elif split == "test":
# 			patients = self.metadata["volume_n"][self.metadata["rand"] > 0.8].values
# 		elif split == "all":
# 			patients = self.metadata["volume_n"].values



# 		tmp = []
# 		for n in self.idx_to_volume_n:
# 			# print(n)
# 			if n in patients:
# 				tmp.append(n)

# 		# print(patient_n)
# 		# print(self.idx_to_volume_n)

# 		self.idx_to_volume_n = tmp
# 		# print(self.idx_to_volume_n)
		
# 		if self.split == "test":
# 			self.nodule_n = self.metadata["volume_n"][self.metadata["rand"] > 0.8].values


# 		self.transform = transform
# 		self.patient_n = patient_n
# 		self.top_crop = top_crop
# 		self.diseased = diseased
# 		self.volumes = {}
# 		self.disease_loc = {}

# 		for i in range(len(self.idx_to_volume_n)):
# 			n = self.idx_to_volume_n[i]

# 			self.volumes[n] = np.load("/datasets/wbmri/head_volumes/{}.npy".format(n)) / 255.0
# 			if self.diseased:

# 				# sample disease loc
# 				idx = np.random.randint(self.tissue_coords[0][n].shape[-1])
# 				sample_idx = self.tissue_coords[0][n][:, idx]
# 				self.disease_loc[n] = sample_idx

# 				self.volumes[n] = insert_nodule(self.volumes[n], self.disease_loc[n], 0.5, 5)

# 	def __len__(self):
# 		# print(len(self.idx_to_volume_n))
# 		if self.patient_n:
# 			# i = self.idx_to_volume_n[]
# 			# print(self.tissue_coords[0][self.patient_n].shape, "============")
# 			return self.tissue_coords[0][self.patient_n].shape[-1]

# 		return len(self.idx_to_volume_n)



# 	def __getitem__(self, index):
		
# 		if self.patient_n:
# 			idx = index
# 			i = self.patient_n
# 		else:
# 			# volume n
# 			i = self.idx_to_volume_n[index]


			
# 			# sample crop
# 			idx = np.random.randint(self.tissue_coords[0][i].shape[-1])



# 		# print(self.tissue_coords[0][i].shape, idx, i)
# 		sample_idx = self.tissue_coords[0][i][:, idx]



# 		# # slices = []
# 		# slices = np.zeros((self.channel, self.crop_size, self.crop_size), dtype=np.float16)

# 		# diseased = random.random() > 0.5


# 		# # load full volume
# 		# for s in range(self.channel):

# 		#     fn = "/datasets/wbmri/slice_png/volume_{}_{}_{}.png".format(self.body_part, i, sample_idx[0] - self.slice_radius + s)
		

			
# 		#     bod_volume = io.imread(fn, as_gray=True).astype(np.float16)
# 		#     # print(bod_volume.shape)
			
# 		#     crop = bod_volume[sample_idx[1] - self.radius: sample_idx[1] + self.radius, 
# 		#                         sample_idx[2] - self.radius: sample_idx[2] + self.radius]
# 		#     # print(crop.shape)
			
# 		#     # slices.append(crop)
# 		#     if self.split == "test":
# 		#         if not self.patient_n and i in self.nodule_n:
# 		#             crop = insert_nodule(crop, 150, 10)
# 		#         elif self.patient_n and diseased:
# 		#             crop = insert_nodule(crop, 150, 10)
# 		#     slices[s] = crop

# 		#     # print(type(slices[s]))


# 		# # im = np.concatenate(slices, axis=0)
# 		# slices = slices / 255
# 		# im = torch.from_numpy(slices).float()
		
# 		# if self.transform:
# 		#     sample = self.transform(im)
# 		# # print(im.shape)
		
# 		# if self.split == "test":
# 		#     if not self.patient_n:
# 		#         return im, 1 if i in self.nodule_n else 0                
# 		#     elif self.patient_n:
# 		#         return im, 1 if diseased else 0

# 		# return im



		

# 		# print(coords.shape, volume[coords].shape, tissue_coords.shape, torch.min(volume), torch.max(volume), fn)
# 		# print(index, len(coords[0]) * len(coords[1]) * len(coords[2]), len(tissue_coords[0]), tissue_coords.shape)

# 		# volume = np.load("/datasets/wbmri/head_volumes/{}.npy".format(i)) / 255.0

# 		if self.top_crop:
# 			return self.volumes[i][16:17, :32, 80:112], i, 0

# 		crop = self.volumes[i][sample_idx[0] - self.slice_radius: sample_idx[0] + self.slice_radius + 1, 
# 						sample_idx[1] - self.radius: sample_idx[1] + self.radius, 
# 						sample_idx[2] - self.radius: sample_idx[2] + self.radius]

# 		# torchvision.utils.save_image(crop[0], fp="crop_samples/{}.png".format(index))

# 		if self.diseased:
# 			disease_in_crop = self.disease_loc[i][0] == sample_idx[0] and np.abs(self.disease_loc[i][1] - sample_idx[1]) < 20 and np.abs(self.disease_loc[i][2] - sample_idx[2]) < 20
# 		else:
# 			disease_in_crop = 0

# 		# if self.transform:
# 		#     crop = self.transform(crop)

# 		# sample /= 255


# 		return crop, i, disease_in_crop

def hist_match(source, template):
	"""
	Adjust the pixel values of a grayscale image such that its histogram
	matches that of a target image

	Arguments:
	-----------
		source: np.ndarray
			Image to transform; the histogram is computed over the flattened
			array
		template: np.ndarray
			Template image; can have different dimensions to source
	Returns:
	-----------
		matched: np.ndarray
			The transformed output image
	"""

	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)


def image_histogram_equalization(image, number_bins=256):
	# from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

	# get image histogram
	image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
	cdf = image_histogram.cumsum() # cumulative distribution function
	cdf = 255 * cdf / cdf[-1] # normalize

	# use linear interpolation of cdf to find new pixel values
	image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

	image_eq_histogram = np.histogram(image.flatten(), number_bins, density=True)

	# print(image_histogram, "qqqqqqq")
	# print(image_eq_histogram)

	return image_equalized.reshape(image.shape), cdf



class CropMaker:

	def __init__(self, split="train", crop_size=64, body_part="head", channel=5, interp_z=False):

		self.metadata = pd.read_csv("/datasets/wbmri/anonym_wbmri_metadata.csv")
		self.metadata = self.metadata.fillna("0")

		self.data_path = "/datasets/wbmri"

		self.body_part = body_part
		# self.fns = get_crop_coords("wb", body_part)
		# self.fns += get_crop_coords("era", body_part)
		self.fns = get_crop_coords(body_part)
		
		print(self.metadata)
		self.channel = channel
		self.crop_size = crop_size
		self.radius = crop_size // 2
		self.slice_radius = channel // 2
		self.interp_z = interp_z


	def save_volume_crops(self):
		def read_volume(i):
			# n_slices = len(os.listdir("/datasets/wbmri/headless_preproc/volume_{}".format(i)))
			n_slices = len(os.listdir("/datasets/wbmri/normal_preproc/volume_{}".format(i)))


			volume = None
			for z in range(n_slices):
				# s = np.array(Image.open("/datasets/wbmri/headless_preproc/volume_{}/{}.png".format(i, z)))[np.newaxis, ...]

				# s = np.array(Image.open("/datasets/wbmri/preproc5/volume_{}/{}.png".format(i, z)))[np.newaxis, ...]
				s = np.array(Image.open("/datasets/wbmri/normal_preproc/volume_{}/{}.png".format(i, z)))[np.newaxis, ...]

				# print(s.shape)
				if volume is None:
					volume = s
				else:
					volume = np.concatenate((volume, s), 0)


			return volume

		hist_template = read_volume(11)


		correct_crop_list = os.listdir("/datasets/wbmri/headless_preproc_crops/")
		for i, l_x, l_y, u_x, u_y in tqdm(self.fns):

			if "chest_{}.png".format(i) not in correct_crop_list:
				continue

			try:
				volume = read_volume(i)
				
			except:
				continue

			vol_labels = np.zeros_like(volume)
			# print(l_x, u_x, l_y, u_y)
			for s in range(volume.shape[0]):
				try:
				# if i == 5 and s == 15:

					slice_label = Image.open("/datasets/wbmri/labelled_volumes/volume_{}/slice_{}_label/label.png".format(i, s))
					slice_label = (np.array(slice_label) > 0).astype(np.uint8)
					
					vol_labels[s] = slice_label
					# print(slice_label.shape, "bbb")
					print(i, s)
				except:
					pass

			# print(np.sum(vol_labels), "a")
			vol_labels = vol_labels[:, l_x: u_x, l_y: u_y].astype(np.float)
			volume = volume[:, l_x: u_x, l_y: u_y].astype(np.float)

			im = volume.astype(np.uint8)
			resized_labels = vol_labels.astype(np.uint8)
			# # print(np.sum(vol_labels),= "b")
			# # print(i, np.sum(vol_labels), "=====")

			# # print(vol_labels.shape)


			# if int(self.metadata[self.metadata["volume_n"] == i]["dim1"]) < 20:
			# 	print(i, "---------")
			# 	continue

			# shape = list(volume.shape)

			# if self.interp_z:
			# 	shape[0] = 32

			
			# pix_spacing = float(self.metadata[self.metadata["volume_n"] == i]["pixelspacing1"])

			
			# # head uses 192 x 192 mm
			# # shape[1] = int(shape[1] * pix_spacing) 
			# # shape[2] = int(shape[2] * pix_spacing)
			
			# # chest
			# shape[1] = 192
			# shape[2] = 192

			# # print(shape, volume.shape)
			# # print(np.sum(vol_labels))
			# vol_labels = torch.from_numpy(vol_labels).unsqueeze(0).unsqueeze(0)
			# # print(i, vol_labels.shape, torch.sum(vol_labels))

			# volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)

			# if pix_spacing == 0:
			# 	pix_spacing = 1

			# # print(shape, self.metadata["pixelspacing1"][i])
			# # print(torch.max(vol_labels), "--------------------------------")
			# resized_labels = torch.nn.functional.interpolate(vol_labels, mode="nearest", size=shape).squeeze(0).squeeze(0).numpy()
			# resized = torch.nn.functional.interpolate(volume, mode="trilinear", size=shape).squeeze(0).squeeze(0).numpy()
			

			# # take middle 192 columns and 192 rows starting 5mm above head

			# coords = np.indices(resized.shape)

			# # print(coords.shape)

			# # print(np.amax(coords[1][resized > 0.3]), np.amin(coords[1][resized > 0.3]))
			# # print(i, np.amin(resized), np.amax(resized), "----------")
			# # print(coords[1][resized > 100].shape, coords.shape)
			# if self.body_part == "head":

			# 	highest_point = max(np.amin(coords[1, 5:-5][resized[5:-5] > 50]) - 5, 0)
			# 	# print(highest_point, "===============")

			# 	# if shape[1] - highest_point < 192:
			# 	#     print(highest_point, shape[1])
			# 	#     continue

			# 	im = resized[:, highest_point: highest_point + 192, resized.shape[2] // 2 - 96: resized.shape[2] // 2 + 96]

			# elif self.body_part == "chest":
			# 	# print("a")
			# 	im = resized[:, :, :]
			# elif self.body_part == "abdomen":
			# 	im = resized[:, :, :]


			# # im = image_histogram_equalization(im)[0]
			# ks = (im.shape[0] // 2, im.shape[2]//4, im.shape[2] // 4)
			# # print(new_arr.shape)
			# # print(im)
			# # im[im < 0] = 0
			# # im[im > 1] = 1

			# # im = exposure.equalize_hist(im.astype(np.uint8), mask=im > 10)

			# im = exposure.equalize_adapthist(im.astype(np.uint8), kernel_size=ks, clip_limit=0.0001)


			# # im[im > 10] = exposure.match_histograms(im[im > 10], hist_template[hist_template > 10])
			# # print(np.amax(im))

			# # print(np.amin(im), np.amax(im), "------------------------------------------")
			# im = (im * 255).astype(np.uint8)
			# # im = im.astype(np.uint8)
			# resized_labels = (resized_labels * 255).astype(np.uint8)


			# np.save("/datasets/wbmri/{}_volumes_preproc5/{}.npy".format(self.body_part, i), im)
			# # if torch.sum(vol_labels) > 0:

			# np.save("/datasets/wbmri/{}_volumes_preproc5/{}_labels.npy".format(self.body_part, i), resized_labels)
			# # print(i, np.sum(resized_labels))

			# # volume -= np.amin(volume)
			# # volume /= np.amax(volume)


			# # volume *= 255
			# # volume = volume.astype(np.uint8)


			print(i)

			for s in range(len(im)):

				os.makedirs("/datasets/wbmri/slice_png_{}_normal/volume_{}".format(self.body_part, i), exist_ok=True)
				Image.fromarray(im[s]).save("/datasets/wbmri/slice_png_{}_normal/volume_{}/{}.png".format(self.body_part, i, s))
				os.makedirs("/datasets/wbmri/slice_png_{}_labels_normal/volume_{}".format(self.body_part, i), exist_ok=True)
				Image.fromarray(resized_labels[s]).save("/datasets/wbmri/slice_png_{}_labels_normal/volume_{}/{}.png".format(self.body_part, i, s))



	def save_indices(self):

		data = {}
		for i in tqdm(range(1600)):
			try:
				volume = np.load("/datasets/wbmri/head_volumes/{}.npy".format(i)) / 255.0
			except:
				continue
			coords = np.indices((volume.shape[0] - self.channel + 1, volume.shape[1] - self.crop_size + 1, volume.shape[2] - self.crop_size + 1))


			coords[0] += self.slice_radius
			coords[1] += self.radius
			coords[2] += self.radius


			tissue_int = 0.2


			tissue_idx = volume[coords[0], coords[1], coords[2]] > tissue_int

			tissue_coords = coords[:, tissue_idx]

		
			print(coords.shape, tissue_coords.shape)

			tissue_coords = tissue_coords.reshape(3, -1)            
			tissue_coords = tissue_coords[:, (tissue_coords[1] % 5 == 0) & (tissue_coords[2] % 5 == 0)]

			print(len(tissue_coords.reshape(-1))/ len(coords.reshape(-1))) 

			data[i] = tissue_coords
			

		# data = {}
		# for i, l_x, l_y, u_x, u_y in tqdm(self.fns):
		#     print(i)
		#     fn = "/datasets/wbmri/new_wbmri_preproc4/volume_{}.npy".format(i)
		#     # volume = np.load(fn)[:, l_x: u_x, l_y: u_y].astype(np.float)
		#     volume = np.load(fn)[:, l_x: u_x, l_y: u_y].astype(np.float)

		#     print(volume.shape)

		#     volume -= np.amin(volume)
		#     volume /= np.amax(volume)

		#     volume = torch.from_numpy(volume).float()

		#     # Set these to whatever you want for your gaussian filter
		#     kernel_size = 15
		#     sigma = 3

		#     # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
		#     x_cord = torch.arange(kernel_size)
		#     x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
		#     y_grid = x_grid.t()
		#     xy_grid = torch.stack([x_grid, y_grid], dim=-1)

		#     mean = (kernel_size - 1)/2.
		#     variance = sigma**2.

		#     # Calculate the 2-dimensional gaussian kernel which is
		#     # the product of two gaussian distributions for two different
		#     # variables (in this case called x and y)
		#     gaussian_kernel = (1./(2.*math.pi*variance)) *\
		#                       torch.exp(
		#                           -torch.sum((xy_grid - mean)**2., dim=-1) /\
		#                           (2*variance)
		#                       )
		#     # Make sure sum of values in gaussian kernel equals 1.
		#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

		#     # Reshape to 2d depthwise convolutional weight
		#     gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
		#     gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

		#     gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1,
		#                                 kernel_size=kernel_size, groups=1, bias=False)#.cuda()

		#     gaussian_filter.weight.data = gaussian_kernel
		#     gaussian_filter.weight.requires_grad = False

		#     with torch.no_grad():
		#         blur = gaussian_filter(volume.unsqueeze(1)).squeeze(1).numpy()


		#     coords = np.indices((volume.shape[0] - self.channel + 1, volume.shape[1] - self.crop_size + 1, volume.shape[2] - self.crop_size + 1))


		#     coords[0] += self.slice_radius
		#     coords[1] += self.radius
		#     coords[2] += self.radius


		#     tissue_int = 0.3


		#     # print(blur.shape, coords.shape)
		#     tissue_idx = blur[coords[0], coords[1], coords[2]] > tissue_int

		#     # print(tissue_idx.shape)
		#     tissue_coords = coords[:, tissue_idx]

		
		#     print(coords.shape, tissue_coords.shape)

		#     tissue_coords = tissue_coords.reshape(3, -1)            
		#     tissue_coords = tissue_coords[:, (tissue_coords[1] % 5 == 0) & (tissue_coords[2] % 5 == 0)]

		#     print(len(tissue_coords.reshape(-1))/ len(coords.reshape(-1)))

		#     data[i] = tissue_coords
			

		np.savez("/datasets/wbmri/tissue_coords2/{}_coords.npy".format(self.body_part), data)



