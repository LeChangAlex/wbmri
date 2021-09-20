from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr
import math
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

from sklearn import metrics

import matplotlib.pyplot as plt
from source.wbmri_crops import BodyPartDataset

import wandb
# import scikitplot as skplt

import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import scipy

import heapq
import queue
from torch.nn.functional import softplus, softmax


class MaxPriorityQueue(queue.PriorityQueue):
	def __init__(self):
		super(queue.PriorityQueue, self).__init__()


	def put(self, item):	
		max_item = - item[0], item[1]		
		# print(max_item)
		super(queue.PriorityQueue, self).put(max_item)


class MedianPool2d(nn.Module):
	""" Median pool (usable as median filter when stride=1) module.
	
	Args:
		 kernel_size: size of pooling kernel, int or 2-tuple
		 stride: pool stride, int or 2-tuple
		 padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
		 same: override padding and enforce same padding, boolean
	"""
	def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
		super(MedianPool2d, self).__init__()
		self.k = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _quadruple(padding)  # convert to l, r, t, b
		self.same = same

	def _padding(self, x):
		if self.same:
			ih, iw = x.size()[2:]
			if ih % self.stride[0] == 0:
				ph = max(self.k[0] - self.stride[0], 0)
			else:
				ph = max(self.k[0] - (ih % self.stride[0]), 0)
			if iw % self.stride[1] == 0:
				pw = max(self.k[1] - self.stride[1], 0)
			else:
				pw = max(self.k[1] - (iw % self.stride[1]), 0)
			pl = pw // 2
			pr = pw - pl
			pt = ph // 2
			pb = ph - pt
			padding = (pl, pr, pt, pb)
		else:
			padding = self.padding
		return padding
	
	def forward(self, x):
		# using existing pytorch functions and tensor ops so that we get autograd, 
		# would likely be more efficient to implement from scratch at C/Cuda level
		x = F.pad(x, self._padding(x), mode='reflect')
		x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
		x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
		return x

def insert_nodule(vol, intensity=0.6, sigma=7, n_nodule=1):

	idx_seed1= [56937, 26732, 26700, 140505, 105911, 17328, 18298, 56750, 93689, 98449, 47142, 89690, 135446, 6023, 98114, 5144, 28623, 103717, 72517, 51767, 57063, 33875, 142768, 41120, 11472, 53915, 101237, 121462, 28454, 128177, 110671, 20212, 1512, 5116, 115946, 17665, 123208, 81595, 12174, 66947, 46075, 41594, 1611, 118365, 101321, 38834, 70417, 101715, 82380, 63084, 55292, 94712, 6397, 22514, 57930, 113390, 38815, 97091, 9296, 2147, 121891, 57187, 15943, 52149, 53312, 130715, 125233, 110578, 17419, 62749, 81597, 137203, 91948, 89107, 110655, 60599, 88403, 38229, 20820, 63593, 142544, 66191, 141634, 32318, 82017, 32520, 14622, 107832, 74356, 87900, 15914, 28246, 34875, 72049, 44793, 30664, 55971, 6980, 116187, 116749, 45926, 56220, 51332, 135005, 92505, 78232, 17671, 809, 29945, 27546, 35339, 115353, 44817, 54142, 73619, 10381, 12820, 24463]
	locs = []
				
	# idx_list = []
	for i in range(vol.shape[0]):
		locs1 = []
		for j in range(n_nodule):

			coords = np.indices((vol.shape[1] - 9, vol.shape[2] - 32, vol.shape[3] - 32))

			coords[0] += 5
			coords[1] += 16
			coords[2] += 16

			tissue_int = 0.1


			tissue_idx = vol[i, coords[0], coords[1], coords[2]] > tissue_int

			# print(tissue_idx.shape, coords.shape)
			a = tissue_idx.cpu().numpy()
			tissue_coords = coords[:, a]

			idx = np.random.randint(tissue_coords.shape[-1])
			# idx = idx_seed1[i]

			# idx_list.append(idx)
			loc = tissue_coords[:, idx]
			# print(loc)



			x, y = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
			d = np.sqrt(x * x + y * y)
			nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))#[np.newaxis, ...]

			nodule_z, nodule_x, nodule_y = loc[0], loc[1], loc[2]
			# nodule_x, nodule_y = vol.shape[2] // 2, vol.shape[3] // 2

			# print(vol.shape, nodule.shape)
			# print(nodule_y, nodule_x)
			# vol[i, nodule_z - 1: nodule_z + 2, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] += torch.from_numpy(nodule * intensity).cuda()
			vol[i, nodule_z, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] += torch.from_numpy(nodule * intensity).cuda()

			locs1.append(loc)
		locs.append(locs1)
	# print(idx_list)
	vol[vol > 1] = 1

	return vol, locs
	

def insert_nodule_patch(im, intensity=0.4, sigma=5):

	im = im.clone()
	labs = []
	masks = []
				
	n_list = []
	# idx_list = []
	for i in range(im.shape[0]):

		coords = np.indices((max(1, im.shape[1] - 2), 192, 192))

		
		tissue_int = 0.1


		tissue_idx = im[i, coords[0], coords[1], coords[2]] > tissue_int

		# print(tissue_idx.shape, coords.shape)
		a = tissue_idx.cpu().numpy()
		tissue_coords = coords[:, a]

		# if tissue_coords.shape[1] <= 100:
		# 	labs.append(0)
		# 	masks.append((torch.zeros_like(im[i, 1:-1]) > 0.05).cuda().bool())

		# 	continue

		if tissue_coords.shape[-1] == 0:
			masks.append(torch.zeros_like(im[0, 0]).bool().unsqueeze(0))
			continue

		idx = np.random.randint(tissue_coords.shape[-1])
		# idx = idx_seed1[i]

		# idx_list.append(idx)
		loc = tissue_coords[:, idx]
		# print(loc)



		# x, y = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
		# d = np.sqrt(x * x + y * y)
		# nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))#[np.newaxis, ...]

		nodule_z, nodule_x, nodule_y = loc[0], loc[1], loc[2]

		
		# nodule_x, nodule_y = vol.shape[2] // 2, vol.shape[3] // 2

		# print(vol.shape, nodule.shape)
		# print(nodule_y, nodule_x)
		# vol[i, nodule_z - 1: nodule_z + 2, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] += torch.from_numpy(nodule * intensity).cuda()

		if nodule_x > 176:
			nodule_x = 176
		if nodule_y > 176:
			nodule_y = 176
		if nodule_x < 16:
			nodule_x = 16
		if nodule_y < 16:
			nodule_y = 16

		# mask = (im[i, 1, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] > tissue_int).int().cuda()
		
		x, y = np.meshgrid(np.linspace(0, 191, 192), np.linspace(0, 191, 192))
		d = np.sqrt((x - nodule_y) ** 2 + (y - nodule_x) ** 2)
		nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))#[np.newaxis, ...]
		mask = (im[i, 0] > tissue_int).int().cuda()



		# im[i, 1:-1, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] += torch.from_numpy(nodule * intensity).cuda().unsqueeze(0) * mask
		# im[i, 1, nodule_x - 15: nodule_x + 15, nodule_y - 15: nodule_y + 15] += torch.from_numpy(nodule * intensity).cuda() * mask
		im[i, 0] += torch.from_numpy(nodule * intensity).cuda() * mask
		labs.append(1)
		n_list.append(i)
		# print((mask * (torch.from_numpy(nodule) > 0.05).cuda()).shape)
		# print((mask * (torch.from_numpy(nodule) > 0.05).cuda()).bool().shape)
		masks.append((mask * (torch.from_numpy(nodule) > 0.15).cuda()).bool().unsqueeze(0))


	im[im > 1] = 1


	# im = im.clone()


	# x, y = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
	# d = np.sqrt(x * x + y * y)
	# nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

	# # for i in range(im.shape[0]):

	# im[i, 1:-1, j-15: j+15, k-15:k+15] = torch.from_numpy(nodule * intensity).cuda()

	# return im, labs, n_list

	return im, masks, n_list



def make_heatmap(diff, im):
	im = im.unsqueeze(0).repeat(3, 1, 1)

	heat = im.clone()
	heat[0, :, :] += diff * 0.5

	# print(diff.sum())
	# print(im.sum())

	return torch.cat((diff.unsqueeze(0).repeat(3, 1, 1), im, heat), 1)


def nodule_in_patch(loc, z, x, y):
	for i in range(len(loc)):
		if (abs(loc[i][0] - z) <= 0 and abs(loc[i][1] - x) <= 14 and abs(loc[i][2] - y) <= 14):
			return True
	return False

def auprc_plot(precision, recall):

	plt.clf()
	fig, ax0 = plt.subplots()
	ax0.step(recall, precision, where='post')

	# ax0.xlabel('Recall')
	# ax0.ylabel('Precision')
	# ax0.ylim([0.0, 1.05])
	# ax0.xlim([0.0, 1.0])
	# ax0.title('Average precision score {}'.format())
	
	plt.savefig("auprc.png")
	plt.close()


def compute_stats(losses, labels, plot=False):
	# loglik_auroc = metrics.roc_auc_score(y_all, loglik4all)    
	auroc = metrics.roc_auc_score(labels, losses)
	fpr, tpr, thresholds = metrics.roc_curve(labels, losses)
	if plot:
		plt.plot(fpr,tpr)
		plt.savefig("auroc.png")
		plt.close()

	youden_idx = np.argmax(tpr - fpr)
	sensitivity = tpr[youden_idx]
	specificity = 1 - fpr[youden_idx]
	# loglik_auprc = metrics.average_precision_score(y_all, loglik4all)
	precision, recall, thresholds = metrics.precision_recall_curve(labels, losses)
	auprc = metrics.auc(recall, precision)

	return auroc, auprc, sensitivity, specificity

def bootstrap(losses, labels, p=0.1):


	labels = np.array(labels)
	losses = np.array(losses)

	nodule_losses = losses[labels == 1]
	healthy_losses = losses[labels == 0]


	aurocs, auprcs, sens, specs = [], [], [], []
	# for p in proportions:
	for i in range(100):
		idx = np.random.choice(np.arange(len(nodule_losses)), replace=False, size=int(len(healthy_losses) * p))
		ls = nodule_losses[idx]
		# print(ls, healthy_losses)
		p_ls = np.concatenate((healthy_losses, ls))

		p_labs = np.concatenate((np.zeros(len(healthy_losses)), np.ones(len(ls))))

		auroc, auprc, sensitivity, specificity = compute_stats(p_ls, p_labs, plot=i == 0)

		aurocs.append(auroc)
		auprcs.append(auprc)
		sens.append(sensitivity)
		specs.append(specificity)

	return aurocs, auprcs, sens, specs


def histogram(data, y, fn, upper=100):

	bins = np.linspace(-10, 10, 100)
	# bins = np.linspace(-4000, 4000, 30)

	# pyplot.hist
	# pyplot.hist(y, bins, alpha=0.5, label='y')
	# pyplot.legend(loc='upper right')
	# pyplot.show()

	# print(y.shape)
	# y = torch.tensor(y)
	# data = torch.tensor(data)

	plt.clf()
	fig, axes = plt.subplots(nrows=2)
	ax0, ax1 = axes.flatten()
	
	# print(data[y==1], "================")

	# bp = ax1.hist(data, density=True, label=y)#, showfliers=True, whis=3, density=True)
	# _ = ax0.hist(data[y == 0], bins, density=True, alpha=1)
	# _ = ax1.hist(data[y == 1], bins, density=True, alpha=1)

	# print(torch.min(data[y==0]), torch.max(data[y==0]))
	# print(torch.min(data[y==1]), torch.max(data[y==1]),"===========")

	_ = ax0.hist(data[y == 0], density=True, alpha=1)
	_ = ax1.hist(data[y == 1], density=True, alpha=1)
	# _ = ax0.hist(data[torch.logical_not(y)], density=False, alpha=1, bins=bins)
	# _ = ax1.hist(data[y], density=False, alpha=1, bins=bins)
	# _ = ax0.hist(data[y == 0], density=False, alpha=1, bins=bins)
	# _ = ax1.hist(data[y == 1], density=False, alpha=1, bins=bins)
	# ax1.legend(loc='upper right')
	# ax1.set_xticklabels(y, rotation=45)

	plt.tight_layout()
	plt.savefig("hist_" + fn)
	plt.close()


	# bins = np.linspace(-10, 10, 100)
	# bins = np.linspace(-4000, 4000, 30)

	# pyplot.hist
	# pyplot.hist(y, bins, alpha=0.5, label='y')
	# pyplot.legend(loc='upper right')
	# pyplot.show()


	plt.clf()

	fig, axes = plt.subplots(nrows=2)
	ax0, ax1 = axes.flatten()
	
	# print(data[y==1], "================")

	# bp = ax1.hist(data, density=True, label=y)#, showfliers=True, whis=3, density=True)
	# _ = ax0.hist(data[y == 0], bins, density=True, alpha=1)
	# _ = ax1.hist(data[y == 1], bins, density=True, alpha=1)

	# print(torch.min(data[y==0]), torch.max(data[y==0]))
	# print(torch.min(data[y==1]), torch.max(data[y==1]),"===========")

	# _ = ax0.hist(data[y == 0], density=True, alpha=1)
	# _ = ax1.hist(data[y == 1], density=True, alpha=1)
	_ = ax0.hist(data[y == 0], density=True, alpha=1, bins=100)
	_ = ax1.hist(data[y == 1], density=True, alpha=1, bins=100)

	# ax1.legend(loc='upper right')
	# ax1.set_xticklabels(y, rotation=45)

	plt.tight_layout()
	plt.savefig("hist2_" + fn)
	plt.close()
	# skplt.metrics.plot_roc_curve(data, y)
	# plt.savefig("roc_" + fn)
	# plt.close()


def dice_coeff(pred, target):
	smooth = 1.
	num = pred.size(0)
	m1 = pred.view(num, -1).float()  # Flatten
	m2 = target.view(num, -1).float()  # Flatten
	intersection = (m1 * m2).sum().float()


	# print(pred.sum(), intersection, m1.sum() + m2.sum())
	return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def score_fct(S, mus, sigmas):


	# C = torch.sum((S * mus) / (sigmas ** 2)).item()

	# # print(C, "a")
	# B = torch.sum((mus ** 2) / (sigmas ** 2)).item()
	# # print(B, "b")

	# if C <= B:
	# 	return 0

	# out = ((C - B) ** 2) / (2 * B)

	num = torch.sum((S * mus)  / (sigmas ** 2)).item()
	den = torch.sum((mus ** 2) / (sigmas ** 2)).item()

	# print(num, den)
	q = num / den
	# print(q)
	if q < 1:
		return 0, 1

	# print(q, "qqqqqqq")
	out = (2 * q * mus * S - (q ** 2) * (mus ** 2) - 2 * S * mus + mus ** 2) / 2 / (sigmas ** 2)
	out = torch.sum(out).item()
	# print(out)	q1	

	return out, q


def priority_fct(S, mus, sigmas):

	return S / mus


def update_scores(pq, S, mus, sigmas, route):

	l = []
	while not pq.empty():
		_, coord = pq.get()


		new_route = np.concatenate((route, np.array([[coord[0]], [coord[1]]])), 1) 
		# print(new_route.shape)
		new_score, q = score_fct(S[new_route], mus[new_route], sigmas[new_route])


		l.append((new_score, coord))

	for e in l:
		pq.put(e)

def sliding_window_features(model, im, K=1):

	rec_losses = torch.zeros((im.shape[0], 192, 192))
	# sigmas = torch.zeros((6, 192, 192))
	# mus = torch.zeros((6, 192, 192))
	rec_params = torch.zeros((im.shape[0], 2, 192, 192))

	for j in tqdm(range(11)):
		for k in range(11):
			# print(j% 2, k % 2)
			# if j % 2 == 0 or k % 2 == 0:
			# 	# print("a")	
			# 	continue
			x = (j + 1) * 16
			y = (k + 1) * 16

			# rng = random.random()
			# # if rng < 0.8:
			# # 	continue

			# # patch_im, batch_labels, vols = insert_nodule_patch(im, x, y, args.intensity, args.sigma)


			# # print(len(batch_labels), batch_labels[0].shape)
			# # print(torch.stack(batch_labels, 0).shape, "----")

			# batch_labels = torch.stack(batch_labels, 0)[:, :, x - 16: x + 16, y - 16: y + 16]
			# # patch_im = patch_im[np.array(batch_labels) == 1]

			# # batch_labels = [1] * patch_im.shape[0]
			# if patch_im.shape[0] > 10:
			# 	Image.fromarray((patch_im[10, 1] * 255).cpu().numpy().astype(np.uint8)).save("nodule_im/{}_{}_{}_{}.png".format(j, k, args.intensity, args.sigma))



			mask = torch.zeros_like(im)
			mask[:, :, x - 16: x + 16, y - 16: y + 16] = 1

			# predict = model.generate_samples_params(im, mask)[:, 0, 0:3]

			rec_loss, kl, pr, rec_params_cur = model.features(im, mask, K=K)

			# rec_loss = -rec_loss


			# rec_params_cur = rec_params_cur[:, 0]

			# print(rec_params_cur.shape)
			rec_params_cur[:, :, 1] = softplus(rec_params_cur[:, :, 1])
			rec_params_cur[:, :, 1] = rec_params_cur[:, :, 1].clamp(min=0.01)
			
			rec_params_cur = rec_params_cur.mean(1)

			# print(rec_params_cur.shape)

			mus = rec_params_cur[:, 0:1][mask == 1]
			sigmas = rec_params_cur[:, 1:2][mask == 1]
			# print(mus.shape, sigmas.shape)


			# print(rec_params_cur)
			rec_losses[:, x - 16: x + 16, y - 16: y + 16] += rec_loss.reshape(im.shape[0], 32, 32).cpu()
			rec_params[:, 0, x - 16: x + 16, y - 16: y + 16] += mus.reshape(im.shape[0], 32, 32).cpu()
			rec_params[:, 1, x - 16: x + 16, y - 16: y + 16] += sigmas.reshape(im.shape[0], 32, 32).cpu()

			

	rec_losses[:, 16:-16] /= 2
	rec_losses[:, :, 16:-16] /= 2

	rec_params[:, :, 16:-16] /= 2
	rec_params[:, :, :, 16:-16] /= 2

	# mus[:, 16:-16] /= 2
	# mus[:, :, 16:-16] /= 2

	# sigmas[:, 16:-16] /= 2
	# sigmas[:, :, 16:-16] /= 2

	return rec_losses.cuda(), rec_params.unsqueeze(1).cuda()


def get_vae_features(model, patch_im, K):

	model.eval()
	rec_loss, kl, pr, rec_params = model.features(patch_im, middle_mask=False, K=25)
	rec_loss = MedianPool2d(stride=1, kernel_size=5, padding=2)(rec_loss)


	return rec_loss, kl, pr, rec_params.mean(1)

def get_subsets(S, mus, sigmas, seeds, n, z, labels, rec_losses):

	# def route_score(S, mus, sigmas, route):
		# return score_fct(S[])



	coords = np.indices(S.shape)

	pixel_scores = torch.zeros_like(S)

	T_x = S / mus
	global_max_score = float("-inf")

	for i, j in tqdm(seeds):

		pq = queue.PriorityQueue()

		q = [(i, j)]

		pq.put((0, (i, j)))

		route0 = []
		route1 = []


		max_score = float("-inf")


		visited = {}

		iteration = 0
		# print("=============")
		while not pq.empty() and iteration < 200:
			current_prio, coord = pq.get()
			# print(current_prio)
			iteration += 1
			# print(n)
			# coord = q.pop()


			route = np.array([route0 + [coord[0]], route1 + [coord[1]]])

			score, q = score_fct(S[route], mus[route], sigmas[route])
			

			if score > max_score:
				# print("mmmmmmm")
				max_score = score
				# print(score, len(route0))

				route0.append(coord[0])
				route1.append(coord[1])

				update_scores(pq, S, mus, sigmas, route)


				left = max(coord[0] - 1, 0), coord[1]
				right = min(coord[0] + 1, 191), coord[1]
				down = coord[0], max(coord[1] - 1, 0)
				up = coord[0], min(coord[1] + 1, 191)


				if not left in visited:
					# q.append(left)
					left_route = np.array([route0 + [left[0]], route1 + [left[1]]])
					left_score, _ = score_fct(S[left_route], mus[left_route], sigmas[left_route])
					pq.put((left_score, left))
					visited[left] = 1
				
				if not right in visited:
					# q.append(right)
					right_route = np.array([route0 + [right[0]], route1 + [right[1]]])
					right_score, _ = score_fct(S[right_route], mus[right_route], sigmas[right_route])
					pq.put((right_score, right))
					visited[right] = 1
				
				if not up in visited:
					# q.append(up)
					up_route = np.array([route0 + [up[0]], route1 + [up[1]]])
					up_score, _ = score_fct(S[up_route], mus[up_route], sigmas[up_route])
					pq.put((up_score, up))
					visited[up] = 1

				if not down in visited:
					# q.append(down)
					down_route = np.array([route0 + [down[0]], route1 + [down[1]]])
					down_score, _ = score_fct(S[down_route], mus[down_route], sigmas[down_route])
					pq.put((down_score, down))
					visited[down] = 1

				# left = max(coord[0] - 2, 0), coord[1]
				# right = min(coord[0] + 2, 191), coord[1]
				# down = coord[0], max(coord[1] - 2, 0)
				# up = coord[0], min(coord[1] + 2, 191)

				# if not left in visited:
				# 	# q.append(left)
				# 	left_route = np.array([route0 + [left[0]], route1 + [left[1]]])
				# 	left_score = score_fct(S[left_route], mus[left_route], sigmas[left_route])
				# 	pq.put((left_score, left))
				# 	visited[left] = 1
				
				# if not right in visited:
				# 	# q.append(right)
				# 	right_route = np.array([route0 + [right[0]], route1 + [right[1]]])
				# 	right_score = score_fct(S[right_route], mus[right_route], sigmas[right_route])
				# 	pq.put((right_score, right))
				# 	visited[right] = 1
				
				# if not up in visited:
				# 	# q.append(up)
				# 	up_route = np.array([route0 + [up[0]], route1 + [up[1]]])
				# 	up_score = score_fct(S[up_route], mus[up_route], sigmas[up_route])
				# 	pq.put((up_score, up))
				# 	visited[up] = 1

				# if not down in visited:
				# 	# q.append(down)
				# 	down_route = np.array([route0 + [down[0]], route1 + [down[1]]])
				# 	down_score = score_fct(S[down_route], mus[down_route], sigmas[down_route])
				# 	pq.put((down_score, down))
				# 	visited[down] = 1
			# i += 1


		# print(n)
		for i in range(len(route0)):
			coord = route0[i], route1[i]

			if pixel_scores[coord] < max_score:
				pixel_scores[coord] = max_score


		if max_score > global_max_score:
			global_max_score = max_score
			print(max_score,"-----", iteration, q)

			canvas = S.unsqueeze(0).repeat(3, 1, 1) * 255

			mask = canvas.clone()

			mask[0, route[0], route[1]] = 255
			# print(labels.shape)
			a = labels.reshape(1, 192, 192).repeat(3, 1, 1).cpu() * 255
			# print(labels.shape)

			# labels = labels.unsqueeze(0).repeat(3, 1, 1) * 255
			# print(labels.shape)
			priority_map = (S / mus).unsqueeze(0).repeat(3, 1, 1).cpu() / torch.max(S/mus) * 255
			# priority_map = sigmas.unsqueeze(0).repeat(3, 1, 1).cpu() * 255

			mus_map = mus.unsqueeze(0).repeat(3, 1, 1).cpu() * 255

			rec_losses_map = (rec_losses / torch.max(rec_losses)).unsqueeze(0).repeat(3, 1, 1).cpu() * 255
			sigmas_map = sigmas.unsqueeze(0).repeat(3, 1, 1).cpu() / torch.max(sigmas) * 255
			# print(sigmas)
			canvas = torch.cat((canvas.cpu(), mask.cpu(), a, priority_map, mus_map.cpu(), sigmas_map, rec_losses_map), 1)
			canvas = np.rollaxis(canvas.cpu().numpy().astype(np.uint8), 0, 3)


			Image.fromarray(canvas).save("ltss_figs/test_{}_{}.png".format(z, n))

		# print(j, "-------------------------------------------------")

		# print("outter---------------------------------------------")


	pixel_scores[pixel_scores == 0] = torch.min(pixel_scores[pixel_scores > 0])
	print(torch.min(pixel_scores))
	return pixel_scores

	# score_fct(mus, sigmas)





parser = ArgumentParser(description='')

parser.add_argument('--model', type=str, action='store', required=True)
parser.add_argument('--model_dir', type=str, action='store', required=True)
parser.add_argument('--ltss', action='store_true')
parser.add_argument('--vae', action='store_true')
parser.add_argument('--dense', action='store_true')


parser.add_argument('--channels', type=int, action='store', required=True)
parser.add_argument('--intensity', type=float, action='store', required=True)
parser.add_argument('--sigma', type=float, action='store', required=True)
parser.add_argument('--z_dim', type=int, action='store', required=True)
parser.add_argument('--body_part', type=str, action='store', required=True)


parser.add_argument('--exp', type=str)
args = parser.parse_args()


location = 'cuda'
checkpoint = torch.load(args.model_dir + "/" + args.model,
						map_location=location)

model_module = import_module(args.model_dir + ".model")


# import mask generator
mask_generator = model_module.mask_generator


# build VAEAC on top of the imported networks
# model = VAE(
# 	model_module.reconstruction_log_prob,
# 	model_module.encoder_network_hm,
# 	model_module.generative_network_hm,
# )


# model = VAE(
# 	model_module.reconstruction_log_prob,
# 	model_module.encoder_network,
# 	model_module.generative_network,
# )

if args.vae:
	if args.dense:
		proposal_network, generative_network = model_module.get_dense_vae_networks(args.channels, args.z_dim)
	else:
		proposal_network, generative_network = model_module.get_vae_networks(args.channels, args.z_dim)
	model = VAE(
		model_module.reconstruction_log_prob,
		proposal_network, 
		generative_network,
		# mask_generator,
		channels=args.channels
	)

else:
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



model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# validation_iwae = checkpoint['validation_iwae']
# train_vlb = checkpoint['train_vlb']

model = model.cuda()
model.eval()

validation_dataset = BodyPartDataset(split="test", all_slices=True, n_slices=args.channels, body_part=args.body_part, nodule=True, real_labels=True)

# pixel_losses = []
# labels = []

bs = 512
# pixel_losses = np.zeros((len(validation_dataset), 11, 11))
# kl_losses = np.zeros((len(validation_dataset), 11, 11))
# klpr_losses = np.zeros((len(validation_dataset), 11, 11))
pixel_losses = []
kl_losses = []
# klpr_losses = []

# rec_losses = np.zeros((len(validation_dataset), 15, 192, 192))
 
# rec_losses = []
# labels = np.zeros((len(validation_dataset), 11, 11))
labels = []
volume_n_labels = []
# images = torch.zeros((len(validation_dataset), 3, 192, 192))
# diffs = torch.zeros((len(validation_dataset), 3, 192, 192))
# predicts = 

f = open("{}_{}_{}.csv".format(args.model, args.intensity, args.sigma), "w")

for i in tqdm(range(0, len(validation_dataset), bs)):
	with torch.no_grad():

		volume, batch_labels, volume_n = validation_dataset[i:i+bs]
		volume = volume.cuda()
		# volume, loc = insert_nodule(volume, n_nodule=5)

		nodule_idx = batch_labels.reshape(batch_labels.shape[0], -1).sum(-1) > 0


		# ONLY SELECT NODULE SLICES
		volume = volume[nodule_idx]
		batch_labels = batch_labels[nodule_idx]
		volume_n = volume_n[nodule_idx]
		if volume.shape[0] == 0:
			continue
		# for z in tqdm(range(10, 32)):
		for z in [0]:


			# im = volume[:, z - (args.channels // 2): z + (args.channels // 2 + 1) - ((args.channels + 1) % 2)]
			im = volume
			patch_im = im

			# slice_labels = batch_labels[:, z]
			slice_labels = batch_labels

			heatmap = torch.zeros_like(im).cpu()[:, :3]



			# for group in ["nodule", "healthy"]:
			for group in ["all"]:

				# if rng > 0.95:
				# 	diseased =
				# if group == "all":

				# elif group == "nodule":
				# 	rng = random.random()
				# 	# if rng < 0.8:
				# 	# 	continue

				# 	# patch_im, batch_labels, vols = insert_nodule_patch(im, x, y, args.intensity, args.sigma)
				# 	patch_im, batch_labels, vols = insert_nodule_patch(im, args.intensity, args.sigma)


				# 	# print(len(batch_labels), batch_labels[0].shape)
				# 	# print(torch.stack(batch_labels, 0).shape, "----")

				# 	batch_labels = torch.stack(batch_labels, 0)#[:, :, x - 16: x + 16, y - 16: y + 16]
				# 	# patch_im = patch_im[np.array(batch_labels) == 1]

				# 	# batch_labels = [1] * patch_im.shape[0]
				# 	# if patch_im.shape[0] > 10:
				# 	# 	Image.fromarray((patch_im[10, 1] * 255).cpu().numpy().astype(np.uint8)).save("nodule_im/{}_{}_{}_{}.png".format(j, k, args.intensity, args.sigma))


				# else:
				# 	patch_im = im

				# if patch_im.shape[0] == 0:
				# 	continue

				# predict = model.generate_samples_params(im, mask)[:, 0, 0:3]

				# =====================
				
				# VAE
				# rec_loss, kl, pr, rec_params = model.features(patch_im, middle_mask=False, K=1)

				if args.vae:
					rec_loss, kl, pr, rec_params = get_vae_features(model, patch_im, K=25)
				else:
					# VAEAC
					rec_loss, rec_params = sliding_window_features(model, patch_im, K=25)


				pixel_losses.append(rec_loss)


				# nodule_in_slice = []
				# for s in range(slice_labels.shape[0]):
				# 	if slice_labels[s].sum() > 0:

				# 		nodule_in_slice.append(torch.ones((1, 1, 192, 192))) * nodule_i
				# 		nodule_i += 1
				# 	else:
				# 		nodule_in_slice.append(torch.zeros((1, 1, 192, 192)))

				# nodule_in_slice = torch.cat(nodule_in_slice, 0)
				# nodule_in_slice = slice_labels[]

				labels.append(slice_labels)
				volume_n_labels.append(volume_n)


				# =====================


				# rec_loss = MedianPool2d(stride=1, kernel_size=5, padding=2)(rec_loss.reshape(118, 1, 192, 192)).reshape(-1)

				# print(rec_params.shape)
				rec_loss = -rec_loss
				# print(torch.min(rec_loss), torch.max(rec_loss), "-----------------------")
				
				for n in range(batch_labels.shape[0]):
				# for n in range(6):

					# LTSS

					# print(rec_params[n].shape)

					S = patch_im[n].squeeze(0)
					mus = rec_params[n].squeeze(0)[0]
					sigmas = rec_params[n].squeeze(0)[1]
				



					priority = priority_fct(S, mus, sigmas)

					# seeds0 = scipy.signal.argrelextrema(priority.cpu().numpy(), np.greater, axis=0, order=2)
					# seeds1 = scipy.signal.argrelextrema(priority.cpu().numpy(), np.greater, axis=1, order=2)

					seeds0 = scipy.signal.argrelextrema(priority.cpu().numpy(), np.greater, axis=0, order=2)
					seeds1 = scipy.signal.argrelextrema(priority.cpu().numpy(), np.greater, axis=1, order=2)

					# print((S.cpu() > mus.cpu()).shape, mus.shape, S.shape)

					seeds_g = np.indices(S.shape)[:, S.cpu() > mus.cpu()]

					seeds0 = set([(seeds0[0][i], seeds0[1][i]) for i in range(len(seeds0[0]))])
					seeds1 = set([(seeds1[0][i], seeds1[1][i]) for i in range(len(seeds1[0]))])
					seeds_g = set([(seeds_g[0][i], seeds_g[1][i]) for i in range(len(seeds_g[0]))])



					seeds = list(seeds0.intersection(seeds1).intersection(seeds_g))
					seeds = sorted(seeds, key=lambda x: x[0])

					# print(seeds, "----------------------------")
					# print(len(seeds0), len(seeds1),len(seeds0.intersection(seeds1)))
					# print(priority.shape)




					max_dice = 0
					max_i = 0
					s = 0


					for i in range(1):
						pass
						# print(rec_loss.shape, slice_labels.shape)

						pred_mask = (rec_loss[n] > i * 0.01 - 6).float()
						label_mask = (slice_labels[n] > 0).float()
						# print(torch.sum(label_mask))

						if torch.sum(label_mask) > 0:
							# print(seeds)
							# print(label_mask.shape)
							# print(rec_loss.shape, "rrrrrrrr")

							
							dice = dice_coeff(pred_mask.cuda(), label_mask.cuda())

							canvas = patch_im[n, 0].cpu().unsqueeze(0).repeat(3, 1, 1) * 255

							# canvas = patch_im[0, 1].cpu().unsqueeze(0).repeat(3, 1, 1) * 255


							pred = pred_mask.cpu()#.repeat(3, 1, 1) * 255
							lab = label_mask.cpu()#.repeat(3, 1, 1) * 255


							tp = pred * lab
							fp = pred * (1 - lab)

							tn = (1 - pred) * (1 - lab)
							fn = (1 - pred) * lab

							# print(pred.shape, lab.shape, tp.shape)
							# print(pred, "------------------")
							# print(lab, "=======================")

							# print(torch.sum(tp), torch.sum(fp), torch.sum(tn), torch.sum(fn), "=========================")
							rl_auroc, rl_auprc, rl_sensitivity, rl_specificity = compute_stats(rec_loss[n].reshape(-1).cpu().numpy(), label_mask.reshape(-1).cpu().numpy())

							# auroc = pixel_scores.reshape(-1)[label_mask.reshape(-1)]

							print("rec loss: auroc", rl_auroc, "auprc", rl_auprc, "sensitivity", rl_sensitivity, "specificity", rl_specificity)

							if args.ltss:

								pixel_scores = get_subsets(S, mus, sigmas, seeds, n, z, label_mask, rec_loss[n])
								auroc, auprc, sensitivity, specificity = compute_stats(pixel_scores.reshape(-1).cpu().numpy(), label_mask.reshape(-1).cpu().numpy())
								print("auroc", auroc, "auprc", auprc, "sensitivity", sensitivity, "specificity", specificity)

							# print(i, dice)
							if dice > max_dice:
								max_dice = dice
								max_i = i
								s = torch.sum(tp), torch.sum(fp), torch.sum(tn), torch.sum(fn)


							tp = tp.repeat(3, 1, 1) * 255
							fp = fp.repeat(3, 1, 1) * 255
							tn = tn.repeat(3, 1, 1) * 255
							fn = fn.repeat(3, 1, 1) * 255


							canvas = torch.cat((canvas, tp, fp, fn), 1)
							canvas = np.rollaxis(canvas.numpy().astype(np.uint8), 0, 3)


							Image.fromarray(canvas).save("seg_figs/{}_{}_{}.png".format(n, z, i))
					# print(torch.sum(label_mask))
					# print("z", z, "n", n, "max i", max_i, "dice", max_dice, "tp fp tn fn", s)
					# print(pred)
					# print(lab, "-------------------")

				# print(torch.max(rec_loss), torch.min(rec_loss))
				# print(rec_loss.shape, "----------------")

				# predict = rec_params[:, 0, :args.channels]
				# # _, predict = model.batch_iwae(im, mask, 1)

				# # print(predict.shape)

				# # predict = predict[:, :]#.unsqueeze(1)

				# # print(predict.shape, "======")
				# # rec_losses[:, z - 10, x - 16: x + 16, y - 16: y + 16] += rec_loss.cpu().numpy()

				# predict = predict.clamp(0, 1)


				# # Image.fromarray((predict[0, 1] * 255).cpu().numpy().astype(np.uint8)).save("predictions/{}_{}.png".format(j, k))


				# # print(torch.max(predict), torch.max(im))

				# # diff = (patch_im - predict)[:, 0:3] ** 2
				# # print(diff.shape)
				# # pixel_loss = diff[:, :, x - 16: x + 16, y - 16: y + 16].cpu()
				# # pixel_loss = ((im[:, 1:2] - predict)[:, :, x - 16: x + 16, y - 16: y + 16] ** 2).cpu()

				# # print(torch.sum(predict[:, 0, x - 16: x + 16, y - 16: y + 16]), "------")

				# # print(torch.sum(heatmap[:, 0, x - 16: x + 16, y - 16: y + 16]), "=======")
				# # heatmap[i: i + bs, :, x - 16: x + 16, y - 16: y + 16] += pixel_loss
				# # print(torch.sum(heatmap[:, 1, x - 16: x + 16, y - 16: y + 16]), "=========")

				# # Image.fromarray((predict[0, 1, x - 16: x + 16, y - 16: y + 16] * 255).cpu().numpy().astype(np.uint8)).save("predictions/{}_{}.png".format(j, k))

				# # print(patch_im.shape)
				# canvas = patch_im[0, 0].cpu().unsqueeze(0).repeat(3, 1, 1) * 255

				# # canvas = patch_im[0, 1].cpu().unsqueeze(0).repeat(3, 1, 1) * 255


				# patch_predict = predict[0, 0].cpu().unsqueeze(0).repeat(3, 1, 1) * 255
				
				# diff = patch_predict.clone()
				# diff[0, rec_loss.reshape(len(validation_dataset), 192, 192)[0] > 0] = 255

				# nodule_mask = slice_labels.cpu().float().clone().reshape(len(validation_dataset), 1, 192, 192)[0].repeat(3, 1, 1) * 255
				# # patch_predict = predict[0, 1].cpu().unsqueeze(0).repeat(3, 1, 1) * 255
				# canvas = torch.cat((canvas, patch_predict, diff, nodule_mask), 1)
				# canvas = np.rollaxis(canvas.numpy().astype(np.uint8), 0, 3)


				# Image.fromarray(canvas).save("predictions/vae_{}_{}.png".format(n, group))


				# # pl = - rec_loss.cpu()
				# # pl = torch.sum(pixel_loss, dim=(1, 2, 3))
				# # pixel_losses[i: i + bs, j, k] = pl
				# # kl_losses[i: i + bs, j, k] = kl.cpu()
				# # klpr_losses[i: i + bs, j, k] = (kl + pr).cpu()


				# # volume_n += vols

				# # if group == "nodule":
				# # 	labels += batch_labels
				# # else:
				# # 	labels += [0] * pl.shape[0]
				# if group == "all":
				# 	plt.clf()
				# 	fig, axes = plt.subplots(nrows=2)
				# 	ax0, ax1 = axes.flatten()
					

				# 	_ = ax0.hist(batch_labels.reshape(-1), density=False, alpha=1)

				# 	plt.tight_layout()
				# 	plt.savefig("hist_batch_labels.png")
				# 	plt.close()


				# 	# print((slice_labels > 100).shape)
				# 	# print(torch.max(slice_labels))
				# 	# positive_rec_loss = rec_loss.reshape(-1)[(slice_labels > 80).reshape(-1).bool()]
				# 	# labels.append(torch.ones(positive_rec_loss.reshape(-1).shape[0]))
				# 	# pixel_losses.append(positive_rec_loss.cpu())


				# 	# negative_rec_loss = rec_loss.reshape(-1)[(slice_labels <= 80).reshape(-1).bool()]
				# 	# labels.append(torch.zeros(negative_rec_loss.reshape(-1).shape[0]))
				# 	# pixel_losses.append(negative_rec_loss.cpu())


				# elif group == "nodule":
				# 	# print(rec_loss.shape, batch_labels.shape)
				# 	# if batch_labels != []:
				# 	# print(rec_loss.shape, batch_labels.shape)

				# 	rec_loss = rec_loss.reshape(-1)[(slice_labels > 0).reshape(-1).bool()]
				# 	# labels += batch_labels
				# 	labels.append(torch.ones(rec_loss.reshape(-1).shape[0]))
				# else:
				# 	rec_loss = rec_loss.reshape(-1)
				# 	labels.append(torch.zeros(rec_loss.shape[0]))


				# # pixel_losses.append(rec_loss.cpu())
				# # kl_losses += kl.cpu().tolist()


				# # pixel_losses += list(pl)

				# # for t in range(len(pl)):

				# # 	if pl[t] > 10000:

				# # 		canvas = torch.cat((patch_im[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(),
				# # 			predict[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(), 
				# # 			diff[t, 1, x - 16: x + 16, y - 16: y + 16].cpu()),
				# # 			# heatmap[0, 0, x - 16: x + 16, y - 16: y + 16].cpu(),)
				# # 			1)


				# # 		Image.fromarray((canvas * 255).cpu().numpy().astype(np.uint8)).save("patch_predictions_nodule/{}_{}_{}_1_{}.png".format(t, j, k, pl[t]))

				# 	# # print(pixel_loss.shape)
				# 	# # pixel_losses[i: i + bs, j, k] = pl

				# 	# if nodule_in_patch(loc[t], z, x, y):
				# 	# # if (abs(loc[t][0] - z) <= 1 and abs(loc[t][1] - x) <= 12 and abs(loc[t][2] - y) <= 12):
				# 	# # if loc[t][0] == z and abs(loc[t][1] - x) <= 12 and abs(loc[t][2] - y) <= 12:

				# 	# 	nodule_pres = 1
				# 	# 	labels[t, j, k] = 1

				# 	# 	canvas = torch.cat((im[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(),
				# 	# 	predict[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(), 
				# 	# 	diff[t, 1, x - 16: x + 16, y - 16: y + 16].cpu()),
				# 	# 	# heatmap[0, 0, x - 16: x + 16, y - 16: y + 16].cpu(),)
				# 	# 	1)


				# 	# 	# if t == 0:
				# 	# 	Image.fromarray((canvas * 255).cpu().numpy().astype(np.uint8)).save("patch_predictions_nodule/{}_{}_{}_1_{}.png".format(t, j, k, pl[t]))
				# 	# 	# Image.fromarray((canvas * 255).cpu().numpy().astype(np.uint8)).save("patch_predictions_nodule/1_{}.png".format(pl[t]))

				# 	# 	# print("1111111")

				# 	# elif t == 0:
				# 	# # elif pl[t] > 10000:
				# 	# 	canvas = torch.cat((im[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(),
				# 	# 	predict[t, 1, x - 16: x + 16, y - 16: y + 16].cpu(), 
				# 	# 	diff[t, 1, x - 16: x + 16, y - 16: y + 16].cpu()),
				# 	# 	# heatmap[0, 0, x - 16: x + 16, y - 16: y + 16].cpu(),)
				# 	# 	1)
				# 	# 	Image.fromarray((canvas * 255).cpu().numpy().astype(np.uint8)).save("patch_predictions_nodule/0_{}.png".format(pl[t]))
				# 	# 	labels[t, j, k] = 0
						
				# 	# else:

				# 	# 	nodule_pres = 0

				# 	# 	labels[t, j, k] = 0



				# 		# for l in range(predict.shape[0]):
				# 		# 	Image.fromarray((canvas * 255).cpu().numpy().astype(np.uint8)).save("patch_predictions/{}_{}_{}_{}.png".format(l, j, k, nodule_pres))


				# 		# print(len(pixel_losses), len(labels))


				# # # im = volume[i]
				# # for n in range(i, i + predict.shape[0]):
				# # 	heat = make_heatmap(heatmap[n, 1], im[n, 1].cpu())
					
				# # 	heat = heat.clamp(0, 1)

				# # 	# heat = (np.transpose(heat.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
				# # 	heat = (np.transpose(heat.numpy(), (1, 2, 0))* 255).astype(np.uint8)

				# # 	Image.fromarray(heat).save("heatmaps/{}_{}.png".format(n, z))
				# # print(pixel_losses.shape, labels.shape)
				# # labels = torch.concatenate(pixel_losses, 0)
				# # labels = torch.concatenate(pixel_losses, 0)

				# # histogram(torch.cat(pixel_losses), torch.cat(labels), "test.png")
				# # histogram(torch.cat(pixel_losses, 0), torch.cat(labels, 0), "{}_pl_{}.png".format(args.model, z))



			# print(pixel_losses)
			# histogram(torch.cat(pixel_losses, 0), torch.cat(labels, 0), "{}_pl_{}.png".format(args.model, z))
			# histogram(kl_losses, labels, "{}_kl_{}.png".format(args.model, z)	)
			# histogram(klpr_losses.reshape(-1), labels.reshape(-1), "{}_klpr_hist_{}.png".format(args.model, z), upper=50)

				

			# # loglik_auroc = metrics.roc_auc_score(y_all, loglik4all)    
			# pixel_auroc = metrics.roc_auc_score(labels.reshape(-1), pixel_losses.reshape(-1))
			# fpr, tpr, thresholds = metrics.roc_curve(labels.reshape(-1), pixel_losses.reshape(-1))
			# youden_idx = np.argmax(tpr - fpr)
			# sensitivity = tpr[youden_idx]
			# specificity = 1 - fpr[youden_idx]
			# # loglik_auprc = metrics.average_precision_score(y_all, loglik4all)
			# precision, recall, thresholds = metrics.precision_recall_curve(labels.reshape(-1), pixel_losses.reshape(-1))
			# pixel_auprc = metrics.auc(recall, precision)

			# pixel_auroc, pixel_auprc, pixel_sensitivity, pixel_specificity = compute_stats(pixel_losses, labels, plot=True)
			# # kl_auroc, kl_auprc, kl_sensitivity, kl_specificity = compute_stats(kl_losses, labels)
			# # klpr_auroc, klpr_auprc, klpr_sensitivity, klpr_specificity = compute_stats(klpr_losses.reshape(-1), labels.reshape(-1))

			# # pixel_auprc = metrics.average_precision_score(labels.reshape(-1), pixel_losses.reshape(-1))
			# # print("specificity:", specificity)
			# # print("sensitivity:", sensitivity)
			# # print("auroc:", pixel_auroc)
			# # print("auprc:", pixel_auprc)


			# a, b, c, d = bootstrap(pixel_losses, labels, p=0.001)
			# s = "p=0.001 p_specificity: {:.2f} {:.2f} p_sensitivity: {:.2f} {:.2f} p_auroc: {:.2f} {:.2f} p_auprc: {:.2f} {:.2f}\n".format(np.mean(c), np.std(c), np.mean(d), np.std(d), np.mean(a), np.std(a), np.mean(b), np.std(b))
			# print(s)
			# f.write(s)
			# a, b, c, d = bootstrap(pixel_losses, labels, p=0.01)
			# s = "p=0.01 p_specificity: {:.2f} {:.2f} p_sensitivity: {:.2f} {:.2f} p_auroc: {:.2f} {:.2f} p_auprc: {:.2f} {:.2f}\n".format(np.mean(c), np.std(c), np.mean(d), np.std(d), np.mean(a), np.std(a), np.mean(b), np.std(b))
			# print(s)
			# f.write(s)
			# a, b, c, d = bootstrap(pixel_losses, labels, p=0.1)
			# s = "p=0.1 p_specificity: {:.2f} {:.2f} p_sensitivity: {:.2f} {:.2f} p_auroc: {:.2f} {:.2f} p_auprc: {:.2f} {:.2f}\n".format(np.mean(c), np.std(c), np.mean(d), np.std(d), np.mean(a), np.std(a), np.mean(b), np.std(b))
			# print(s)
			# f.write(s)
				# auprc_plot(precision, recall)
			# print(z)
			# print("p_specificity: {} p_sensitivity: {} p_auroc: {} p_auprc: {} kl_specificity: {} kl_sensitivity: {} kl_auroc: {} kl_auprc: {}\n".format(
				# pixel_specificity, pixel_sensitivity, pixel_auroc, pixel_auprc, kl_specificity, kl_sensitivity, kl_auroc, kl_auprc))
			# print("kl_specificity: {} kl_sensitivity: {} kl_auroc: {} kl_auprc: {}\n".format(kl_specificity, kl_sensitivity, kl_auroc, kl_auprc))
			# print("klpr_specificity: {} klpr_sensitivity: {} klpr_auroc: {} klpr_auprc: {}\n".format(klpr_specificity, klpr_sensitivity, klpr_auroc, klpr_auprc))

			# f.write("p_specificity: {} p_sensitivity: {} p_auroc: {} p_auprc: {} kl_specificity: {} kl_sensitivity: {} kl_auroc: {} kl_auprc: {}\n".format(
				# pixel_specificity, pixel_sensitivity, pixel_auroc, pixel_auprc, kl_specificity, kl_sensitivity, kl_auroc, kl_auprc))
			# f.write("kl_specificity: {} kl_sensitivity: {} kl_auroc: {} kl_auprc: {}\n".format(kl_specificity, kl_sensitivity, kl_auroc, kl_auprc))
			# f.write("klpr_specificity: {} klpr_sensitivity: {} klpr_auroc: {} klpr_auprc: {}\n".format(klpr_specificity, klpr_sensitivity, klpr_auroc, klpr_auprc))



f.close()




np.save("{}.labels.npy".format(args.model), torch.cat(labels, 0).cpu().numpy())
np.save("{}.volume_n_labels.npy".format(args.model), torch.cat(volume_n_labels, 0).cpu().numpy())
np.save("{}.losses.npy".format(args.model), torch.cat(pixel_losses, 0).cpu().numpy())
# np.save("{}_{}_{}.vols.npy".format(args.model, args.intensity, args.sigma), np.array(volume_n))
