import torch
from tqdm import tqdm
import wandb
from sklearn import metrics
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple, _quadruple

from numba  import jit
# import tensorflow as tf
import random

def extend_batch(batch, dataloader, batch_size):
	"""
	If the batch size is less than batch_size, extends it with
	data from the dataloader until it reaches the required size.
	Here batch is a tensor.
	Returns the extended batch.
	"""
	while batch.shape[0] != batch_size:
		dataloader_iterator = iter(dataloader)
		nw_batch = next(dataloader_iterator)
		print(nw_batch)
		if nw_batch.shape[0] + batch.shape[0] > batch_size:
			nw_batch = nw_batch[:batch_size - batch.shape[0]]
		batch = torch.cat([batch, nw_batch], 0)
	return batch


def extend_batch_tuple(batch, dataloader, batch_size):
	"""
	The same as extend_batch, but here the batch is a list of tensors
	to be extended. All tensors are assumed to have the same first dimension.
	Returns the extended batch (i. e. list of extended tensors).
	"""
	while batch[0].shape[0] != batch_size:
		dataloader_iterator = iter(dataloader)
		nw_batch = next(dataloader_iterator)
		if nw_batch[0].shape[0] + batch[0].shape[0] > batch_size:
			nw_batch = [nw_t[:batch_size - batch[0].shape[0]]
						for nw_t in nw_batch]
		batch = [torch.cat([t, nw_t], 0) for t, nw_t in zip(batch, nw_batch)]
	return batch

@jit
def fast_auc(y_true, y_prob):
	y_true = np.asarray(y_true)
	y_true = y_true[np.argsort(y_prob)]
	nfalse = 0
	auc = 0
	n = len(y_true)
	for i in tqdm(range(n)):
		y_i = y_true[i]
		nfalse += (1 - y_i)
		auc += y_i * nfalse
	auc /= (nfalse * (n - nfalse))
	return auc


def fast_auc2(y_prob, y_true, step=100):
	y_true = np.asarray(y_true)
	y_true = y_true[np.argsort(y_prob)]
	nfalse = 0
	auc = 0
	n = len(y_true)
	fp = 0
	tp = 0

	fp_list = []
	tp_list = []

	for i in tqdm(range(0, n, step)):
		# tp += np.sum(y_true[i:i + step])
		# fp += np.sum(1 - y_true[i:i + step])

		# fp_list.append(fp)
		# tp_list.append(tp)
		y_i_v = y_true[i:i + step]
		y_i = np.sum(y_i_v)
		nfalse += (step - np.sum(y_i_v))
		auc += y_i * nfalse
	auc /= (nfalse * (n - nfalse))

	return auc
	
def wb_mask(dataset, index, model, full_res_mask=False):


	# for i in range(len(dataset)):
	windows, full_im, window_features, (l_x, u_x, l_y, u_y), window_heights, slice_numbers, resized_h, window_widths, label, nodule_vol = dataset.ordered_windows(index)
	windows, window_features = windows.cuda(), window_features.cuda()

	with torch.no_grad():
		iwae, rec_loss, kls, window_recs, rec_loss_t = model.batch_iwae(windows, None, window_features, 1)


	radius = windows.shape[1] // 2
	resized_label = None	

	if dataset.body_part == "chest":
		predictions = torch.zeros((full_im.shape[0], resized_h, 256))
		recs = torch.zeros((full_im.shape[0], resized_h, 256))



		for i in range(len(slice_numbers)):

			s = slice_numbers[i]
			h = window_heights[i]
			
						
			predictions[s, h: h + 256, :] += rec_loss_t[i, radius].cpu()
			recs[s, h: h + 256, :] += window_recs[i, radius].cpu()

		
		# divide overlaps by 2
		heights = sorted(list(set(window_heights)))
		for i in range(len(heights) - 1):
			h = heights[i]
			next_h = heights[i+1]


			predictions[:, next_h :h+256] /= 2
			recs[:, next_h :h+256] /= 2
		if full_res_mask:
			recs = nn.functional.interpolate(recs.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2, u_x - l_x),
					mode="nearest").squeeze(0).squeeze(0)

			predictions = nn.functional.interpolate(predictions.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2, u_x - l_x),
					mode="nearest").squeeze(0).squeeze(0)

			label = nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2, u_x - l_x),
					mode="nearest").squeeze(0).squeeze(0)


	elif dataset.body_part == "legs":

		predictions = torch.zeros((full_im.shape[0], resized_h, 512))
		recs = torch.zeros((full_im.shape[0], resized_h, 512))



		for i in range(len(slice_numbers)):

			s = slice_numbers[i]
			h = window_heights[i]
			w = window_widths[i]
			
			predictions[s, h: h + 256, w: w + 256] += rec_loss_t[i, radius].cpu()
			recs[s, h: h + 256, w: w + 256] += window_recs[i, radius].cpu()

		
		# divide overlaps by 2
		heights = sorted(list(set(window_heights)))
		for i in range(len(heights) - 1):
			h = heights[i]
			next_h = heights[i+1]


			predictions[:, next_h :h+256] /= 2
			recs[:, next_h :h+256] /= 2


		if full_res_mask:

			predictions = nn.functional.interpolate(predictions.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2 + full_im.shape[1] % 2, full_im.shape[2]),
					mode="nearest").squeeze(0).squeeze(0)

			recs = nn.functional.interpolate(recs.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2 + full_im.shape[1] % 2, full_im.shape[2]),
					mode="nearest").squeeze(0).squeeze(0)
			label = nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0), 
				size=(full_im.shape[0], full_im.shape[1] // 2, u_x - l_x),
					mode="nearest").squeeze(0).squeeze(0)


	return full_im, predictions, (l_x, u_x, l_y, u_y), label, nodule_vol, recs




def fast_auprc(y_prob, y_true, step=100):
	y_true = np.asarray(y_true)
	y_true = y_true[np.argsort(y_prob)][::-1]
	nfalse = 0
	auc = 0
	n = len(y_true)
	
	tp = 0
	n_positive = 0
	for i in tqdm(range(0, n, step)):

		y_i_v = y_true[i:i + step]
		y_i = np.sum(y_i_v)
		tp += y_i
		precision = tp / (i + step)
		

		n_positive += y_i
		auc += y_i * precision
	  
	auc /= n_positive

	#     y_i_v = y_true[i:i + step]
	#     y_i = np.sum(y_i_v)
	#     nfalse += (step - np.sum(y_i_v))
	#     auc += y_i * nfalse

	# auc /= (nfalse * (n - nfalse))

	return auc


# def fast_auc2(y_true, y_prob, step=100):
#     y_true = np.asarray(y_true)
#     y_true = y_true[np.argsort(y_prob)]
#     nfalse = 0
#     auc = 0
#     n = len(y_true)
#     fp = 0
#     tp = 0

#     fp_list = []
#     tp_list = []

#     for i in tqdm(range(0, n, step)):
#         tp += np.sum(y_true[i:i + step])
#         fp += np.sum(1 - y_true[i:i + step])

#         fp_list.append(fp)
#         tp_list.append(tp)
#         # y_i_v = y_true[i:i + step]
#         # y_i = np.sum(y_i_v)
#         # nfalse += (step - np.sum(y_i_v))
#         # auc += y_i * nfalse
#     # auc /= (nfalse * (n - nfalse))

#     fpr = np.array(fp_list) / np.sum(1 - y_true)

#     tpr = np.array(tp_list) / np.sum(y_true)
#     return metrics.auc(fpr, tpr)


def approx_stats(y_true, y_prob, n_thresh=100):

	max_val = np.amax(y_prob)
	min_val = np.amin(y_prob)
	p = np.sum(y_true)
	n = np.sum(1 - y_true)

	tpr_list = []
	fpr_list = []
	for i in tqdm(range(n_thresh + 1)):
		threshold = min_val + (max_val - min_val) * i / n_thresh


		tp = (y_prob > threshold).astype(np.float16) * (y_true == 1).astype(np.float16)
		tpr = np.sum(tp) / p

		fp = (y_prob > threshold).astype(np.float16) * (y_true == 0).astype(np.float16)
		fpr = np.sum(tp) / n

		tpr_list.append(tpr)
		fpr_list.append(fpr)

	auroc = metrics.auc(tpr_list, fpr_list)
	print(auroc)


	return auroc


def compute_stats(losses, labels, plot=False):
	# loglik_auroc = metrics.roc_auc_score(y_all, loglik4all)    
	auroc = metrics.roc_auc_score(labels, losses)
	print("auroc", auroc)
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

class MedianPool3d(nn.Module):
	""" Median pool (usable as median filter when stride=1) module.
	
	Args:
		 kernel_size: size of pooling kernel, int or 2-tuple
		 stride: pool stride, int or 2-tuple
		 padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
		 same: override padding and enforce same padding, boolean
	"""
	def __init__(self, kernel_size=(3, 3, 3), stride=1, padding=0, same=False):
		super(MedianPool3d, self).__init__()
		self.k = kernel_size
		self.stride = _triple(stride)
		self.padding = _triple(padding) * 2  # convert to l, r, t, b
		self.same = same

	def forward(self, x):
		# using existing pytorch functions and tensor ops so that we get autograd, 
		# would likely be more efficient to implement from scratch at C/Cuda level

		x = F.pad(x, self.padding, mode='constant')
		x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1]).unfold(4, self.k[2], self.stride[2])
		x = x.contiguous().view(x.size()[:5] + (-1,)).median(dim=-1)[0]
		return x
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


def get_validation_score(val_dataset, model, post_proc=False, compute_auroc=True):

	first = {}

	flat_losses = []
	flat_labels = []

	mask_losses = []
	rand = int(random.random() * len(val_dataset))
	
	for i in tqdm(range(len(val_dataset))):

		full_im, loss_mask, (l_x, u_x, l_y, u_y), nodule_label, nodule_vol, nodule_recs = wb_mask(val_dataset, i, model)
	

		if post_proc:
			postproc_loss_mask = MedianPool3d(kernel_size=(3, 5, 5), stride=1, padding=2)(loss_mask.unsqueeze(0).unsqueeze(0))
			mask_losses.append(postproc_loss_mask.reshape(-1))
		else:
			mask_losses.append(loss_mask.reshape(-1))


		flat_losses.append(loss_mask.reshape(-1))


		flat_labels.append(nodule_label.reshape(-1))
		if i == rand:
			first["batch_nodule"] = nodule_vol
			first["batch_labels"] = nodule_label
			first["nodule_rec_loss_t"] = loss_mask
			first["nodule_recs"] = nodule_recs



	flat_losses = torch.cat(flat_losses)
	flat_labels = torch.cat(flat_labels)
	loss_mask = torch.cat(mask_losses)



	auroc = fast_auc2(loss_mask, flat_labels)
	auprc = fast_auprc(loss_mask, flat_labels)

	sensitivity = 0
	specificity = 0
	first["auroc"] = auroc
	first["auprc"] = auprc
	first["rec_loss"] = torch.mean(flat_losses)
	first["sensitivity"] = sensitivity
	first["specificity"] = specificity

	# first["channel_loss"] = channel_loss

	return first



def get_validation_iwae(val_dataloader, 
						batch_size,
						model, 
						num_samples, 
						verbose=False, 
						middle_mask=False,
						post_proc=False,
						compute_auroc=True):
	"""
	Compute mean IWAE log likelihood estimation of the validation set.
	Takes validation dataloader, mask generator, batch size, model (VAEAC)
	and number of IWAE latent samples per object.
	Returns one float - the estimation.
	"""
	model.eval()
	all_losses = []
	all_labels = []
	all_pixels = []

	d_z = {}
	d_age = {}
	d_weight = {}
	d_sex = {}

	prev_n = -1
	volume_losses_buffer = []
	volume_buffer = []
	volume_labels_buffer = []
	volume_recs_buffer = []




	first = {}
	cum_size = 0
	avg_iwae = 0
	avg_rl = 0

	iterator = val_dataloader
	if verbose:
		iterator = tqdm(iterator)

	channel_loss = None


	for (batch, metadata, batch_n, batch_z, batch_sex, batch_age, batch_weight, batch_nodule, batch_labels) in iterator:

		radius = batch.shape[1] // 2
		init_size = batch.shape[0]
		
		if channel_loss is None:
			channel_loss = torch.zeros(batch.shape[1])

		# batch = extend_batch(batch, val_dataloader, batch_size)
		# mask = mask_generator(batch)
		mask = None
		if next(model.parameters()).is_cuda:
			batch_nodule = batch_nodule.cuda()

			batch = batch.cuda()
			metadata = metadata.cuda()
		with torch.no_grad():
			
			iwae, rec_loss, kls, recs, rec_loss_t = model.batch_iwae(batch, mask, metadata, num_samples, middle_mask=middle_mask)
			_, _, _, nodule_recs, nodule_rec_loss_t = model.batch_iwae(batch_nodule, mask, metadata, num_samples, middle_mask=middle_mask)
			# rec_loss_avg - kl, rec_loss_avg, kl, rec_params, rec_loss
			# iwae = iwae[:init_size]
			iter_rec_loss = rec_loss.sum(-1).sum(-1).sum(0).cpu()
			# print(rec_loss.shape, iter_rec_loss.shape, "====================")

			channel_loss += iter_rec_loss


			avg_rl = (avg_rl * (cum_size / (cum_size + iwae.shape[0])) +
						rec_loss_t.sum()  / (cum_size + iwae.shape[0])) 


			avg_iwae = (avg_iwae * (cum_size / (cum_size + iwae.shape[0])) +
						iwae.sum() / (cum_size + iwae.shape[0]))
			cum_size += iwae.shape[0]

		

		if verbose:
			iterator.set_description('Validation IWAE: %g' % avg_iwae)

		if len(first) == 0:
			first["rec_loss_t"] = rec_loss_t
			first["kls"] = kls
			first["recs"] = recs
			first["batch"] = batch


			first["batch_nodule"] = batch_nodule
			first["batch_labels"] = batch_labels
			first["nodule_rec_loss_t"] = nodule_rec_loss_t
			first["nodule_recs"] = nodule_recs
			# rec_losses, kls, recs, batch, nodule_rec_losses, nodule_recs

		# print(nodule_rec_losses.shape)
		for j in range(batch.shape[0]):
			n = batch_n[j].item()
			z = batch_z[j].item()
			age = batch_age[j].item()
			sex = batch_sex[j].item()
			weight = batch_weight[j].item()
			

			if prev_n != -1 and n != prev_n:
				#### Post processing #####

				if post_proc:
					with torch.no_grad():
						volume_losses = torch.cat(volume_losses_buffer, 0).unsqueeze(0).unsqueeze(0)

						volume_losses = MedianPool3d(kernel_size=(5, 5, 5), stride=1, padding=2)(volume_losses)
				else:
					volume_losses = volume_losses_buffer

				all_pixels += volume_buffer
				all_losses += volume_losses_buffer
				all_labels += volume_labels_buffer


				volume_buffer = []
				volume_recs_buffer = []
				volume_losses_buffer = []
				volume_labels_buffer = []

			prev_n = n
			volume_recs_buffer.append(nodule_recs[j, radius])
			volume_losses_buffer.append(nodule_rec_loss_t[j, radius])
			volume_labels_buffer.append(batch_labels[j, radius])
			volume_buffer.append(batch_nodule[j, radius])


	if post_proc:
		with torch.no_grad():
			volume_losses = torch.cat(volume_losses_buffer, 0).unsqueeze(0).unsqueeze(0)

			volume_losses = MedianPool3d(kernel_size=(5, 5, 5), stride=1, padding=2)(volume_losses)
	else:
		volume_losses = volume_losses_buffer

	all_pixels += volume_buffer
	all_losses += volume_losses_buffer
	all_labels += volume_labels_buffer

	if compute_auroc:
		flat_losses = torch.cat(all_losses).reshape(-1).cpu().numpy()#[:10000000]
		flat_pixels = torch.cat(all_pixels).reshape(-1).cpu().numpy()#[:10000000]

	# flat_losses -= np.amin(flat_losses)
	# flat_losses /= np.amax(flat_losses)

		flat_labels = torch.cat(all_labels).reshape(-1).cpu().numpy()#[:10000000]
		# auroc, auprc, sensitivity, specificity = compute_stats(- flat_losses, flat_labels)

	# auroc_calc = tf.keras.metrics.AUC(
	#     num_thresholds=30, curve='ROC',
	#     summation_method='interpolation', name=None, dtype=None,
	#     thresholds=None, multi_label=False, label_weights=None
	# )
	# auroc_calc.update_state(flat_labels, flat_losses)
	# auroc = auroc_calc.result().numpy()


	# auprc_calc = tf.keras.metrics.AUC(
	#     num_thresholds=30, curve='PR',
	#     summation_method='interpolation', name=None, dtype=None,
	#     thresholds=None, multi_label=False, label_weights=None
	# )
	# auprc_calc.update_state(flat_labels, flat_losses)
	# auprc = auprc_calc.result().numpy()
		
	# print(auroc, "2222222")
	# auroc = fast_auc(flat_losses, flat_labels)
		# auroc = fast_auc2(flat_pixels, flat_labels)
		# auprc = fast_auprc(flat_pixels, flat_labels)
		auroc = fast_auc2(flat_losses, flat_labels)
		auprc = fast_auprc(flat_losses, flat_labels)
	else:

		auroc = 0
		auprc = 0

	first["auroc"] = auroc

	# print(auroc, "11111")
	sensitivity = 0
	specificity = 0
	first["val_iwae"] = float(avg_iwae)    
	first["rec_loss"] = float(avg_rl)
	first["auprc"] = auprc
	first["sensitivity"] = sensitivity
	first["specificity"] = specificity
	first["channel_loss"] = channel_loss


	return first

		# print(recs.shape, masks.shape, batch.shape, "----------------")
