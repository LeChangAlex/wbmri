import numpy as np
import argparse
import imutils
import glob
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from skimage import exposure
from skimage.exposure import match_histograms
from argparse import ArgumentParser
from skimage import io
import torch
CUDA_LAUNCH_BLOCKING=1


parser = argparse.ArgumentParser()
parser.add_argument("--body_part", default="head", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=1550, type=int)


def dist_conv(input, weights, sf=0.25):

	input = torch.nn.functional.interpolate(input, scale_factor=(sf, sf), mode="bilinear")
	weights = torch.nn.functional.interpolate(weights, scale_factor=(sf, sf), mode="bilinear") 

	conv_layer = torch.nn.Conv2d(1, 1, kernel_size=weights.shape, bias=False)
	
	# conv = torch.nn.functional.conv2d(volume.unsqueeze(1), weight=resized) / resized.shape[2] / resized.shape[3]
	conv_layer.weight = torch.nn.Parameter(weights)
	conv = torch.nn.parallel.data_parallel(conv_layer, input, device_ids=range(1))

	conv = torch.nn.functional.interpolate(conv, scale_factor=(1/sf, 1/sf), mode="bilinear")

	return conv

def read_im(fn):
	im = io.imread(fn, as_gray=True)
	im = (im / 255).astype(np.float32)
	
	return im



def match_template(volume, template, lower_scale, upper_scale):
	with torch.no_grad():
		# match using pytorch
		volume_sq = volume ** 2
		max_val = 0

		sf = 0.2


		(tH, tW) = template.shape[:2]
		r = wbmri.shape[1] / tW



		found = None
		for scale in np.linspace(lower_scale * r, upper_scale * r, 10)[::-1]:
			try:	
				resized = imutils.resize(template, width=int(template.shape[1] * scale))
				# print("1")



				resized = torch.from_numpy(resized).cuda()
				resized = resized.unsqueeze(0).unsqueeze(0) / (torch.sum(resized ** 2) ** 0.5)
				
				conv = dist_conv(volume.unsqueeze(1), resized) 
				# conv = torch.nn.functional.conv2d(volume.unsqueeze(1), weight=resized) / resized.shape[2] / resized.shape[3]
				# conv = torch.nn.parallel.data_parallel(conv_layer, volume.unsqueeze(1), device_ids=range(4))
				# print(conv)


				# print(torch.isnan(conv).any())
				sum_t = torch.ones_like(resized)# / (resized.shape[2] * resized.shape[3]  * sf * sf)
				# print("3")
				patch_sums = dist_conv(volume_sq.unsqueeze(1), sum_t) 
				# patch_sums = torch.nn.functional.conv2d(volume_sq.unsqueeze(1), weight=sum_t) / resized.shape[2] / resized.shape[3]
				# print(torch.isnan(patch_sums).any())

				patch_sums_sqr = (patch_sums + 1e-5) ** 0.5
				# print("4")
				# print(torch.isnan(patch_sums_sqr).any())

				result = (conv / (patch_sums_sqr + 1e-5))

				# print("5")
				# print(result.dtype)
				# print(torch.isnan(result).any())

				# print(type(result.argmax()))
				# result = result.cpu()

				# print("6")
				idx = result.argmax()
				# print("7")
				# print(idx.item())
				idx = idx.item()
				# print("7")

				idx0 = idx // result.shape[2] // result.shape[3]
				idx1 = (idx - idx0 * result.shape[2] * result.shape[3]) // result.shape[3]
				idx2 = idx % result.shape[3]

				idx = [idx0, idx1, idx2]

				val = result[idx0, 0, idx1, idx2]
				# idx[1] += resized.shape[2] / 2
				# idx[2] += resized.shape[3] / 2
				# print(idx, val)
				# print("8")
				# print(val, idx)
				if val >= max_val:
					found = (val, idx, scale, resized.shape[2], resized.shape[3])
					max_val = val
			except:
				print(scale)
				pass
			
		print(found)
		
	return found


args = parser.parse_args() ## comment for debug

# BAD WB
# BAD = [
# 	16, 26, 34,35, 38, 40, 41, 69, 71, 73, 74, 110, 113, 129, 133, 136, 140, 144, 145, 161, 175, 188, 196, 211, 224, 232, 233, 235, 239, 252, 253, 256, 262, 264, 271, 272, 274, 330, 373, 375, 377, 378, 384, 389, 395, 396, 402, 438, 454, 458, 495, 

# ]

# volume 11 slice 27
# chest

if args.body_part == "chest":
	top_tp = read_im("/datasets/wbmri/crop_templates/top_crop_headless.png")
	mid_tp = read_im("/datasets/wbmri/crop_templates/mid_crop_headless.png")
	bot_tp = read_im("/datasets/wbmri/crop_templates/chest_bottom_crop.png")


elif args.body_part == "abdomen":
	template_fn = "abdomen_crop.png"
elif args.body_part == "head":
	template_fn = "head_crop.png"
elif args.body_part == "rl":
	template_fn = "right_leg_crop.png"
elif args.body_part == "ll":
	template_fn = "left_leg_crop.png"




# f = open("chest_headless_coords.csv", "w")
f = open("chest_preproc5_coords.csv", "w")

for i in tqdm(range(args.start, args.end)):
	# if i in BAD:
	# 	continue
		
	try:
		# vol_exist = cv2.imread("/datasets/wbmri/preproc5_crops/chest_{}.png".format(i))


		found = None
		
		n_slices = len(os.listdir("/datasets/wbmri/preproc5_bright/volume_{}".format(i)))
		volume = None
	except:
		continue

	for j in range(n_slices):
		try:
			wbmri = io.imread("/datasets/wbmri/preproc5_bright/volume_{}/{}.png".format(i, j), as_gray=True)
		except:
			continue
		if volume is None:
			volume = torch.zeros((n_slices, wbmri.shape[0], wbmri.shape[1]))

		if j > 10:
			volume[j] = torch.from_numpy(wbmri)

	volume = volume.cuda() / 255

	print(volume.shape)
	try:
		(val_top, idx_top, scale_top, h_top, w_top) = match_template(volume[:, :volume.shape[1] // 2], top_tp, lower_scale=0.3, upper_scale=1)
		(val_mid, idx_mid, scale_mid, h_mid, w_mid) = match_template(volume, mid_tp, lower_scale=0.3, upper_scale=0.8)

		# (val_bot, idx_bot, scale_bot, w_bot, h_bot) = match_template(volume, bot_tp)
	except:
		continue




	

	# for j in range(n_slices // 2, n_slices // 4 * 3):
	# 	# wbmri = cv2.imread("/datasets/wbmri/preproc5/volume_{}/{}.png".format(i, j)).astype(np.float32)

	# 	wbmri = io.imread("/datasets/wbmri/preproc5/volume_{}/{}.png".format(i, j), as_gray=True).astype(np.float32)

	# 	# wbmri = volume[j]
	# 	# print(wbmri.shape)
	# 	# plt.imshow(wbmri)
	# 	# plt.show()

	# 	(tH, tW) = template.shape[:2]
	# 	r = wbmri.shape[1] / tW

	# 	if args.body_part == "head":
	# 		wbmri = wbmri[:wbmri.shape[0] // 3, :]
	# 		upper_scale = 0.9
	# 		lower_scale = 0.5
	# 	elif args.body_part == "rl":
	# 		wbmri = wbmri[:, :wbmri.shape[1] // 2]
	# 		upper_scale = 0.5
	# 		lower_scale = 0.25
	# 	elif args.body_part == "ll":
	# 		wbmri = wbmri[:, wbmri.shape[1] // 2:]
	# 		upper_scale = 0.5
	# 		lower_scale = 0.25
	# 	else:				
	# 		upper_scale = 0.75
	# 		lower_scale = 0.5
	# 	# loop over the scales of the image


	# 	for scale in np.linspace(lower_scale * r, upper_scale * r, 5)[::-1]:
	# 		try:
	# 			resized = imutils.resize(template, width=int(template.shape[1] * scale))
	# 			# result = cv2.matchTemplate(wbmri, resized, cv2.TM_CCOEFF_NORMED)


	# 			print(type(resized), resized.shape)
	# 			# try:
	# 			result = cv2.matchTemplate(wbmri, resized, cv2.TM_CCOEFF_NORMED)


	# 			# except:
	# 			# 	print("fail")
	# 			# 	continue


	# 			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	# 			# check to see if the iteration should be visualized
	# 			# draw a bounding box around the detected region

	# 			if found is None or maxVal > found[0]:
	# 				found = (maxVal, maxLoc, scale, j, wbmri)
	# 				# print(found)
	# 				# print(found[:4])

	# 		except:
	# 			continue

	# print(i, found[:-1])

	fig, ax = plt.subplots(1, 3)
	
	# Display the image
	ax[0].imshow(volume.cpu()[idx_top[0]])
	ax[1].imshow(volume.cpu()[idx_mid[0]])
	ax[2].imshow(volume.cpu()[idx_mid[0]])
	
	# Create a Rectangle patch
	# rect = patchetests.Rectangle((found[1][0], found[1][1]), template.shape[1] * found[2], template.shape[0] * found[2], linewidth=2, edgecolor='r', facecolor='none')
	top_rect = patches.Rectangle((idx_top[2], idx_top[1]), w_top, h_top, linewidth=2, edgecolor='r', facecolor='none')
	mid_rect = patches.Rectangle((idx_mid[2], idx_mid[1]), w_mid, h_mid, linewidth=2, edgecolor='r', facecolor='none')
	# bot_rect = patches.Rectangle((idx_bot[2], idx_bot[1]), h_bot, w_bot, linewidth=2, edgecolor='r', facecolor='none')
	bot_rect = patches.Rectangle((idx_mid[2], idx_top[1] + h_top), w_mid, (idx_mid[1] - idx_top[1] - h_top + h_mid) * 1.8, linewidth=2, edgecolor='r', facecolor='none')
	
	# Add the patch to the Axes
	ax[0].add_patch(top_rect)
	ax[1].add_patch(mid_rect)
	ax[2].add_patch(bot_rect)


	# plt.savefig("/datasets/wbmri/preproc5_bright_crops/{}_{}".format(args.body_part, i))
	plt.clf()
	rect = ((idx_mid[2], idx_top[1] + h_top), w_mid, (idx_mid[1] - idx_top[1] - h_top + h_mid) * 1.8)

	line = str(i) + "," + str(rect) + "\n"
	f.write(line.replace("(", "").replace(")", ""))




	# os.makedirs("/datasets/wbmri/slice_png_{}_headless/{}_{}".format(args.body_part, i)))

f.close()

	# except:
	# 	continue

	# cv2.imwrite("new_wbmri/chest_wb/im_{}.png".format(i), volume[found[3],
	#                                                    found[1][1]: found[1][1] + found[4].shape[0],
	#                                                    found[1][0]: found[1][0] + found[4].shape[