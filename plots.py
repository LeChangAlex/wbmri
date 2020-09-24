import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from tqdm import tqdm


losses = np.load("best_checkpoint_dvae512.tar.losses.npy")
labels = np.load("best_checkpoint_dvae512.tar.labels.npy")
volume_n = np.load("best_checkpoint_dvae512.tar.volume_n_labels.npy")

print(losses.reshape(-1).shape[0], "datapoints")


losses = losses.reshape(losses.shape[0], -1)
volume_n = volume_n.reshape(losses.shape[0], -1)
labels = labels.reshape(labels.shape[0], -1)

plot_losses = []
nodule_losses = []
nodule_volumes = []
x = []
# for i in tqdm(np.unique(volume_n)):
n = 1
for i in tqdm(np.unique(volume_n)):


	plot_losses.append(losses[volume_n == i].reshape(-1))

	label_count = losses[np.logical_and(volume_n == i, labels > 0)].reshape(-1).shape[0]
	if label_count > 0:
		print(i)
		nodule_losses.append(losses[np.logical_and(volume_n == i, labels > 0)].reshape(-1))

		x.append(n * np.ones_like(nodule_losses[-1]))
	n += 1

print("plotting")

# x = list(range(len(np.unique(volume_n)[:3])))
# Multiple box plots on one Axes
fig, ax = plt.subplots(figsize=(20, 20))
bp = ax.boxplot(plot_losses)
ax.scatter(np.concatenate(x), np.concatenate(nodule_losses), color="red")

ax.set_ylim([-300, 200])
print("saving")
plt.savefig("boxplot.png")