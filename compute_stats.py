import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import sklearn.metrics


def bootstrap(losses, labels, p=0.1):


    labels = np.array(labels)
    losses = np.array(losses)

    print(losses.shape)
    print(labels.shape)
    nodule_losses = losses[labels == 1]
    healthy_losses = losses[labels == 0]

    print(nodule_losses.shape)
    print(healthy_losses.shape)


    aurocs, auprcs, sens, specs = [], [], [], []
    # for p in proportions:
    # for i in range(100):
    if int(len(healthy_losses) * p) > len(nodule_losses):
        ls = nodule_losses
        print("no bootstrap")
    else:
        idx = np.random.choice(np.arange(len(nodule_losses)), replace=False, size=int(len(healthy_losses) * p))
        ls = nodule_losses[idx]
    # print(ls, healthy_losses)
    p_ls = np.concatenate((healthy_losses, ls))

    p_labs = np.concatenate((np.zeros(len(healthy_losses)), np.ones(len(ls))))




      # auroc, auprc, sensitivity, specificity = compute_stats(p_ls, p_labs, plot=i == 0)

    # aurocs.append(auroc)
      # auprcs.append(auprc)
      # sens.append(sensitivity)
      # specs.append(specificity)

    # return aurocs, auprcs, sens, specs
    return p_ls, p_labs


intensity = 0.3
sigma = 5

fn = "best_checkpoint_svae.tar_0.3_5.0.{}.npy"

# fn = "best_checkpoint_vaeac_hm.tar_0.3_5.0.{}.npy"
# fn = "best_checkpoint_dvae_hm.tar_0.3_5.0.{}.npy"
# fn = "best_checkpoint_sf96affine.tar_0.3_5.0.{}.npy"
print(fn)
# lax = np.load("best_checkpoint_chest_sf96affine.tar_{}_{}.0.labels.npy".format(intensity, sigma))
# loss = np.load("best_checkpoint_chest_sf96affine.tar_{}_{}.0.losses.npy".format(intensity, sigma))

lab = np.load(fn.format("labels"))
loss = np.load(fn.format("losses"))

# vols = np.load("best_checkpoint_sf96affine.tar_{}_{}.0.vols.npy".format(intensity, sigma))

# 121 * 15 * 118 = 214170
boot_loss, boot_lab = bootstrap(loss, lab, p=0.001)

fpr, tpr, _ = roc_curve(boot_lab, boot_loss)
roc_auc = auc(fpr, tpr)

print("auroc:", roc_auc)

cm = sklearn.metrics.confusion_matrix(lab, loss > 1000)
# print(cm.reshape(-1))
# print(cm)

# print(cm[1, 0])


# PLOT PRC
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(boot_lab, boot_loss)
auprc = sklearn.metrics.auc(recall, precision)
print("auprc:", auprc)

plt.step(recall, precision, where='post', label='{}_{} (area = {})'.format(intensity, sigma, auprc))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
# plt.title('AUPRC: {}'.format(auprc))
plt.legend(loc="lower right")
plt.xticks(np.arange(0, 1, 0.1))
plt.yticks(np.arange(0, 1, 0.1))

plt.savefig("test.png")