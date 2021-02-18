# Inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/event_handling/image_slices_viewer.py

# Usage:
# python scroller.py VOLUME_DIR_PATH

# VOLUME_DIR_PATH is the path of the directory of the volume to scroll through,
# remove faulty slices, pre-process, and run the algorithm and save the stats.

# Example: python run_algorithm.py /home/abhishekmoturu/PycharmProjects/mri_gan_cancer/new_wbmri/diseased_slices

from __future__ import print_function

import os
import sys
import glob
import numpy as np
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from source.wbmri_crops import read_png_volume
# volume path
volume_path = sys.argv[1]
preproc_path = "preproc/" + volume_path.split("/")[-1][:-4]
pred_path = "predictions/" + volume_path.split("/")[-1][:-4]
# raw_path = "raw/volume_" + volume_path.split("_")[-1].split(".")[0] + ".npy"

rem_lst = []

# handle scrolling through volume
class IndexTracker(object):
    def __init__(self, ax, X, n):
        self.ax = ax
        self.n = n
        ax.set_title('scrolling through {}'.format(n))
        self.X = X
        if np.max(X) > 1:
            vmaximum = 255
        else:
            vmaximum = 1
        self.slices = X.shape[0]
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind], cmap='gray', vmin=0, vmax=vmaximum)
        self.update()

    def onscroll(self, event):
        if event.button == 'down' and self.ind < self.slices - 1:
            self.ind = self.ind + 1
        elif event.button == 'up' and self.ind > 0:
            self.ind = self.ind - 1
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

# set up display, volume, and scroller
fig = plt.figure(figsize=(15,7))
ax1 = plt.subplot2grid((1,4), (0,0),)
ax2 = plt.subplot2grid((1,4), (0,1),)
ax3 = plt.subplot2grid((1,4), (0,2),)



raw = read_png_volume(volume_path)


preproc = []
for i in range(len(os.listdir(preproc_path))):
    filename = "{}.png".format(i)
    if not os.path.isdir(os.path.join(preproc_path, filename)):
        im = np.array(Image.open(os.path.join(preproc_path, filename)))
        preproc.append(im.tolist())
preproc = np.array(preproc)

pred = []
for i in range(len(os.listdir(pred_path))):
    filename = "{}.png".format(i)
    print(filename)
    if not os.path.isdir(os.path.join(pred_path, filename)):
        im = np.array(Image.open(os.path.join(pred_path, filename)))
        pred.append(im.tolist())
pred = np.array(pred)

tracker = IndexTracker(ax1, raw, volume_path)

pre_tracker = IndexTracker(ax2, preproc, volume_path)

pred_tracker = IndexTracker(ax3, pred, volume_path)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('scroll_event', pre_tracker.onscroll)
fig.canvas.mpl_connect('scroll_event', pred_tracker.onscroll)


pre_ax = plt.axes([0.74, 0.75, 0.15, 0.1])
p_but = Button(pre_ax, 'PRE-PROCESS\nVOLUME', color='white', hovercolor='yellow')

rem_ax = plt.axes([0.74, 0.55, 0.15, 0.1])
r_but = Button(rem_ax, 'REMOVE\nSLICE', color='gray', hovercolor='gray')

spl_ax = plt.axes([0.74, 0.35, 0.15, 0.1])
s_but = Button(spl_ax, 'SPLIT\nINTO PARTS', color='gray', hovercolor='gray')

run_ax = plt.axes([0.74, 0.15, 0.15, 0.1])
a_but = Button(run_ax, 'RUN\nALGO', color='gray', hovercolor='gray')

pre = False
# to preprocess the slices
def on_pre_click(event):
    global pre
    if not pre: # if not pre-processed
        print('Running pre-processing...')
        # preprocess X
        # TODO: ...
        fig.set_title('scrolling through {}\nPRE-PROCESSED'.format(tracker.n, ' '.join(str(x) for x in rem_lst)))
        r_but.color = 'white'
        r_but.hovercolor = 'red'
        s_but.color = 'white'
        s_but.hovercolor = 'orange'
        p_but.color = 'gray'
        p_but.hovercolor = 'gray'
        tracker.im.axes.figure.canvas.draw()
        pre = True # set pre-processed to True
        print('DONE pre-processing!')
    else:
        print('Can only pre-process once.')

spl = False

# to remove faulty slices
def on_rem_click(event):
    global pre
    global spl
    if pre and not spl:
        if tracker.ind not in rem_lst:
            print('Removing slice {}'.format(tracker.ind))
            rem_lst.append(tracker.ind)
            ax.set_title('scrolling through {}\nPRE-PROCESSED\ndeleted slices: [{}]'.format(tracker.n, ' '.join(str(x) for x in rem_lst)))
            tracker.im.axes.figure.canvas.draw()
        else:
            print('Slice {} has already been removed.'.format(tracker.ind))
    elif not pre:
        print('Please pre-process before removing any slices.')
    else:
        print('Cannot remove slices after splitting into parts.')

# to split the body into each of the parts
def on_spl_click(event):
    global spl
    global pre
    if pre and not spl: # if pre-processed and not split into body parts
        print('Splitting into parts...')
        global X
        # remove faulty slices from X
        X = np.delete(X, rem_lst)
        # split into body parts and display them
        # TODO: ...
        ax.set_title('scrolling through {}\nPRE-PROCESSED, SPLIT INTO PARTS\ndeleted slices: [{}]'.format(tracker.n, ' '.join(str(x) for x in rem_lst)))
        r_but.color = 'gray'
        r_but.hovercolor = 'gray'
        s_but.color = 'gray'
        s_but.hovercolor = 'gray'
        a_but.color = 'white'
        a_but.hovercolor = 'green'
        tracker.im.axes.figure.canvas.draw()
        spl = True
        print('DONE splitting into parts!')
    elif not pre:
        print('Please pre-process before splitting into parts.')
    else:
        print('Can only split once.')

# to run algorithm and save stats
ran = False
def on_run_click(event):
    global pre
    global spl
    global ran
    if pre and spl and not ran: # if pre-processed and split into body parts
        print('Running algorithm... please do not click this button again.')
        global X
        # run algo
        # TODO: ...

        # save results
        # TODO: ...
        ax.set_title('scrolling through {}\ndeleted slices: [{}]\nPRE-PROCESSED, SPLIT INTO PARTS, RAN ALGORITHM'.format(tracker.n, ' '.join(str(x) for x in rem_lst)))
        a_but.color = 'gray'
        a_but.hovercolor = 'gray'
        tracker.im.axes.figure.canvas.draw()
        print('DONE running the algorithm! Results saved in {}.'.format('FILENAME.txt'))
        ran = True
    elif not ran:
        print('Please pre-process and split into parts before running the algorithm.')
    else:
        print('Already ran the algorithms and saved the results.')
        

p_but.on_clicked(on_pre_click)
pre_ax._button = p_but

r_but.on_clicked(on_rem_click)
rem_ax._button = r_but

s_but.on_clicked(on_spl_click)
spl_ax._button = s_but

a_but.on_clicked(on_run_click)
run_ax._button = a_but

plt.show()

