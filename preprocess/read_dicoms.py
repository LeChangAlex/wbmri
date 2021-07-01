import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
import sys

from skimage import exposure
import pandas as pd

def preprocess(vol, w_list):

    p2, p98 = np.percentile(vol, (2, 98))
    for i in range(len(w_list)):
        # s = vol[i]
        w_center, w_width = w_list[i]

        vol[i] = exposure.rescale_intensity(vol[i], in_range=(w_center - w_width / 2, w_center + w_width / 2), out_range=(0, 255))   
    # vol = exposure.rescale_intensity(vol, in_range=(p2, p98), out_range=(0, 255))   
    # print(vol.sum())

    # print(np.amin(vol), np.amax(vol))

    # vol = vol.astype(np.float16) / np.amax(vol) * 255

    # vol = exposure.equalize_adapthist(vol, clip_limit=0.03)

    # Equalization
    # vol = exposure.equalize_hist(vol)
    
    return vol

def read_dcm(dir):
    vol = None
    w_list = []
    for fn in sorted(os.listdir(dir)):
        # try:
        # print(dir, fn)

        dcm = pydicom.read_file(os.path.join(dir, fn))
        dcm_vol = dcm.pixel_array

        
        # print(dcm.shape)
        if vol is None:
            vol = dcm_vol[np.newaxis, ...]


            # sex = dcm.PatientSex
            # age = dcm.PatientAge
            # weight = dcm.PatientWeight
            # st = dcm.SliceThickness
            # ss = dcm.SpacingBetweenSlices
        
            # ps = dcm.PixelSpacing
        
            metafeatures = {
                "sex": dcm.PatientSex,
                "age": dcm.PatientAge,
                "weight": dcm.PatientWeight,    
                "slicethickness": dcm.SliceThickness,
                "slicespacing": dcm.SpacingBetweenSlices,
                "pixelspacing1": dcm.PixelSpacing,
                "fn": dir
            }
        else:
            # print(vol.shape)
            vol = np.concatenate((vol, dcm_vol[np.newaxis, ...]), 0)
        # except:
            # print(dir)
        w_list.append((dcm.WindowCenter, dcm.WindowWidth))
    print(dcm)


    return vol, metafeatures, w_list
            


df_cols = [
    "volume_n",
    "sex",
    "age",
    "weight",    
    "slicethickness",
    "slicespacing",
    "pixelspacing1",
    "fn"

]

df = pd.DataFrame(columns=df_cols)

dicom_root = sys.argv[1]
png_root = sys.argv[2]
os.makedirs(png_root, exist_ok=True)

dicom_dirs = sorted(os.listdir(dicom_root))

for i in range(len(dicom_dirs)):

    vol, metafeatures, w_list = read_dcm(os.path.join(dicom_root, dicom_dirs[i]))
    
    vol = preprocess(vol, w_list)


    os.makedirs(os.path.join(png_root, "volume_{}".format(i)), exist_ok=True)
    vol -= np.amin(vol)
    vol = vol / np.amax(vol)

    for j in range(vol.shape[0]):
        skimage.io.imsave(os.path.join(png_root, "volume_{}/{}.png".format(i, j)), (vol[j] * 255).astype(np.uint8))

    metafeatures["volume_n"] = i
    df = df.append(metafeatures, ignore_index=True)

print(df)
df.to_csv("metadata.csv")