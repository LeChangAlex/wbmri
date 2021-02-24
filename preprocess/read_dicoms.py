import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
import sys

import pandas as pd


def read_dcm(dir):
    vol = None

    for fn in sorted(os.listdir(dir)):
        try:
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
                }
            else:
                # print(vol.shape)
                vol = np.concatenate((vol, dcm_vol[np.newaxis, ...]), 0)
        except:
            print(dir)



    return vol, metafeatures
            


df_cols = [
    "volume_n",
    "sex",
    "age",
    "weight",    
    "slicethickness",
    "slicespacing",
    "pixelspacing1",

]

df = pd.DataFrame(columns=df_cols)

dicom_root = sys.argv[1]
png_root = sys.argv[2]
os.makedirs(png_root, exist_ok=True)

dicom_dirs = sorted(os.listdir(dicom_root))

for i in range(len(dicom_dirs)):

    vol, metafeatures = read_dcm(os.path.join(dicom_root, dicom_dirs[i]))
    


    os.makedirs(os.path.join(png_root, "volume_{}".format(i)), exist_ok=True)
    vol -= np.amin(vol)
    vol = vol / np.amax(vol)

    for j in range(vol.shape[0]):
        skimage.io.imsave(os.path.join(png_root, "volume_{}/{}.png".format(i, j)), (vol[j] * 255).astype(np.uint8))

    metafeatures["volume_n"] = i
    df = df.append(metafeatures, ignore_index=True)

print(df)
df.to_csv("metadata.csv")