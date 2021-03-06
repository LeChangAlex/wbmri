# Wb-MRI Cancer Detection Pipeline

## Preprocessing

### 1. Data formatting 
  Based on the data that was shared, each directory seemed to contain a few volumes. 
  The folder with stitched wbMRI DICOM's should be moved into a separate directory.
  Call `preprocess/read_dicoms.py PATH_TO_DICOM_ROOT png_volumes`.
  
  PATH_TO_DICOM_ROOT should be structured like this:
  
  
    - PATH_TO_DICOM_ROOT/volume_1/1.dcm
    - PATH_TO_DICOM_ROOT/volume_1/2.dcm
    - ...

    - PATH_TO_DICOM_ROOT/volume_2/1.dcm
    - PATH_TO_DICOM_ROOT/volume_2/2.dcm
    - ...


  The wbMRI images are stored as png files, with one directory per volume.
  e.g. 
  
    - png_volumes/volume_1/1.png
    - png_volumes/volume_1/2.png
    - ...

    - png_volumes/volume_2/1.png
    - png_volumes/volume_2/2.png
    - ...


 
 
### 2. Image Intensity Corrections (Histogram equalization, N4 Bias Correction, Noise removal)

  Call preprocess/preprocess.py with argument the directory containing volume directories

  e.g. 
  ```
  python preprocess/preprocess.py ./volumes
  ```
  This creates a directory called "preproc_volumes" which follows the structure of "volumes".
  The file "wbmri_metadata.csv" is also created, which contains various DICOM header metafeatures for each volume.
  
  
  
### 3. Chest Template Matching
  Call `preprocess/find_template.py`
  
  This script locates the coordinates of the chest crop from each volume and stores them in chest_coords.csv.
  It also produces a new directory which shows the located chest of each volume in a plot per image. 
  Each file should contain 3 bounding boxes; the rightmost contains the chest. 
  Delete plots with improperly bounded chests.

  

## Testing

  Call `run_algorithm.py` with 3 arguments: the directory of the volume, the model checkpoint for the chest, and the checkpoint for the legs.
  
  e.g. 
  ```
  python run_algorithm.py --vol_dir volumes/volume_1 --chest_ckpt vae_models/condyzswfixedy_180.tar --legs_ckpt vae_models/legs_180.tar
  ```
  The tool allows you to scroll through each volume on the left and provides the corresponding anomaly mask on the right.
