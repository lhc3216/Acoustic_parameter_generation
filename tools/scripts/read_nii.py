import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm



def print_nii_info(file_path):
    try:
        nii_img = nib.load(file_path)
        
        data = nii_img.get_fdata()
        
        dimensions = data.shape
        
        voxel_size = nii_img.header.get_zooms()
        
        affine = nii_img.affine
        
        print(f"Value range: [{np.min(data):.2f}, {np.max(data):.2f}]")

        return dimensions, voxel_size

        
    except Exception as e:
        print(f"Error reading file: {str(e)}")


if __name__ == "__main__":

    root = r"/home/dell/lhc/generate_density/dataset_resampled_0.5mm/val/ct"
    for file in os.listdir(root):
        file_name = os.path.join(root, file)
        d, v = print_nii_info(file_name)
        print(d, v)