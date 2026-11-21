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
        
        print(f"Image dimensions: {dimensions}")
        print(f"Voxel spacing (mm): {voxel_size}")

        return dimensions, voxel_size

        
    except Exception as e:
        print(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    
    # per_file = r'H:\all_dataset\dataset_resampled_0.5mm\val\ct\per_001_ct.nii.gz'
    # print_nii_info(per_file)

    root_path = r'/home/dell/lhc/generate_density/new_dataset/test/ct'
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        print_nii_info(file_path)