import os
import glob
import numpy as np
import torch
from data.sd_dataset import ExtractAcousticSlice
from pathlib import Path
from tqdm import tqdm
from monai.data import PersistentDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, MapTransform, EnsureTyped, ScaleIntensityRangePercentilesd
)

TARGET_SHAPE_3D = (512, 512, 256)

def get_sd_data_dicts_all_slices(root_dir, split="train"):
    split_dir = os.path.join(root_dir, split)
    density_files = sorted(glob.glob(os.path.join(split_dir, "density", "*.nii.gz")))
    all_slice_dicts = []
    for density_path in density_files:
        patient_id = os.path.basename(density_path)[:7]
        speed_path = os.path.join(split_dir, "speed", f"{patient_id}_speed.nii.gz")
        ct_path = os.path.join(split_dir, "ct", f"{patient_id}_ct.nii.gz")
        mr_path = os.path.join(split_dir, "mr", f"{patient_id}_mr.nii.gz")
        if os.path.exists(speed_path):
            for z_idx in range(1, TARGET_SHAPE_3D[2] - 1): 
                all_slice_dicts.append({
                    "ct": ct_path,
                    "mr": mr_path,
                    "density": density_path,
                    "speed": speed_path,
                    "patient_id": patient_id,
                    "slice_idx": z_idx  
                })
    return all_slice_dicts

def get_static_cache_transforms():
    return Compose([
        LoadImaged(keys=["ct", "mr", "density", "speed"]),
        EnsureChannelFirstd(keys=["ct", "mr", "density", "speed"]),
        ScaleIntensityRanged(keys=["ct"], a_min=-1024.0, a_max=2000.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRangePercentilesd(keys=["mr"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True, relative=False),
        ScaleIntensityRanged(keys=["density"], a_min=1000.0, a_max=1950.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["speed"], a_min=1482.3, a_max=3100.0, b_min=-1.0, b_max=1.0, clip=True),
        ExtractAcousticSlice(),
        EnsureTyped(keys=["condition", "target"])
    ])


def build_dataloader(root_dir, split='val', cache_dir=None, batch_size=8, num_workers=8):
    if split == "train":
        raise ValueError("Only val/test split is supported for now")
    print(f"load {split} dataset...")
    all_patient_dicts = get_sd_data_dicts_all_slices(root_dir, split=split)
    static_transforms = get_static_cache_transforms()
    dataset_cached = PersistentDataset(
        data=all_patient_dicts, 
        transform=static_transforms, 
        cache_dir=os.path.join(cache_dir, "single_slice_v1") if cache_dir is not None else None,
    )
    val_loader = DataLoader(
        dataset_cached, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate 
    )
    return val_loader
