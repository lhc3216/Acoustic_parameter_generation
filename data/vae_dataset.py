import os
import glob
import numpy as np
import torch
from monai.data import PersistentDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, MapTransform, EnsureTyped
)

TARGET_SHAPE_3D = (512, 512, 256)

class ExtractSpecificVAE2DSliced(MapTransform):
    def __init__(self):
        super().__init__(keys=["density", "speed"])

    def __call__(self, data):
        d = dict(data)
        z = d["slice_idx"]  
        density_slice = d["density"][:, :, :, z]
        speed_slice = d["speed"][:, :, :, z]
        d["target"] = torch.cat([speed_slice, density_slice], dim=0)
        del d["density"], d["speed"]
        return d

def get_vae_data_dicts_all_slices(root_dir, split="train"):
    split_dir = os.path.join(root_dir, split)
    density_files = sorted(glob.glob(os.path.join(split_dir, "density", "*.nii.gz")))
    all_slice_dicts = []
    for density_path in density_files:
        patient_id = os.path.basename(density_path)[:7]
        speed_path = os.path.join(split_dir, "speed", f"{patient_id}_speed.nii.gz")
        if os.path.exists(speed_path):
            for z_idx in range(1, TARGET_SHAPE_3D[2] - 1): 
                all_slice_dicts.append({
                    "density": density_path,
                    "speed": speed_path,
                    "slice_idx": z_idx  
                })
    return all_slice_dicts

# ==========================================
# ==========================================
def get_static_cache_transforms():
    return Compose([
        LoadImaged(keys=["density", "speed"]),
        EnsureChannelFirstd(keys=["density", "speed"]),
        ScaleIntensityRanged(keys=["density"], a_min=1000.0, a_max=1950.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["speed"], a_min=1482.3, a_max=3100.0, b_min=-1.0, b_max=1.0, clip=True),
        ExtractSpecificVAE2DSliced(),
        EnsureTyped(keys=["target"])
    ])

# ==========================================
# ==========================================
def get_dynamic_aug_transforms(split_name='train'):
    if split_name == 'train':
        return Compose([
            RandFlipd(keys=["target"], spatial_axis=0, prob=0.5), 
            RandFlipd(keys=["target"], spatial_axis=1, prob=0.5), 
            RandRotate90d(keys=["target"], prob=0.5, spatial_axes=(0, 1))
        ])
    else:
        return Compose([])

if __name__ == "__main__":
    DATASET_ROOT = "/home/dell/lhc/generate_density/dataset_resampled_0.5mm"
    CACHE_DIR = os.path.join(DATASET_ROOT, "vae_persistent_cache_512")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    train_dicts = get_vae_data_dicts_all_slices(DATASET_ROOT, split="train")
    
    static_transforms = get_static_cache_transforms()
    train_dataset = PersistentDataset(
        data=train_dicts, 
        transform=static_transforms, 
        cache_dir=CACHE_DIR
    )
    
    dynamic_aug_transforms = get_dynamic_aug_transforms(split_name='train')
    
    # ==========================================
    # ==========================================
    def dynamic_collate_fn(batch):
        augmented_batch = [dynamic_aug_transforms(item) for item in batch]
        return list_data_collate(augmented_batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,        
        shuffle=True, 
        num_workers=8,      
        pin_memory=True,
        collate_fn=dynamic_collate_fn
    )
    
    for batch in train_loader:
        print("\n=== VAE cached loading and augmentation check passed ===")
        print(f"Target shape: {batch['target'].shape}") 
        break