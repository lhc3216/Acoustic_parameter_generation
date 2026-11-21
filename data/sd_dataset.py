import os
import glob
import random
import numpy as np
import torch
from monai.data import PersistentDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, MapTransform, EnsureTyped, ScaleIntensityRangePercentilesd,
)

TARGET_SHAPE_3D = (512, 512, 256)
TRAIN_Z_MIN = 3
TRAIN_Z_MAX = 250

class ExtractAcousticSlice(MapTransform):
    """Extract one CT/MRI slice and its targets, preserving the z index."""

    def __init__(self):
        super().__init__(keys=["ct", "mr", "density", "speed"])

    def __call__(self, data):
        d = dict(data)
        z = int(d["slice_idx"])
        shape = d["ct"].shape
        for key in self.keys:
            if len(d[key].shape) != 4 or d[key].shape[0] != 1 or d[key].shape != shape:
                raise ValueError("CT, MRI, speed, and density must share shape [1, H, W, Z].")
        if not 0 <= z < shape[-1]:
            raise ValueError(f"slice_idx {z} is outside the volume depth {shape[-1]}.")
        ct_slice = d["ct"][..., z]
        mr_slice = d["mr"][..., z]
        density_slice = d["density"][..., z]
        speed_slice = d["speed"][..., z]
        d["condition"] = torch.cat([ct_slice, mr_slice], dim=0)
        d["target"] = torch.cat([speed_slice, density_slice], dim=0)
        del d["ct"], d["mr"], d["density"], d["speed"]
        return d

def get_sd_data_dicts_patients(root_dir, split="train"):
    split_dir = os.path.join(root_dir, split)
    density_files = sorted(glob.glob(os.path.join(split_dir, "density", "*.nii.gz")))
    all_patient_dicts = []
    for density_path in density_files:
        patient_id = os.path.basename(density_path)[:7]
        speed_path = os.path.join(split_dir, "speed", f"{patient_id}_speed.nii.gz")
        ct_path = os.path.join(split_dir, "ct", f"{patient_id}_ct.nii.gz")
        mr_path = os.path.join(split_dir, "mr", f"{patient_id}_mr.nii.gz")
        if os.path.exists(speed_path):
            all_patient_dicts.append({
                    "patient_id": patient_id,
                    "ct": ct_path,
                    "mr": mr_path,
                    "density": density_path,
                    "speed": speed_path,
                })
    return all_patient_dicts


def get_sd_data_dicts_all_slices(root_dir, split="train", z_min=TRAIN_Z_MIN, z_max=TRAIN_Z_MAX):
    patient_dicts = get_sd_data_dicts_patients(root_dir, split=split)
    all_slice_dicts = []
    for p in patient_dicts:
        for z_idx in range(z_min, z_max):
            all_slice_dicts.append(
                {
                    "ct": p["ct"],
                    "mr": p["mr"],
                    "density": p["density"],
                    "speed": p["speed"],
                    "patient_id": p["patient_id"],
                    "slice_idx": z_idx,
                }
            )
    return all_slice_dicts

# ==========================================
# ==========================================
def get_static_cache_transforms():
    return Compose([
        LoadImaged(keys=["ct", "mr", "density", "speed"]),
        EnsureChannelFirstd(keys=["ct", "mr", "density", "speed"]),
        ScaleIntensityRanged(keys=["ct"], a_min=-1024.0, a_max=2000.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRangePercentilesd(keys=["mr"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True, relative=False),
        ScaleIntensityRanged(keys=["density"], a_min=1000.0, a_max=1950.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["speed"], a_min=1482.3, a_max=3100.0, b_min=-1.0, b_max=1.0, clip=True),
        ExtractAcousticSlice(),
    ])

# ==========================================
# ==========================================
def get_dynamic_aug_transforms(split_name='train'):
    if split_name == 'train':
        return Compose([
            EnsureTyped(keys=["condition", "target"]),
            RandFlipd(keys=["condition", "target"], spatial_axis=0, prob=0.5), 
            RandFlipd(keys=["condition", "target"], spatial_axis=1, prob=0.5), 
            RandRotate90d(keys=["condition", "target"], prob=0.5, spatial_axes=(0, 1))
        ])
    else:
        return Compose([
            EnsureTyped(keys=["condition", "target"]),
        ])

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dynamic_collate_fn(dynamic_transforms):
    def _dynamic_collate_fn(batch):
        augmented_batch = [dynamic_transforms(item) for item in batch]
        return list_data_collate(augmented_batch)
    return _dynamic_collate_fn

def build_train_dataloader(
    root_dir,
    split='train',
    cache_dir=None,
    batch_size=8,
    num_workers=8,
    rank=0,
    world_size=1,
):
    if split != 'train':
        raise ValueError("Only train split is supported for now")
    print(f"Loading {split} dataset...")
    
    all_slice_dicts = get_sd_data_dicts_all_slices(root_dir, split=split, z_min=TRAIN_Z_MIN, z_max=TRAIN_Z_MAX)
    if world_size > 1:
        total_slices = len(all_slice_dicts)
        usable_slices = (total_slices // world_size) * world_size
        if usable_slices <= 0:
            raise RuntimeError(
                f"Not enough slices ({total_slices}) for world_size={world_size}."
            )
        all_slice_dicts = all_slice_dicts[:usable_slices]
        all_slice_dicts = all_slice_dicts[rank::world_size]
    if len(all_slice_dicts) == 0:
        raise RuntimeError(
            f"No training slices after sharding (rank={rank}, world_size={world_size}). "
            "Please check dataset size and world_size."
        )

    static_transforms = get_static_cache_transforms()
    dataset_cached = PersistentDataset(
        data=all_slice_dicts,
        transform=static_transforms, 
        cache_dir=os.path.join(cache_dir, "single_slice_v1") if cache_dir is not None else None,
    )

    dynamic_transforms = get_dynamic_aug_transforms(split_name=split)
    dynamic_collate_fn = build_dynamic_collate_fn(dynamic_transforms)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dynamic_collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=world_size > 1,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        dataset_cached,
        shuffle=True,
        **loader_kwargs,
    )
    
    return train_loader

