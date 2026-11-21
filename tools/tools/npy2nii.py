from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from tqdm import tqdm


def npy_slices_to_nii(
    input_dir: Union[str, Path],
    output_nii_gz: Union[str, Path],
    per_id: str,
    refer_affine: np.ndarray,
) -> None:
    """
    Read all .npy slices in input_dir, stack to 3D, and save as .nii.gz.

    Args:
        input_dir: Folder containing .npy slice files.
        output_nii_gz: Output .nii.gz file path.
    """
    input_dir = Path(input_dir)
    output_nii_gz = Path(output_nii_gz)

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise ValueError(f"No .npy files found in: {input_dir}")

    slices = [np.load(f) for f in files]
    volume = np.stack(slices, axis=2)
    first_slice = volume[..., :1]
    last_slice = volume[..., -1:]

    volume = np.concatenate([first_slice, volume, last_slice], axis=2)
    nii = nib.Nifti1Image(volume, affine=refer_affine)
    nib.save(nii, str(output_nii_gz))


if __name__ == "__main__":
    root = Path(r'/home/dell/lhc/generate_density/latent_diffusion/sd_z_compare/both/class_np')
    target = Path(r'/home/dell/lhc/generate_density/latent_diffusion/sd_z_compare/both/class_nii')
    refer_nii_path = r'/home/dell/lhc/generate_density/dataset_resampled_0.5mm/val/ct/per_001_ct.nii.gz'
    refer_nii = nib.load(refer_nii_path)
    refer_affine = refer_nii.affine

    modality_list = ['density_orig', 'density_recon', 'speed_orig', 'speed_recon']

    for modality in tqdm(modality_list, desc='Converting modalities'):
        input_dir = root / modality
        output_dir = target / modality
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        for per in input_dir.iterdir():
            npy_slices_to_nii(str(per), str(output_dir / f'{per.name}_{modality}.nii.gz'), per.name, refer_affine)