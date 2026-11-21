from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def npy_slices_to_nii(input_dir: str | Path, output_nii_gz: str | Path, per_id: str, refer_affine: np.ndarray) -> None:
    """
    Read all .npy slices in input_dir, stack to 3D, and save as .nii.gz.

    Args:
        input_dir: Folder containing .npy slice files.
        output_nii_gz: Output .nii.gz file path.
    """
    input_dir = Path(input_dir)
    output_nii_gz = Path(output_nii_gz)

    density_files = sorted(input_dir.glob("*density_recon*.npy"))
    if not density_files:
        raise ValueError(f"No .npy files found in: {input_dir}")

    speed_files = sorted(input_dir.glob("*speed_recon*.npy"))
    if not speed_files:
        raise ValueError(f"No .npy files found in: {input_dir}")

    density_slices = [np.load(f) for f in density_files]
    speed_slices = [np.load(f) for f in speed_files]
    density_volume = np.stack(density_slices, axis=0)
    speed_volume = np.stack(speed_slices, axis=0)

    density_nii = nib.Nifti1Image(density_volume, affine=refer_affine)
    speed_nii = nib.Nifti1Image(speed_volume, affine=refer_affine)
    
    nib.save(density_nii, str(output_nii_gz / f"{per_id}_density.nii.gz"))
    nib.save(speed_nii, str(output_nii_gz / f"{per_id}_speed.nii.gz"))


if __name__ == "__main__":
    root = Path(r'/home/dell/lhc/generate_density/latent_diffusion/test_outputs/vae_test/np')
    target = Path(r'/home/dell/lhc/generate_density/latent_diffusion/test_outputs/vae_test/vae_nii')
    refer_nii_path = r"/home/dell/lhc/generate_density/new_dataset/test/ct/per_125_ct.nii.gz"
    refer_nii = nib.load(refer_nii_path)
    refer_affine = refer_nii.affine

    for per in root.iterdir():
        if int(per.name[4:]) < 10:
            print(f'start to convert {per.name}')
            npy_slices_to_nii(str(per), str(target), per.name, refer_affine)
            print(f'end to convert {per.name}')