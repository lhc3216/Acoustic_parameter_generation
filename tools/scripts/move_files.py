import os
import shutil
import numpy as np
from pathlib import Path
import nibabel as nib

# ==========================================
# ==========================================
def get_info(file_path):
    try:
        nii_img = nib.load(file_path)
        dimensions = nii_img.header.get_data_shape()
        voxel_size = nii_img.header.get_zooms()
        return dimensions, voxel_size
    except Exception as e:
        print(f"Error reading file: {file_path} - {str(e)}")
        return None, None

def is_patient_standard(patient_id, split, root, ref_dim, ref_vox):
    modalities = ['ct', 'mr', 'density', 'speed']
    for mod in modalities:
        file_path = root / split / mod / f"{patient_id}_{mod}.nii.gz"
        
        if not file_path.exists():
            print(f"Warning: patient {patient_id} is missing {mod} and does not meet the reference criteria.")
            return False
            
        dim, vox = get_info(file_path)
        
        if dim is None or vox is None:
            return False
            
        if dim != ref_dim:
            return False
            
        if not np.allclose(vox, ref_vox, atol=1e-3):
            return False
            
    return True

# ==========================================
# ==========================================
def get_patients_in_split(root, split):
    ct_dir = root / split / 'ct'
    patients = []
    if ct_dir.exists():
        for f in ct_dir.glob("*.nii.gz"):
            patients.append(f.name[:7])
    return sorted(patients)

def move_patient_data(patient_id, src_split, dst_split, root):
    modalities = ['ct', 'mr', 'density', 'speed']
    for mod in modalities:
        src_file = root / src_split / mod / f"{patient_id}_{mod}.nii.gz"
        dst_file = root / dst_split / mod / f"{patient_id}_{mod}.nii.gz"
        
        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
        else:
            print(f"Warning: source file not found for move: {src_file}")

# ==========================================
# ==========================================
if __name__ == "__main__":
    root = Path("/home/dell/lhc/generate_density/dataset")
    
    refer_file = root / 'val' / 'ct' / 'per_001_ct.nii.gz'
    refer_dimensions, refer_voxel_size = get_info(refer_file)
    print(f"Reference dimensions: {refer_dimensions}, voxel spacing: {refer_voxel_size}\n")

    val_num = 0
    test_num = 0
    
    for pid in get_patients_in_split(root, 'val'):
        if not is_patient_standard(pid, 'val', root, refer_dimensions, refer_voxel_size):
            move_patient_data(pid, 'val', 'train', root)
            val_num += 1
            print(f"Moved validation patient {pid} to train because its modalities do not meet the reference criteria.")
            
    for pid in get_patients_in_split(root, 'test'):
        if not is_patient_standard(pid, 'test', root, refer_dimensions, refer_voxel_size):
            move_patient_data(pid, 'test', 'train', root)
            test_num += 1
            print(f"Moved test patient {pid} to train because its modalities do not meet the reference criteria.")
            
    print(f"\nMoved {val_num} patients out of val and {test_num} out of test.\n")

    if val_num > 0 or test_num > 0:
        train_patients = get_patients_in_split(root, 'train')
        
        val_replenished = 0
        test_replenished = 0
        
        for pid in train_patients:
            if is_patient_standard(pid, 'train', root, refer_dimensions, refer_voxel_size):
                
                if val_replenished < val_num:
                    move_patient_data(pid, 'train', 'val', root)
                    val_replenished += 1
                    print(f"Moved eligible training patient {pid} to val ({val_replenished}/{val_num}).")
                    continue
                
                if test_replenished < test_num:
                    move_patient_data(pid, 'train', 'test', root)
                    test_replenished += 1
                    print(f"Moved eligible training patient {pid} to test ({test_replenished}/{test_num}).")
                    
                if val_replenished == val_num and test_replenished == test_num:
                    break
        
        if val_replenished < val_num or test_replenished < test_num:
            print("\nWarning: not enough eligible training patients to replenish val and test.")
            print(f"Val patients needed: {val_num}; replenished: {val_replenished}.")
            print(f"Test patients needed: {test_num}; replenished: {test_replenished}.")
        else:
            print("\nDataset redistribution complete. All validation and test modalities meet the reference criteria.")
    else:
        print("Scan complete: all val and test modalities meet the reference criteria; no moves needed.")