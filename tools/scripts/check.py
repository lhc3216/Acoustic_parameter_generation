import os
from pathlib import Path

def verify_dataset(root_dir):
    root = Path(root_dir)
    splits = ['train', 'val', 'test']
    modalities = ['ct', 'mr', 'density', 'speed']
    
    split_complete_patients = {split: set() for split in splits}
    
    error_logs = []
    
    print("==================================================")
    print("Starting dataset integrity check...")
    print("==================================================\n")

    # ----------------------------------------------------
    # ----------------------------------------------------
    for split in splits:
        print(f"Scanning [{split}] split...")
        split_dir = root / split
        
        mod_pids = {mod: set() for mod in modalities}
        
        for mod in modalities:
            mod_dir = split_dir / mod
            if mod_dir.exists():
                for file_path in mod_dir.glob("*.nii.gz"):
                    patient_id = file_path.name[:7]
                    mod_pids[mod].add(patient_id)
            else:
                error_logs.append(f"Error: directory not found: {mod_dir}")
                
        all_pids_in_split = set().union(*mod_pids.values())
        
        for pid in all_pids_in_split:
            missing_mods = [mod for mod in modalities if pid not in mod_pids[mod]]
            
            if missing_mods:
                error_logs.append(f"Warning: [{split}] patient {pid} is missing: {', '.join(missing_mods)}")
            else:
                split_complete_patients[split].add(pid)
                
        print(f"   -> Found {len(all_pids_in_split)} patients; {len(split_complete_patients[split])} have all modalities.")

    # ----------------------------------------------------
    # ----------------------------------------------------
    print("\nChecking for patient overlap across splits (data leakage)...")
    
    train_pids = split_complete_patients['train']
    val_pids = split_complete_patients['val']
    test_pids = split_complete_patients['test']
    
    leak_train_val = train_pids & val_pids
    leak_train_test = train_pids & test_pids
    leak_val_test = val_pids & test_pids
    
    is_leaking = False
    if leak_train_val:
        error_logs.append(f"Data leakage: patients shared by train and val: {sorted(list(leak_train_val))}")
        is_leaking = True
    if leak_train_test:
        error_logs.append(f"Data leakage: patients shared by train and test: {sorted(list(leak_train_test))}")
        is_leaking = True
    if leak_val_test:
        error_logs.append(f"Data leakage: patients shared by val and test: {sorted(list(leak_val_test))}")
        is_leaking = True
        
    if not is_leaking:
        print("   -> No patient overlap across splits.")

    # ----------------------------------------------------
    # ----------------------------------------------------
    print("\n==================================================")
    print("Dataset summary (patients with all modalities)")
    print("==================================================")
    
    total_valid = len(train_pids) + len(val_pids) + len(test_pids)
    print(f"Total valid patients: {total_valid}")
    print(f"  ├─ Train: {len(train_pids)}")
    print(f"  ├─ Val: {len(val_pids)}")
    print(f"  └─ Test: {len(test_pids)}")
    
    if error_logs:
        print("\nWarning: the following issues require review:")
        for log in error_logs:
            print("  " + log)
    else:
        print("\nDataset checks passed: all modalities are paired and no patient overlap was found.")

if __name__ == "__main__":
    DATASET_ROOT = "/home/dell/lhc/generate_density/dataset"
    verify_dataset(DATASET_ROOT)