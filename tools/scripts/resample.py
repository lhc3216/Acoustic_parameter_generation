import os
import SimpleITK as sitk
import concurrent.futures
from tqdm import tqdm
import multiprocessing

def resample_image(sitk_image, target_spacing=(0.5, 0.5, 0.5), is_label=False, pad_value=0.0):
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetDefaultPixelValue(float(pad_value))
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) 
    else:
        resampler.SetInterpolator(sitk.sitkLinear) 
        
    return resampler.Execute(sitk_image)

def pad_or_crop_to_target(sitk_image, target_size=(512, 512, 256), pad_value=0.0):
    current_size = sitk_image.GetSize()
    
    if current_size == target_size:
        return sitk_image
        
    pad_lower, pad_upper = [], []
    crop_lower, crop_upper = [], []
    
    for i in range(3):
        diff = target_size[i] - current_size[i]
        if diff > 0:
            lower = diff // 2
            upper = diff - lower
            pad_lower.append(lower)
            pad_upper.append(upper)
            crop_lower.append(0)
            crop_upper.append(0)
        else:
            pad_lower.append(0)
            pad_upper.append(0)
            lower = (-diff) // 2
            upper = (-diff) - lower
            crop_lower.append(lower)
            crop_upper.append(upper)
            
    if sum(pad_lower) + sum(pad_upper) > 0:
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(pad_lower)
        pad_filter.SetPadUpperBound(pad_upper)
        pad_filter.SetConstant(float(pad_value))
        sitk_image = pad_filter.Execute(sitk_image)
        
    if sum(crop_lower) + sum(crop_upper) > 0:
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize(crop_lower)
        crop_filter.SetUpperBoundaryCropSize(crop_upper)
        sitk_image = crop_filter.Execute(sitk_image)
        
    return sitk_image

def process_single_file(args):
    src_path, dst_path, mod, target_spacing, target_size, pad_val = args
    try:
        img = sitk.ReadImage(src_path)
        is_label = True if mod in ['density', 'speed'] else False
        
        resampled_img = resample_image(img, target_spacing, is_label, pad_value=pad_val)
        
        final_img = pad_or_crop_to_target(resampled_img, target_size, pad_value=pad_val)
        
        sitk.WriteImage(final_img, dst_path)
        return True, f"Success: {os.path.basename(src_path)}"
    except Exception as e:
        return False, f"Error processing {src_path}: {str(e)}"

def process_dataset_multiprocess(src_root, dst_root, target_spacing=(0.5, 0.5, 0.5), target_size=(512, 512, 256), max_workers=None):
    splits = ['train', 'val', 'test']
    modalities = ['ct', 'mr', 'density', 'speed'] 
    
    tasks = []
    for split in splits:
        for mod in modalities:
            src_dir = os.path.join(src_root, split, mod)
            dst_dir = os.path.join(dst_root, split, mod)
            
            if not os.path.exists(src_dir):
                continue
            
            os.makedirs(dst_dir, exist_ok=True)
                
            if mod == 'ct':
                pad_val = -1024.0
            elif mod in ['mr', 'mri']:
                pad_val = 0.0
            elif mod == 'density':
                pad_val = 1000.0  
            elif mod == 'speed':
                pad_val = 1482.3  
            else:
                pad_val = 0.0
                
            for filename in os.listdir(src_dir):
                if filename.endswith('.nii.gz'):
                    src_path = os.path.join(src_dir, filename)
                    dst_path = os.path.join(dst_dir, filename)
                    tasks.append((src_path, dst_path, mod, target_spacing, target_size, pad_val))

    total_tasks = len(tasks)
    print(f"Found {total_tasks} files to process.")
    if total_tasks == 0:
        return

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting parallel processing; worker count: {max_workers}")

    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                print(f"\n{msg}")

    print(f"\nDataset processing complete. Successful files: {success_count}/{total_tasks}")

if __name__ == "__main__":
    source_dataset = "/home/dell/lhc/generate_density/dataset"
    target_dataset = "/home/dell/lhc/generate_density/dataset_resampled_0.5mm"
    GLOBAL_TARGET_SIZE = (512, 512, 256)
    
    process_dataset_multiprocess(
        source_dataset, 
        target_dataset, 
        target_spacing=(0.5, 0.5, 0.5),
        target_size=GLOBAL_TARGET_SIZE,
        max_workers=None
    )