from pathlib import Path
import shutil
from tqdm import tqdm

root = Path(r'/home/dell/lhc/generate_density/latent_diffusion/sd_z_compare/both/np')
target = Path(r'/home/dell/lhc/generate_density/latent_diffusion/sd_z_compare/both/class_np')
list_name = ['density_orig',\
    'density_recon', \
        'speed_orig', \
            'speed_recon']

target_modality_list = [target/name for name in list_name]

for path in target_modality_list:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

for file in tqdm(root.iterdir()):
    per_id = file.name[:7]
    target_modality = file.stem[12:]
    class_path = target / target_modality / per_id
    if not class_path.exists():
        class_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, class_path / file.name)
