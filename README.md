# fus_gen

Research code for generating **sound-speed and density maps from CT/MRI**, using an acoustic VAE and conditional latent diffusion. The repository also includes distillation code and a k-wave-sim module for focused ultrasound simulation.

## Structure

```text
configs/    Training and evaluation configurations
models/     VAE, diffusion, and distillation models
data/       NIfTI loading and preprocessing
train/      Training entry points
test/       Model evaluation entry points
tools/      Metrics, visualization, and data utilities
k-wave-sim/ MATLAB acoustic simulation
```

## Installation

Use Python 3.10 or newer and install a recent [PyTorch](https://pytorch.org/get-started/locally/) build matching your CUDA environment. A CUDA-capable GPU is recommended for training.

```bash
python -m pip install monai diffusers transformers accelerate safetensors hydra-core omegaconf numpy nibabel matplotlib scikit-image Pillow tqdm tensorboard pytorch-msssim
```

Install `SimpleITK` additionally if using `tools/scripts/resample.py`. Dependency versions are not yet pinned.

## Data

Prepare registered NIfTI volumes with separate patient-level `train`, `val`, and `test` splits. Each split uses the same layout:

```text
dataset/train/
├── ct/sub0001_ct.nii.gz
├── mr/sub0001_mr.nii.gz
├── speed/sub0001_speed.nii.gz
└── density/sub0001_density.nii.gz
```
Expected volume size is `512 × 512 × 256`, prepared at `0.5 mm` isotropic spacing. Supply sound speed in **m/s** and density in **kg/m³**. Training and evaluation extract the same single-slice conditions and retain `slice_idx` for z-position encoding.

## Training and evaluation

Run commands from the repository root. Update the paths in `configs/` before starting:

- `data_root`: prepared dataset directory.
- `model.pretrained_model_name_or_path` (SD): Stable Diffusion v1.5 directory containing `unet/`, its configuration, and weights.
- `trainer.vae_ckpt` (SD): trained acoustic VAE checkpoint.

Train the VAE, then set the VAE checkpoint path before training diffusion:

```bash
python train/train_vae.py task=basic_loss+grad+imp trainer.resume_from=null hydra.job.chdir=false
python train/train_sd.py task=acoustic trainer.save_dir=checkpoints/sd trainer.resume_from=null trainer.val_start_step=0 hydra.job.chdir=false
```

Evaluate the corresponding checkpoints:

```bash
python test/test_vae.py task=basic_loss+grad+imp tester.test_pt=vae_best.pth hydra.job.chdir=false
python test/test_sd.py task=acoustic trainer.save_dir=checkpoints/sd tester.test_pt=sd_latest.pth tester.test_save_dir=test_outputs/sd tester.now=paired hydra.job.chdir=false
```

Checkpoints are saved under `checkpoints/`; evaluation writes reconstruction metrics and results to the configured output directory. Hydra accepts command-line overrides such as `trainer.batch_size=16`. On Windows, use `num_workers=0` initially. 


## Acoustic simulation

Requires MATLAB with Image Processing Toolbox and the separately installed [k-Wave toolbox](https://www.k-wave.org/) (version 1.4 API).

Run from the repository root in MATLAB, replacing the example paths:

```matlab
addpath(fullfile(pwd, 'k-wave-sim'));
opts = struct('f0', 550e3);
out = run_acoustic_simulation('/path/to/speed.nii.gz', ...
    '/path/to/density.nii.gz', '/path/to/k-Wave', ...
    fullfile(pwd, 'simulation_outputs', 'case001'), [], opts);
```

`run_acoustic_simulation.m` reads and validates the volumes; `simulate_focused_ultrasound.m` builds the medium and focused bowl, runs the solver, and saves `acoustic_field.mat` and `pressure_slice.png`. Replace `[]` with a one-based `[i j k]` target index in the input volume to specify a focus. Additional parameters are defined in `defaultSimulationOptions` inside the core function.
