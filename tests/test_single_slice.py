"""CPU regression checks with synthetic volumes and a tiny mocked SD backbone.

Run with: python -m unittest discover -s tests -v
"""

import ast
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch import nn

from data.sd_dataset import ExtractAcousticSlice, build_train_dataloader
from data.test_sd_dataset import build_dataloader


ROOT = Path(__file__).resolve().parents[1]


class TinyUNet(nn.Module):
    """Exercise real conditioning and discriminator hooks without SD weights."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(cross_attention_dim=8, block_out_channels=[32, 32])
        self.conv_in = nn.Conv2d(4, 32, 3, padding=1)
        self.mid_block = nn.Conv2d(32, 32, 1)
        self.up_blocks = nn.ModuleList([nn.Conv2d(32, 32, 1)])
        self.conv_out = nn.Conv2d(32, 4, 1)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, sample, _timesteps, encoder_hidden_states):
        hidden = self.conv_in(sample)
        hidden = hidden + encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        hidden = self.up_blocks[0](self.mid_block(hidden))
        return SimpleNamespace(sample=self.conv_out(hidden))


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backbone_module = ModuleType('diffusers')
backbone_module.UNet2DConditionModel = TinyUNet
with patch.dict('sys.modules', {'diffusers': backbone_module}):
    from models.sd.sd import HybridAcousticDiffusion, apply_modality_dropout
    # Load the implementation directly to keep these checks independent of
    # scheduler dependencies imported by models.dmd2.__init__.
    dmd2 = load_source('_dmd2_single_slice_checks', 'models/dmd2/dmd2.py')


class Config(dict):
    __getattr__ = dict.__getitem__


class SingleSliceTests(unittest.TestCase):
    def test_exact_slice_channels_and_metadata(self):
        transform = ExtractAcousticSlice()
        volumes = {key: torch.arange(6).view(1, 1, 1, 6).expand(1, 32, 32, 6) + offset
                   for key, offset in [('ct', 10), ('mr', 20), ('speed', 30), ('density', 40)]}
        for z in (0, 3, 5):
            output = transform(dict(volumes, slice_idx=z, patient_id='sub0001'))
            self.assertEqual(tuple(output['condition'].shape), (2, 32, 32))
            self.assertEqual(tuple(output['target'].shape), (2, 32, 32))
            self.assertTrue(torch.all(output['condition'][0] == 10 + z))
            self.assertTrue(torch.all(output['condition'][1] == 20 + z))
            self.assertTrue(torch.all(output['target'][0] == 30 + z))
            self.assertTrue(torch.all(output['target'][1] == 40 + z))
            self.assertEqual(output['slice_idx'], z)
            self.assertEqual(output['patient_id'], 'sub0001')
        with self.assertRaises(ValueError):
            transform(dict(volumes, slice_idx=6))

    def test_train_and_eval_loaders_use_new_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.broadcast_to(np.arange(256, dtype=np.float32), (16, 16, 256)).copy()
            for split in ('train', 'val'):
                for key, offset in [('ct', 0), ('mr', 1), ('speed', 1500), ('density', 1100)]:
                    folder = root / split / key
                    folder.mkdir(parents=True)
                    nib.save(nib.Nifti1Image(values + offset, np.eye(4)),
                             str(folder / ('sub0001_' + key + '.nii.gz')))
            for split, builder in [('train', build_train_dataloader), ('val', build_dataloader)]:
                cache = root / ('cache_' + split)
                cache.mkdir()
                old_file = cache / 'legacy.pt'
                old_file.write_bytes(b'old three-channel cache')
                loader = builder(root, split=split, cache_dir=cache, batch_size=1, num_workers=0)
                sample = loader.dataset[0]
                self.assertEqual(tuple(sample['condition'].shape), (2, 16, 16))
                self.assertEqual(tuple(sample['target'].shape), (2, 16, 16))
                self.assertEqual(int(sample['slice_idx']), 3 if split == 'train' else 1)
                self.assertEqual(Path(loader.dataset.cache_dir), cache / 'single_slice_v1')
                self.assertEqual(old_file.read_bytes(), b'old three-channel cache')
                torch.testing.assert_close(loader.dataset[0]['condition'], sample['condition'])

    def test_forward_modes_and_z_gradient(self):
        torch.manual_seed(7)
        model = HybridAcousticDiffusion()
        ct, mr = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
        latent, timesteps = torch.randn(2, 4, 4, 4), torch.zeros(2, dtype=torch.long)
        z = torch.tensor([10, 20])
        self.assertEqual(model.ct_encoder.encoder[0].in_channels, 1)
        self.assertEqual(model.mri_encoder.encoder[0].in_channels, 1)
        for ct_input, mr_input in ((ct, mr), (ct, None), (None, mr), (None, None)):
            self.assertEqual(model(latent, timesteps, ct_input, mr_input, z).shape, latent.shape)
        output = model(latent, timesteps, ct, mr, z)
        shifted = model(latent, timesteps, ct, mr, z + 100)
        self.assertFalse(torch.allclose(output, shifted))
        output.square().mean().backward()
        self.assertGreater(model.z_position_encoder[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(latent, timesteps, ct.repeat(1, 3, 1, 1), mr, z)

    def test_dropout_preserves_single_channel(self):
        ct, mr = torch.ones(3, 1, 8, 8), torch.ones(3, 1, 8, 8)
        for mode in ('both', 'ct_only', 'mri_only', 'uncond'):
            probs = {name: float(name == mode) for name in ('both', 'ct_only', 'mri_only', 'uncond')}
            ct_out, mr_out, counts = apply_modality_dropout(ct, mr, probs)
            self.assertEqual(ct_out.shape, ct.shape)
            self.assertEqual(counts[mode], 3)
            self.assertEqual(bool(ct_out.any()), mode in ('both', 'ct_only'))
            self.assertEqual(bool(mr_out.any()), mode in ('both', 'mri_only'))
        ct_out, mr_out, counts = apply_modality_dropout(
            ct, mr, {'both': 1, 'ct_only': 0, 'mri_only': 0, 'uncond': 0},
            weaken_probs={'ct': 1, 'mri': 0}, weaken_range=(0.5, 0.5))
        torch.testing.assert_close(ct_out, ct * 0.5)
        torch.testing.assert_close(mr_out, mr)
        torch.testing.assert_close(ct, torch.ones_like(ct))
        self.assertEqual(counts['ct_weakened'], 3)

    def test_legacy_three_channel_weights_are_incompatible(self):
        model = HybridAcousticDiffusion()
        state = model.state_dict()
        state['ct_encoder.encoder.0.weight'] = torch.zeros(16, 3, 3, 3)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=False)

    def test_distillation_uses_single_slice_teacher(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'teacher.pth'
            torch.save({'model_state_dict': HybridAcousticDiffusion().state_dict()}, checkpoint)
            cfg = Config(model=Config(pretrained_model_name_or_path='mock', latent_channels=4,
                                      cond_channels=4, max_z_slices=256, gradient_checkpointing=False),
                         paths=Config(teacher_sd_ckpt=str(checkpoint)), strict_teacher_load=True)
            teacher = dmd2.build_teacher_model(cfg, torch.device('cpu'))
            generator = dmd2.build_generator_model(cfg, torch.device('cpu'))
            fake = dmd2.build_fake_model(cfg, torch.device('cpu'))
            latent = torch.randn(2, 4, 4, 4)
            image = torch.randn(2, 1, 32, 32)
            z = torch.tensor([12, 45])
            for model in (teacher, generator, fake):
                self.assertEqual(model.ct_encoder.encoder[0].in_channels, 1)
                result = model(latent, torch.zeros(2, dtype=torch.long), image, image, z)
                self.assertEqual(result.shape, latent.shape)
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
            self.assertEqual(tuple(fake.discriminate(latent, image, image, z).shape), (2,))

    def test_training_passes_z_to_model(self):
        tree = ast.parse((ROOT / 'train/train_sd.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'model']
        self.assertTrue(calls)
        for call in calls:
            self.assertTrue(any(arg.arg == 'slice_idx' and isinstance(arg.value, ast.Name)
                                and arg.value.id == 'slice_idx' for arg in call.keywords))


if __name__ == '__main__':
    unittest.main()
