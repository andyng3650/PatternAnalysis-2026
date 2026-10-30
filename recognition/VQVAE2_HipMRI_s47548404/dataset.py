"""
dataset.py - Data loading for HipMRI 2D slices (COMP3710, task 2.4).

Uses the provided patient-level split (train / validate / test).
Each item is one 2D slice, normalised per slice to [0, 1],
returned as a float32 tensor of shape (1, H, W).
"""
import argparse
import glob
import os
import re
from collections import Counter

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validate", "test")


def patient_id(path):
    """'case_004_week_0_slice_10.nii.gz' -> '004'."""
    return re.search(r"_(\d+)_week_", os.path.basename(path)).group(1)


def load_slice(path):
    """Load one NIfTI slice as a float32 numpy array."""
    return nib.load(path).get_fdata(dtype=np.float32)


def normalise(img):
    """Per-slice min-max scaling to [0, 1]. Flat slices become all zeros."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


class HipMRIDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        folder = os.path.join(data_dir, f"keras_slices_{split}")
        self.files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
        if not self.files:
            raise FileNotFoundError(f"No .nii.gz files found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = normalise(load_slice(self.files[idx]))
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def check_split(data_dir, split):
    """Sanity checks: file count, patients, shapes of ALL slices, one batch."""
    ds = HipMRIDataset(data_dir, split)
    patients = {patient_id(f) for f in ds.files}
    shapes = Counter(load_slice(f).shape for f in ds.files)
    print(f"[{split}] {len(ds)} slices from {len(patients)} patients")
    print(f"[{split}] shapes across all slices: {dict(shapes)}")

    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print(f"[{split}] batch shape {tuple(batch.shape)}, "
          f"range {batch.min():.3f} to {batch.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the HipMRI dataset.")
    parser.add_argument("--data-dir", required=True,
                        help="Path to the keras_slices_data folder")
    args = parser.parse_args()
    for split in SPLITS:
        check_split(args.data_dir, split)