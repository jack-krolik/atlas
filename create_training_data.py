#!/usr/bin/env python3
"""
create_training_data.py
=======================
Builds a 5-fold subject-level cross-validation dataset pickle for training_optimized.py.

Subjects are identified by stripping night suffixes (e.g. STNF00003 and STNF00003_1
are the same subject). All nights from a subject stay in the same fold to prevent
data leakage.

Folds are stratified by dataset so each fold has proportional representation from
all 11 source datasets.

Usage:
    python create_training_data.py [--output_path ...] [--context_window 15] [--n_folds 5]

Output:
    A pickle file containing:
        {
            'fold_0': {'train_loader': DataLoader, 'val_loader': DataLoader},
            'fold_1': { ... },
            ...
        }
    Compatible with training_optimized.py's expected format.
"""

import os
import sys
import re
import pickle
import argparse
import numpy as np
import scipy.io as sio
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader


# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = "/userdata/jkrolik/trainingSleepData/NEWPREPROCESSED"

# To:
DATASETS = [
    "STNF", "SASE1", "Psoriasis", "PCOS", "SASE2", "CASI", "MemS"
]

SLEEP_STAGES = {0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'Wake', -1: 'Artifact'}


# ==============================================================================
# Dataset class (must match training_optimized.py's SimpleSequentialDataset)
# ==============================================================================

class SimpleSequentialDataset(torch.utils.data.Dataset):
    """
    Dataset with NaN filtering and context window.
    Identical to the one in training_optimized.py so the pickle is compatible.
    """
    def __init__(self, night_files, context_window=15, max_cache_size=50):
        self.night_files = night_files
        self.context_window = context_window
        self.max_cache_size = max_cache_size
        self.epoch_index = []
        self.night_cache = {}

        for night_idx, (dataset_name, file_path) in enumerate(night_files):
            try:
                mat_data = sio.loadmat(
                    file_path,
                    variable_names=["sig1", "sig2", "sig3", "sig4", "labels"]
                )
                labels = mat_data["labels"].flatten()
                x = np.stack((
                    mat_data["sig1"], mat_data["sig2"],
                    mat_data["sig3"], mat_data["sig4"]
                ), axis=1)

                has_nan = np.any(np.isnan(x), axis=(1, 2))
                valid_mask = (labels != -1) & (~has_nan)
                valid_indices = np.where(valid_mask)[0]

                for i in range(len(valid_indices)):
                    self.epoch_index.append((night_idx, i))

                n_nan = int(np.sum(has_nan))
                n_removed = int(np.sum(~valid_mask))
                print(f"  {os.path.basename(file_path)}: {len(valid_indices)} valid "
                      f"(removed {n_removed}, {n_nan} NaN)")

            except Exception as e:
                print(f"  SKIP {file_path}: {e}")
                continue

    def _load_night(self, night_idx):
        if night_idx in self.night_cache:
            return self.night_cache[night_idx]

        dataset_name, file_path = self.night_files[night_idx]
        mat_data = sio.loadmat(
            file_path,
            variable_names=["sig1", "sig2", "sig3", "sig4", "labels"]
        )
        x = np.stack((
            mat_data["sig1"], mat_data["sig2"],
            mat_data["sig3"], mat_data["sig4"]
        ), axis=1)
        y = mat_data["labels"].flatten()

        has_nan = np.any(np.isnan(x), axis=(1, 2))
        valid_mask = (y != -1) & (~has_nan)
        x = x[valid_mask]
        y = y[valid_mask]

        night_data = {
            "X": torch.FloatTensor(x),
            "Y": torch.LongTensor(y)
        }

        if len(self.night_cache) >= self.max_cache_size:
            oldest_key = min(self.night_cache.keys())
            del self.night_cache[oldest_key]

        self.night_cache[night_idx] = night_data
        return night_data

    def __getitem__(self, idx):
        night_idx, valid_epoch_idx = self.epoch_index[idx]
        night_data = self._load_night(night_idx)

        current_epoch = night_data["X"][valid_epoch_idx]
        current_label = night_data["Y"][valid_epoch_idx]

        context_start = max(0, valid_epoch_idx - self.context_window)
        context_epochs = night_data["X"][context_start:valid_epoch_idx]

        return {
            "current_epoch": current_epoch,
            "context_epochs": context_epochs if len(context_epochs) > 0 else None,
            "label": current_label,
            "night_idx": night_idx
        }

    def __len__(self):
        return len(self.epoch_index)


def simple_collate_fn(batch):
    """Collate function matching training_optimized.py."""
    batch_size = len(batch)
    current_epochs = torch.stack([item['current_epoch'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    night_indices = torch.tensor([item['night_idx'] for item in batch], dtype=torch.long)

    first_context = batch[0].get('context_epochs')
    if first_context is None or len(first_context) == 0:
        return {
            'current_epoch': current_epochs,
            'context_epochs': None,
            'labels': labels,
            'night_indices': night_indices
        }

    max_len = max(
        (len(item['context_epochs']) if item.get('context_epochs') is not None else 0)
        for item in batch
    )
    context_batch = torch.zeros(batch_size, max_len, 4, 3000, dtype=current_epochs.dtype)

    for i, item in enumerate(batch):
        c = item.get('context_epochs')
        if c is not None and len(c) > 0:
            context_batch[i, -len(c):] = c

    return {
        'current_epoch': current_epochs,
        'context_epochs': context_batch,
        'labels': labels,
        'night_indices': night_indices
    }


# ==============================================================================
# Subject / night discovery
# ==============================================================================

def extract_subject_id(filename, dataset_name):
    """
    Extract base subject ID from a filename, grouping multi-night recordings.
    
    Examples:
        preprocessed_STNF_STNF00003.mat   -> STNF_STNF00003
        preprocessed_STNF_STNF00003_1.mat -> STNF_STNF00003  (same subject)
        preprocessed_MSTR_MSTR00001.mat    -> MSTR_MSTR00001
    """
    # Strip prefix and .mat suffix
    stem = filename.replace("preprocessed_", "").replace(".mat", "")
    # Remove trailing _N night index (e.g. _1, _2) but NOT part of the subject ID
    # Subject IDs look like STNF00003, night suffixes are _1, _2 etc.
    # Pattern: {DATASET}_{SUBJECTID} or {DATASET}_{SUBJECTID}_{nightnum}
    # The subject ID itself contains digits, so we match _N at the very end
    # where N is a small number (1-9) after the main subject ID pattern
    base = re.sub(r'_(\d{1})$', '', stem)
    return base


def discover_nights(base_dir, datasets):
    """
    Scan all datasets and return:
        - night_files: list of (dataset_name, filepath)
        - subject_ids: list of subject ID per night (for grouping)
        - dataset_labels: list of dataset index per night (for stratification)
    """
    night_files = []
    subject_ids = []
    dataset_labels = []

    for ds_idx, ds_name in enumerate(datasets):
        ds_dir = os.path.join(base_dir, ds_name)
        if not os.path.isdir(ds_dir):
            print(f"WARNING: Directory not found: {ds_dir}")
            continue

        mat_files = sorted([f for f in os.listdir(ds_dir) if f.endswith('.mat')])
        print(f"{ds_name}: {len(mat_files)} night files")

        for fname in mat_files:
            fpath = os.path.join(ds_dir, fname)
            subj_id = extract_subject_id(fname, ds_name)

            night_files.append((ds_name, fpath))
            subject_ids.append(subj_id)
            dataset_labels.append(ds_idx)

    return night_files, subject_ids, dataset_labels


# ==============================================================================
# Fold creation
# ==============================================================================

def create_folds(night_files, subject_ids, dataset_labels, n_folds=5,
                 context_window=15, seed=42):
    """
    Create n_folds using StratifiedGroupKFold.
    
    - Groups: subject IDs (all nights from same subject in same fold)
    - Stratification: dataset labels (proportional dataset representation per fold)
    """
    night_files_arr = np.array(night_files, dtype=object)
    subject_ids_arr = np.array(subject_ids)
    dataset_labels_arr = np.array(dataset_labels)

    # Map subject IDs to unique group indices for sklearn
    unique_subjects = list(dict.fromkeys(subject_ids))  # preserves order
    subj_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subj_to_idx[s] for s in subject_ids])

    print(f"\nTotal: {len(night_files)} nights, {len(unique_subjects)} unique subjects")
    print(f"Creating {n_folds}-fold CV with subject-level grouping + dataset stratification\n")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = {}

    for fold_idx, (train_indices, val_indices) in enumerate(
        sgkf.split(night_files_arr, dataset_labels_arr, groups)
    ):
        fold_name = f"fold_{fold_idx}"

        train_nights = [night_files[i] for i in train_indices]
        val_nights = [night_files[i] for i in val_indices]

        # Count subjects and datasets per split
        train_subjects = set(subject_ids_arr[train_indices])
        val_subjects = set(subject_ids_arr[val_indices])
        overlap = train_subjects & val_subjects

        print(f"{'='*60}")
        print(f"{fold_name}")
        print(f"  Train: {len(train_nights)} nights, {len(train_subjects)} subjects")
        print(f"  Val:   {len(val_nights)} nights, {len(val_subjects)} subjects")
        if overlap:
            print(f"  WARNING: {len(overlap)} subjects in BOTH train and val!")
        else:
            print(f"  No subject leakage (verified)")

        # Dataset breakdown
        train_ds_counts = defaultdict(int)
        val_ds_counts = defaultdict(int)
        for ds_name, _ in train_nights:
            train_ds_counts[ds_name] += 1
        for ds_name, _ in val_nights:
            val_ds_counts[ds_name] += 1

        print(f"  {'Dataset':<12} {'Train':>6} {'Val':>6}")
        for ds_name in DATASETS:
            tr = train_ds_counts.get(ds_name, 0)
            va = val_ds_counts.get(ds_name, 0)
            if tr + va > 0:
                print(f"  {ds_name:<12} {tr:>6} {va:>6}")

        # Build datasets
        print(f"\n  Building train dataset...")
        train_dataset = SimpleSequentialDataset(train_nights, context_window=context_window)
        print(f"  Building val dataset...")
        val_dataset = SimpleSequentialDataset(val_nights, context_window=context_window)

        # Create placeholder DataLoaders (training script recreates these with proper settings)
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            collate_fn=simple_collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            collate_fn=simple_collate_fn, num_workers=0
        )

        folds[fold_name] = {
            'train_loader': train_loader,
            'val_loader': val_loader,
        }

        print(f"  Train epochs: {len(train_dataset)}, Val epochs: {len(val_dataset)}")
        print()

    return folds


# ==============================================================================
# Label distribution summary
# ==============================================================================

def compute_label_stats(night_files):
    """
    Print overall label distribution by scanning labels from unique files once.
    Only loads the 'labels' variable (not signals), so it's fast.
    """
    print("=" * 60)
    print("LABEL DISTRIBUTION (across all nights)")
    print("=" * 60)

    all_labels = []
    for ds_name, fpath in night_files:
        try:
            mat_data = sio.loadmat(fpath, variable_names=["labels"])
            labels = mat_data["labels"].flatten()
            valid = labels[labels != -1]
            all_labels.append(valid)
        except Exception as e:
            print(f"  Warning: could not read labels from {fpath}: {e}")
            continue

    all_labels = np.concatenate(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)

    print(f"Total valid epochs: {total}")
    for cls, count in zip(unique, counts):
        name = SLEEP_STAGES.get(int(cls), f"Unknown({cls})")
        pct = count / total * 100
        print(f"  {name:<8}: {count:>8} ({pct:>5.1f}%)")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create 5-fold training dataset")
    parser.add_argument('--base_dir', type=str, default=BASE_DIR,
                        help='Path to NEWPREPROCESSED directory')
    parser.add_argument('--output_path', type=str,
                        default='/userdata/jkrolik/NEW_MODELS/TRAINING_DATA_5FOLD_7DS.pkl',
                        help='Output pickle file path')
    parser.add_argument('--context_window', type=int, default=15,
                        help='Context window size')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for fold assignment')
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING DATA CREATION")
    print(f"  Source: {args.base_dir}")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Context window: {args.context_window}")
    print(f"  Output: {args.output_path}")
    print("=" * 60)
    print()

    # Discover all nights
    night_files, subject_ids, dataset_labels = discover_nights(args.base_dir, DATASETS)

    if len(night_files) == 0:
        print("ERROR: No .mat files found!")
        sys.exit(1)

    # Create folds
    folds = create_folds(
        night_files, subject_ids, dataset_labels,
        n_folds=args.n_folds,
        context_window=args.context_window,
        seed=args.seed
    )

    # Print stats (fast — only reads labels, not signals)
    compute_label_stats(night_files)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(folds, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Done! File size: {file_size:.1f} MB")
    print(f"\nTo train:")
    print(f"  torchrun --nproc_per_node=4 training_optimized.py \\")
    print(f"    --data_path {args.output_path}")


if __name__ == "__main__":
    main()