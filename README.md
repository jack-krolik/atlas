# ATLAS — Attention-based Temporal Learning for Automated Sleep Staging

ATLAS is a deep learning model for automated sleep stage classification from a minimal four-channel bipolar EEG montage (F3–C3, C3–O1, F4–C4, C4–O2). It runs on either standard polysomnography or clinical EEG recordings that lack a full PSG instrument set, and produces epoch-level, real-time predictions with variable-length temporal context.

The architecture pairs multi-kernel convolutional feature extraction with GPU-native spectral features, cross-epoch self-attention, and a bidirectional LSTM, and supports subject-adaptive fine-tuning to personalize a pretrained model with a small number of labeled epochs.

This repository contains the optimized model definition and the distributed training pipeline.

## Repository layout

| File | Purpose |
|------|---------|
| `sleepdetector_optimized.py` | Model definition: `FlexibleSleepStageClassifier` and components, plus the `create_flexible_model` / `create_model` factory functions. |
| `training_optimized.py` | DDP + AMP training pipeline with focal loss, OneCycleLR, resumable checkpointing, and per-fold cross-validation. |
| `create_training_data.py` | Builds the 5-fold pickle and defines `SimpleSequentialDataset`. The training script imports it so pickle can resolve dataset classes. |

## Model

The classifier processes one 30-second epoch at a time, optionally conditioned on a window of preceding epochs.

**Per-epoch encoder (`FlexibleIntraEpochCNN`)**
- Three parallel 1D-CNN branches keyed to different timescales: a slow branch (1 s / 0.5 s kernels), a fast branch (0.25 s / 0.125 s kernels), and a spindle branch (0.5 s / 0.25 s kernels).
- A GPU-native spectral extractor (`TorchSpectralFeatureExtractor`) computes band power with `torch.fft`, avoiding a CPU round-trip. Six bands per channel: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), low gamma (30–40 Hz), and sigma/spindle (12–14 Hz). Band powers are log-scaled and NaN-guarded. The FFT path is forced to FP32 even under AMP, since FP16 overflows on raw EEG.
- CNN and spectral features are fused into a per-epoch embedding.

**Temporal model**
- Sinusoidal positional encoding over the context window.
- `CrossEpochAttention`: multi-head self-attention between the current epoch and its context.
- A 2-layer bidirectional LSTM with layer norm, followed by a small classifier head.

**Output**
- Five classes. The label mapping used throughout training is index `0=N3, 1=N2, 2=N1, 3=REM, 4=Wake` (`SLEEP_STAGES = ['N3', 'N2', 'N1', 'REM', 'Wake']`).
- `return_attention=True` exposes the cross-epoch attention weights, and `get_attention_weights_for_epoch` returns the current-epoch-to-context attention for interpretability.

Default configuration: 4 input channels, 100 Hz sampling, 30 s epochs (3000 samples/epoch), 256 CNN features, 128-unit LSTM (×2 layers, bidirectional), 8 attention heads, `max_history=32`.

## Data format

Training reads MATLAB `.mat` files (one per night) via `scipy.io.loadmat`, each containing:
- `sig1`, `sig2`, `sig3`, `sig4` — the four bipolar EEG channels, sampled at 100 Hz and segmented into 30 s epochs (3000 samples each).
- `labels` — integer stage labels per epoch using the mapping above. A label of `-1` marks unscored/invalid epochs.

Epochs containing NaNs or a `-1` label are dropped at load time. The dataset (`SimpleSequentialDataset`) returns the current epoch plus up to `context_window` preceding epochs, and caches whole nights in RAM to avoid repeated disk reads.

The training entry point loads a pickled, pre-split 5-fold structure (the `--data_path` argument), produced by `create_training_data.py`. Because that script defines the dataset class as `__main__.SimpleSequentialDataset` when run directly, the training script uses a custom unpickler to remap those references — keep `create_training_data.py` importable on the path.

## Requirements

- Python 3.10 (developed on pyenv `ucsf`, 3.10.13)
- PyTorch 2.x with CUDA (uses `torch.amp`, `torch.compile`, and `torch.fft`)
- NumPy, SciPy, scikit-learn, tqdm

A CUDA-capable GPU is expected for training. The pipeline falls back to single-GPU mode automatically when not launched under `torchrun`.

## Training

Multi-GPU with `torchrun`:

```bash
torchrun --nproc_per_node=NUM_GPUS training_optimized.py \
    --data_path /path/to/TRAINING_DATA_5FOLD.pkl \
    --output_dir /path/to/outputs \
    --run_name my_run \
    --epochs 50 \
    --batch_size_per_gpu 24
```

Single GPU:

```bash
python training_optimized.py --data_path /path/to/TRAINING_DATA_5FOLD.pkl --run_name my_run
```

### Key arguments

| Argument | Default | Notes |
|----------|---------|-------|
| `--epochs` | 50 | Training epochs per fold. |
| `--lr` | 3e-4 | Max LR for OneCycleLR. |
| `--weight_decay` | 1e-5 | AdamW weight decay. |
| `--batch_size_per_gpu` | 24 | Per-GPU batch size. |
| `--accumulation_steps` | 1 | Gradient accumulation. |
| `--grad_clip` | 5.0 | Gradient norm clip. |
| `--loss_type` | `focal` | `focal` or `ce`. |
| `--focal_gamma` | 2.0 | Focal loss focusing parameter. |
| `--label_smoothing` | 0.05 | Label smoothing. |
| `--weight_method` | `sqrt` | Class weighting: `sqrt`, `balanced`, or `none`. |
| `--context_window` | 15 | Preceding epochs of context. |
| `--no_attention` | off | Disable cross-epoch attention. |
| `--no_amp` | off | Disable mixed precision. |
| `--compile` | off | Enable `torch.compile` (may conflict with FFT ops). |
| `--num_workers` | 8 | DataLoader workers per process. |
| `--night_cache_size` | 50 | Nights cached in RAM per worker. |
| `--run_name` | `run_default` | Stable run name; resubmitting the same name auto-resumes. |
| `--skip_folds` | none | Folds to skip, e.g. `--skip_folds fold_0 fold_1`. |
| `--checkpoint_every` | 3 | Checkpoint cadence in epochs. |

### What the pipeline does

- **Distributed + mixed precision.** `DistributedDataParallel`, AMP with `GradScaler`, cuDNN benchmark mode, `set_to_none=True` zeroing, and an optional `torch.compile` pass.
- **Class imbalance handling.** Focal loss with square-root inverse-frequency class weights (cached after first computation).
- **Schedule.** AdamW with OneCycleLR (10% warmup, cosine anneal).
- **Resumability.** Each run writes to a stable directory keyed on `--run_name`. Progress is tracked per fold; completed folds are skipped and interrupted folds resume from the latest checkpoint. Resubmitting the same job picks up where it left off.
- **Validation.** Macro F1 and accuracy per epoch, with predictions all-gathered across ranks for correct metric computation. The best checkpoint per fold (by macro F1) is saved alongside a per-class classification report.

### Outputs

Under `<output_dir>/<run_name>/`:
- `best_models/` — best checkpoint and classification report per fold.
- `checkpoints/` — latest resumable checkpoint per fold (cleaned up on fold completion).
- `logs/` — timestamped training logs.
- `progress.pkl` — fold completion tracker.

## Performance

On held-out public PSG benchmarks the model reaches 76.2% accuracy (0.742 macro F1) on ABC and 77.6% accuracy (0.718 macro F1) on HomePAP. Across all 11 evaluation datasets it averages 74.2 ± 9.9% accuracy and 0.637 ± 0.146 macro F1, with N1 the dominant source of confusion. Prediction confidence tracks classification difficulty, which supports human-in-the-loop review of uncertain epochs. Subject-adaptive fine-tuning with as few as 67 labeled epochs improved macro F1 on all 11 datasets (+0.033 mean).

The model was trained on 1,946 nights (≈1.8M epochs) of scalp EEG drawn from 12 datasets across three independent sources.

## Citation

Krolik J., Khambhati A.N., Zhao R., Krystal A.K., Fan J.M. *ATLAS: Attention-based Temporal Learning for Automated Sleep Staging from Minimal EEG Montages.* University of California San Francisco. (Manuscript; update with venue/DOI when available.)
