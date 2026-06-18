# training_optimized.py
# =============================================================================
# Optimized training script for FlexibleSleepStageClassifier
# 
# Key changes from training_improved.py:
#   1. DistributedDataParallel (DDP) instead of DataParallel
#   2. Mixed precision training (AMP) with GradScaler
#   3. cuDNN benchmark mode enabled (auto-tunes conv algorithms)
#   4. GPU-native spectral features (no CPU roundtrip)
#   5. torch.compile() support (PyTorch 2.x graph optimization)
#   6. optimizer.zero_grad(set_to_none=True) for faster zeroing
#   7. Higher default batch size and num_workers
#   8. Larger night cache to reduce .mat reloads
#
# Launch with torchrun:
#   torchrun --nproc_per_node=NUM_GPUS training_optimized.py [args]
#
# Single GPU fallback:
#   python training_optimized.py [args]
# =============================================================================

import os
import sys
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import argparse

os.environ['TQDM_DISABLE'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, classification_report
import time
import logging
from datetime import datetime
import pickle
import random


# ==============================================================================
# DDP Utilities
# ==============================================================================

def setup_ddp():
    """
    Initialize DDP if launched via torchrun / torch.distributed.launch.
    Returns (local_rank, world_size, is_distributed).
    Falls back to single-GPU mode if not launched distributed.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size, True
    else:
        # Single GPU fallback
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, False


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Returns True on rank 0 (or if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_rank0(msg, logger=None):
    """Only print/log on rank 0."""
    if is_main_process():
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ==============================================================================
# Focal Loss
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==============================================================================
# Seed & Logging
# ==============================================================================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: deterministic=False and benchmark=True for speed
    # Set deterministic=True only for final reproducibility runs
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if is_main_process():
        print(f"Global seed set to {seed} | cuDNN benchmark=True, deterministic=False")


class RobustFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except OSError as e:
            if e.errno == 116:
                try:
                    self.close()
                    self.stream = self._open()
                    super().emit(record)
                except:
                    pass
            else:
                raise


def setup_logging(log_dir):
    """Setup logging — only rank 0 writes to file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process():
        handlers.append(RobustFileHandler(log_filename))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=handlers,
        force=True
    )

    logger = logging.getLogger(__name__)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Training Session Started (OPTIMIZED DDP + AMP)")
        logger.info("=" * 60)
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"Distributed: {dist.is_initialized()}")
        if dist.is_initialized():
            logger.info(f"World size: {dist.get_world_size()}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)

    return logger, log_filename, timestamp


# ==============================================================================
# Model initialization
# ==============================================================================

# Import from the optimized model file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sleepdetector_optimized import FlexibleSleepStageClassifier, create_flexible_model

# Import create_training_data so pickle can resolve its classes
import create_training_data


class _DataUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps __main__ refs to create_training_data.
    
    When create_training_data.py runs as a script, pickle stores classes as
    __main__.SimpleSequentialDataset. This unpickler redirects those lookups
    to the create_training_data module.
    """
    def find_class(self, module, name):
        if module == '__main__' and hasattr(create_training_data, name):
            return getattr(create_training_data, name)
        return super().find_class(module, name)


def load_training_data(path):
    """Load training data pickle with class remapping."""
    with open(path, 'rb') as f:
        return _DataUnpickler(f).load()


def initialize_model_weights(model):
    """Initialize model weights with standard schemes."""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)
    return model


# ==============================================================================
# Checkpointing & Progress Tracking
# ==============================================================================

def get_run_dirs(output_dir, run_name):
    """Return stable directory paths for a given run_name (no timestamps)."""
    run_dir = os.path.join(output_dir, run_name)
    return {
        'run_dir': run_dir,
        'best_models': os.path.join(run_dir, 'best_models'),
        'checkpoints': os.path.join(run_dir, 'checkpoints'),
        'progress_file': os.path.join(run_dir, 'progress.pkl'),
        'logs': os.path.join(run_dir, 'logs'),
    }


def load_progress(progress_file):
    """
    Load progress tracker. Returns dict:
        {
            'completed_folds': {'fold_0': {'best_val_f1': ..., 'best_val_acc': ...}, ...},
            'started_at': '...',
            'last_updated': '...',
        }
    """
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    return {
        'completed_folds': {},
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'last_updated': None,
    }


def save_progress(progress_file, progress):
    """Save progress tracker (rank 0 only)."""
    if not is_main_process():
        return
    progress['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(progress_file, 'wb') as f:
        pickle.dump(progress, f)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, fold_name,
                   best_val_f1, run_dirs):
    """Save training state to the run's checkpoint directory (rank 0 only)."""
    if not is_main_process():
        return None

    checkpoint_dir = run_dirs['checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_f1': best_val_f1,
        'fold_name': fold_name
    }

    # Keep only the latest checkpoint per fold (overwrite)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{fold_name}.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def find_checkpoint_for_fold(fold_name, run_dirs):
    """Look for an existing checkpoint for this fold. Returns path or None."""
    checkpoint_path = os.path.join(run_dirs['checkpoints'], f'checkpoint_{fold_name}.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, scaler=None, device='cuda'):
    """Load checkpoint and restore training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    model_is_ddp = hasattr(model, 'module')
    state_is_ddp = any(k.startswith('module.') for k in state_dict.keys())

    if model_is_ddp and not state_is_ddp:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not model_is_ddp and state_is_ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_f1 = checkpoint['best_val_f1']
    return start_epoch, best_val_f1, checkpoint['fold_name']


def cleanup_fold_checkpoint(fold_name, run_dirs):
    """Remove checkpoint for a completed fold (rank 0 only)."""
    if not is_main_process():
        return
    checkpoint_path = os.path.join(run_dirs['checkpoints'], f'checkpoint_{fold_name}.pth')
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except:
            pass


# ==============================================================================
# Metrics
# ==============================================================================

SLEEP_STAGES = ['N3', 'N2', 'N1', 'REM', 'Wake']


def log_per_class_metrics(all_labels, all_preds, prefix="", logger=None):
    """Log detailed per-class metrics (rank 0 only)."""
    if not is_main_process():
        return

    logger.info(f"\n{prefix} Per-Class Performance:")
    logger.info("-" * 70)
    logger.info(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    logger.info("-" * 70)

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    for class_idx, class_name in enumerate(SLEEP_STAGES):
        tp = np.sum((labels_arr == class_idx) & (preds_arr == class_idx))
        fp = np.sum((labels_arr != class_idx) & (preds_arr == class_idx))
        fn = np.sum((labels_arr == class_idx) & (preds_arr != class_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(labels_arr == class_idx)

        logger.info(f"{class_name:<8} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

    logger.info("-" * 70)


# ==============================================================================
# Dataset
# ==============================================================================

class SimpleSequentialDataset(torch.utils.data.Dataset):
    """
    Dataset with NaN filtering and context window.
    Increased night cache to reduce .mat reloading overhead.
    """
    def __init__(self, night_files, context_window=15, max_cache_size=50):
        self.night_files = night_files
        self.context_window = context_window
        self.max_cache_size = max_cache_size
        self.epoch_index = []
        self.night_cache = {}

        for night_idx, (dataset_name, file_path) in enumerate(night_files):
            try:
                mat_data = sio.loadmat(file_path, variable_names=["sig1", "sig2", "sig3", "sig4", "labels"])
                labels = mat_data["labels"].flatten()
                x = np.stack((mat_data["sig1"], mat_data["sig2"], mat_data["sig3"], mat_data["sig4"]), axis=1)

                has_nan = np.any(np.isnan(x), axis=(1, 2))
                valid_mask = (labels != -1) & (~has_nan)
                valid_indices = np.where(valid_mask)[0]

                for i, _ in enumerate(valid_indices):
                    self.epoch_index.append((night_idx, i))

                if is_main_process():
                    n_nan = np.sum(has_nan)
                    n_removed = np.sum(~valid_mask)
                    print(f"  {os.path.basename(file_path)}: {len(valid_indices)} valid "
                          f"(removed {n_removed}, {n_nan} NaN)")
            except Exception as e:
                if is_main_process():
                    print(f"  Skipping {file_path}: {e}")
                continue

    def _load_night(self, night_idx):
        if night_idx in self.night_cache:
            return self.night_cache[night_idx]

        dataset_name, file_path = self.night_files[night_idx]
        mat_data = sio.loadmat(file_path, variable_names=["sig1", "sig2", "sig3", "sig4", "labels"])

        x = np.stack((mat_data["sig1"], mat_data["sig2"], mat_data["sig3"], mat_data["sig4"]), axis=1)
        y = mat_data["labels"].flatten()

        has_nan = np.any(np.isnan(x), axis=(1, 2))
        valid_mask = (y != -1) & (~has_nan)
        x = x[valid_mask]
        y = y[valid_mask]

        night_data = {
            "X": torch.FloatTensor(x),
            "Y": torch.LongTensor(y)
        }

        # Larger cache — most HPC nodes have plenty of RAM
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
    """Collate function with pre-allocated context tensor."""
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

    max_len = max((len(item['context_epochs']) if item.get('context_epochs') is not None else 0)
                  for item in batch)
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
# Class weights
# ==============================================================================

def compute_sqrt_class_weights(train_folds, logger):
    """Compute class weights using square root of inverse frequency.
    Loads labels once per unique file instead of per-epoch."""
    print_rank0("Computing square root scaled class weights...", logger)
    
    # Collect unique files across all training splits
    unique_files = set()
    for fold_name, fold_data in train_folds.items():
        ds = fold_data['train_loader'].dataset
        for night_idx in set(ni for ni, _ in ds.epoch_index):
            unique_files.add(ds.night_files[night_idx][1])  # file path

    # Load only labels from each unique file (fast — no signal data)
    all_labels = []
    for fpath in unique_files:
        try:
            mat_data = sio.loadmat(fpath, variable_names=['labels'])
            labels = mat_data['labels'].flatten()
            valid = labels[labels != -1]
            all_labels.append(valid)
        except:
            continue

    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    if len(all_labels) == 0:
        return torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])

    unique_classes, counts = np.unique(all_labels, return_counts=True)
    frequencies = counts / len(all_labels)
    weights = np.sqrt(1.0 / (frequencies * len(unique_classes)))
    weights = weights / weights.mean()

    if is_main_process():
        logger.info("Class distribution and weights:")
        for cls, count, freq, weight in zip(unique_classes, counts, frequencies, weights):
            label = SLEEP_STAGES[cls] if cls < len(SLEEP_STAGES) else f"Unknown({cls})"
            logger.info(f"  {label:<6}: {count:>6} ({freq*100:.1f}%) -> weight={weight:.3f}")

    return torch.FloatTensor(weights)


def get_or_compute_class_weights(weights_path, train_folds, method, logger):
    """Load or compute class weights."""
    default_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    default_weights_path = os.path.join(default_cache_dir, f"class_weights_{method}.pkl")

    if weights_path is None:
        weights_path = default_weights_path

    if os.path.exists(weights_path):
        print_rank0(f"Loading class weights from {weights_path}...", logger)
        with open(weights_path, 'rb') as f:
            cached_data = pickle.load(f)
            return torch.FloatTensor(cached_data['weights'])

    weights = compute_sqrt_class_weights(train_folds, logger) if method == 'sqrt' else torch.ones(5)

    if is_main_process():
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        cache_data = {
            'weights': weights.numpy(),
            'method': method,
            'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(weights_path, 'wb') as f:
            pickle.dump(cache_data, f)

    return weights


# ==============================================================================
# Arguments
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train EEG Sleep Stage Classifier (Optimized DDP + AMP)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4, help='Max learning rate for OneCycleLR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size_per_gpu', type=int, default=24,
                        help='Batch size per GPU (higher with AMP)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # Loss
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--class_weights', type=str, default=None)
    parser.add_argument('--weight_method', type=str, default='sqrt',
                        choices=['balanced', 'sqrt', 'none'])

    # Model
    parser.add_argument('--context_window', type=int, default=15)
    parser.add_argument('--no_attention', action='store_true', default=False)

    # Optimization flags
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='Disable mixed precision (AMP)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile (may conflict with FFT ops)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers per process')
    parser.add_argument('--night_cache_size', type=int, default=50,
                        help='Max nights to cache in RAM per worker')

    # Paths & run management
    parser.add_argument('--output_dir', type=str,
                        default='/userdata/jkrolik/NEW_MODELS/',
                        help='Base directory for all outputs')
    parser.add_argument('--data_path', type=str,
                        default='/userdata/jkrolik/NEW_MODELS/TRAINING_DATA_5FOLD.pkl',
                        help='Path to training data pickle')
    parser.add_argument('--run_name', type=str, default='run_default',
                        help='Fixed name for this training run. Resubmitting with the same '
                             'run_name auto-resumes from where it left off.')
    parser.add_argument('--skip_folds', type=str, nargs='+', default=None,
                        help='Folds to skip (e.g. --skip_folds fold_0 fold_1)')
    parser.add_argument('--checkpoint_every', type=int, default=3,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


# ==============================================================================
# Data preloading
# ==============================================================================

def _preload_dataset(dataset, logger=None):
    """
    Preload all night files into the dataset's cache.
    Eliminates disk I/O during training — first epoch is fast too.
    """
    unique_nights = set(ni for ni, _ in dataset.epoch_index)
    dataset.max_cache_size = len(unique_nights) + 10  # Ensure cache fits everything
    
    loaded = 0
    for night_idx in sorted(unique_nights):
        if night_idx not in dataset.night_cache:
            dataset._load_night(night_idx)
            loaded += 1
    
    # Report memory usage
    total_epochs = sum(d['X'].shape[0] for d in dataset.night_cache.values())
    mem_gb = sum(d['X'].nelement() * 4 + d['Y'].nelement() * 8 
                 for d in dataset.night_cache.values()) / 1e9
    
    if is_main_process() and logger:
        logger.info(f"    Loaded {len(dataset.night_cache)} nights, "
                   f"{total_epochs} epochs, {mem_gb:.1f}GB RAM")


# ==============================================================================
# Training loop for a single fold
# ==============================================================================

def train_fold(fold_name, fold_data, global_class_weights, args, fold_idx,
               total_folds, run_dirs, local_rank, world_size, is_distributed,
               logger):
    """
    Optimized training loop with DDP + AMP.
    Auto-resumes from checkpoint if one exists for this fold.
    """

    device = torch.device(f"cuda:{local_rank}")
    use_amp = not args.no_amp

    print_rank0(f"\n{'='*60}", logger)
    print_rank0(f"Starting {fold_name} ({fold_idx+1}/{total_folds}) — DDP+AMP", logger)
    print_rank0(f"{'='*60}", logger)

    # Log hyperparameters
    if is_main_process():
        logger.info(f"  Max LR: {args.lr}")
        logger.info(f"  Loss: {args.loss_type} (gamma={args.focal_gamma})")
        logger.info(f"  Label smoothing: {args.label_smoothing}")
        logger.info(f"  Grad clip: {args.grad_clip}")
        logger.info(f"  Accumulation: {args.accumulation_steps}")
        logger.info(f"  Context window: {args.context_window}")
        logger.info(f"  AMP: {use_amp}")
        logger.info(f"  Batch/GPU: {args.batch_size_per_gpu}")
        logger.info(f"  Effective batch: {args.batch_size_per_gpu * world_size * args.accumulation_steps}")
        logger.info(f"  Workers: {args.num_workers}")
        logger.info(f"  Distributed: {is_distributed} (world_size={world_size})")

    # ---- Datasets ----
    train_dataset = fold_data['train_loader'].dataset
    val_dataset = fold_data['val_loader'].dataset
    train_dataset.context_window = args.context_window
    val_dataset.context_window = args.context_window

    # Preload all night data into RAM (eliminates disk I/O bottleneck)
    # ~50-60GB for full training set at float32
    print_rank0("  Preloading training data into RAM...", logger)
    _preload_dataset(train_dataset, logger)
    print_rank0("  Preloading validation data into RAM...", logger)
    _preload_dataset(val_dataset, logger)

    # ---- Samplers ----
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=simple_collate_fn,
        drop_last=True  # Avoids uneven batch sizes across GPUs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=simple_collate_fn
    )

    print_rank0(f"  Train batches/GPU: {len(train_loader)}", logger)
    print_rank0(f"  Val batches/GPU: {len(val_loader)}", logger)

    # ---- Model ----
    model = create_flexible_model(
        input_channels=4,
        sampling_rate=100,
        epoch_duration=30,
        use_spectral_features=True,
        num_classes=5,
        use_attention=not args.no_attention
    )
    model = initialize_model_weights(model)
    model = model.to(device)

    # Optional: torch.compile for fused kernels (PyTorch 2.x)
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='default')
            print_rank0("  torch.compile enabled (default mode)", logger)
        except Exception as e:
            print_rank0(f"  torch.compile failed, continuing without: {e}", logger)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # ---- Scheduler ----
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )

    # ---- AMP scaler ----
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- Loss ----
    class_weights = global_class_weights.to(device)
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma,
                             label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args.label_smoothing)

    # ---- Resume ----
    best_val_f1 = 0
    best_val_acc = 0
    start_epoch = 0
    train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []

    # Auto-detect checkpoint for this fold
    resume_checkpoint = find_checkpoint_for_fold(fold_name, run_dirs)
    if resume_checkpoint:
        start_epoch, best_val_f1, _ = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, scaler, device
        )
        print_rank0(f"  RESUMED from epoch {start_epoch}, best F1={best_val_f1:.4f}", logger)

    # ---- Wrap in DDP AFTER loading checkpoint ----
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        print_rank0(f"  Model wrapped in DDP on GPU {local_rank}", logger)

    # ============================
    # TRAINING LOOP
    # ============================
    if is_main_process():
        epoch_pbar = tqdm(range(start_epoch, args.epochs),
                         desc=f"{fold_name}", ncols=110, ascii=True)
    else:
        epoch_pbar = range(start_epoch, args.epochs)

    for epoch in epoch_pbar:
        # Set epoch for distributed sampler (ensures proper shuffling)
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        optimizer.zero_grad(set_to_none=True)

        if is_main_process():
            train_pbar = tqdm(train_loader, desc="  Train", leave=False, ncols=100, ascii=True)
        else:
            train_pbar = train_loader

        for batch_idx, batch in enumerate(train_pbar):
            current_epoch_data = batch['current_epoch'].to(device, non_blocking=True)
            context_epochs = (batch['context_epochs'].to(device, non_blocking=True)
                            if batch['context_epochs'] is not None else None)
            labels = batch['labels'].to(device, non_blocking=True)

            # Detailed diagnostics on first batch (force flush so it appears even if killed)
            if batch_idx < 3:
                ctx_shape = tuple(context_epochs.shape) if context_epochs is not None else None
                n_cnn_passes = current_epoch_data.shape[0]
                if context_epochs is not None:
                    n_cnn_passes += context_epochs.shape[0] * context_epochs.shape[1]
                msg = (f"  Batch 0 shapes: current={tuple(current_epoch_data.shape)}, "
                       f"context={ctx_shape}, "
                       f"total CNN passes={n_cnn_passes}, "
                       f"GPU mem before forward: "
                       f"{torch.cuda.memory_allocated()/1e9:.2f}GB / "
                       f"{torch.cuda.get_device_properties(device).total_memory/1e9:.1f}GB")
                if is_main_process():
                    logger.info(msg)
                    # Force flush all handlers
                    for h in logger.handlers:
                        h.flush()
                print(msg, flush=True)

            if torch.isnan(current_epoch_data).any():
                continue

            # Forward with AMP
            if batch_idx < 3:
                print(f"  [rank {local_rank}] Batch {batch_idx}: starting forward...", flush=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(current_epoch_data, context_epochs)
                loss = criterion(outputs, labels) / args.accumulation_steps
            if batch_idx < 3:
                print(f"  [rank {local_rank}] Batch {batch_idx}: forward OK, loss={loss.item():.4f}, "
                      f"GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

            # Backward with scaler
            scaler.scale(loss).backward()

            if batch_idx < 3:
                print(f"  [rank {local_rank}] Batch {batch_idx}: backward OK, "
                      f"GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB, "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

            # Step with accumulation
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                # scaler.step skips optimizer.step internally if grads are inf/nan
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Always step to keep LR schedule in sync
                optimizer.zero_grad(set_to_none=True)

                if batch_idx < 3:
                    print(f"  [rank {local_rank}] Batch {batch_idx}: optimizer step done, "
                          f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

            # Metrics
            train_loss += loss.item() * args.accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            if is_main_process() and batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'L': f'{loss.item() * args.accumulation_steps:.3f}',
                    'A': f'{train_correct/max(train_total,1):.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.1e}'
                })

            # Log to file every 200 batches so tail -f shows progress
            if is_main_process() and batch_idx > 0 and batch_idx % 200 == 0:
                pct = 100 * batch_idx / len(train_loader)
                avg_loss = train_loss / batch_idx
                acc = train_correct / max(train_total, 1)
                logger.info(f"  [batch {batch_idx}/{len(train_loader)} ({pct:.0f}%)] "
                           f"loss={avg_loss:.4f}, acc={acc:.4f}, "
                           f"lr={scheduler.get_last_lr()[0]:.2e}")

            # Log GPU memory on first batch (helps diagnose OOM)
            if batch_idx == 0 and is_main_process():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                mem_max = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"  GPU memory after batch 0: "
                           f"alloc={mem_alloc:.1f}GB, reserved={mem_reserved:.1f}GB, "
                           f"peak={mem_max:.1f}GB")

            if batch_idx < 3:
                print(f"  [rank {local_rank}] Batch {batch_idx} COMPLETE", flush=True)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            if is_main_process():
                val_pbar = tqdm(val_loader, desc="  Val", leave=False, ncols=100, ascii=True)
            else:
                val_pbar = val_loader

            for batch in val_pbar:
                current_epoch_data = batch['current_epoch'].to(device, non_blocking=True)
                context_epochs = (batch['context_epochs'].to(device, non_blocking=True)
                                if batch['context_epochs'] is not None else None)
                labels = batch['labels'].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(current_epoch_data, context_epochs)
                    loss = criterion(outputs, labels)

                if not torch.isnan(loss):
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

        # ---- Gather metrics across GPUs ----
        if is_distributed:
            # Gather predictions and labels from all ranks for proper F1 computation
            all_val_preds_gathered = [None] * world_size
            all_val_labels_gathered = [None] * world_size
            dist.all_gather_object(all_val_preds_gathered, all_val_preds)
            dist.all_gather_object(all_val_labels_gathered, all_val_labels)
            if is_main_process():
                all_val_preds = [p for sublist in all_val_preds_gathered for p in sublist]
                all_val_labels = [l for sublist in all_val_labels_gathered for l in sublist]
                val_total = len(all_val_labels)
                val_correct = sum(p == l for p, l in zip(all_val_preds, all_val_labels))

        # ---- Compute metrics (rank 0) ----
        if is_main_process():
            train_acc = train_correct / max(train_total, 1)
            train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
            val_acc = val_correct / max(val_total, 1)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            current_lr = scheduler.get_last_lr()[0]

            if (epoch + 1) % 5 == 0:
                log_per_class_metrics(all_val_labels, all_val_preds,
                                     prefix=f"Epoch {epoch+1} Val", logger=logger)

            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            logger.info(f"  LR: {current_lr:.2e}")

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            val_f1_scores.append(val_f1)

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc

                model_to_save = model.module if hasattr(model, 'module') else model
                model_path = os.path.join(run_dirs['best_models'], f'best_model_{fold_name}.pth')
                torch.save(model_to_save.state_dict(), model_path)
                logger.info(f"  New best model! F1: {val_f1:.4f}")

                report = classification_report(all_val_labels, all_val_preds,
                                             target_names=SLEEP_STAGES)
                report_path = os.path.join(run_dirs['best_models'], f'classification_report_{fold_name}.txt')
                with open(report_path, 'w') as f:
                    f.write(f"Epoch {epoch+1}\n")
                    f.write(f"Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}\n\n")
                    f.write(report)

            # Update progress bar
            if isinstance(epoch_pbar, tqdm):
                epoch_pbar.set_postfix({
                    'TrF1': f'{train_f1:.3f}',
                    'VaF1': f'{val_f1:.3f}',
                    'Best': f'{best_val_f1:.3f}'
                })

            # Checkpoint periodically
            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(model, optimizer, scheduler, scaler,
                              epoch, fold_name, best_val_f1, run_dirs)
                logger.info(f"  Checkpoint saved (epoch {epoch+1})")

        # Sync all ranks before next epoch
        if is_distributed:
            dist.barrier()

    # Broadcast best_val_f1 from rank 0 so all ranks have it
    if is_distributed:
        best_tensor = torch.tensor([best_val_f1, best_val_acc], device=device)
        dist.broadcast(best_tensor, src=0)
        best_val_f1 = best_tensor[0].item()
        best_val_acc = best_tensor[1].item()

    return {
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    # ---- DDP setup ----
    local_rank, world_size, is_distributed = setup_ddp()

    set_global_seed(123456789)
    args = parse_arguments()
    output_dir = os.path.abspath(args.output_dir)

    # ---- Stable run directories (same across restarts) ----
    run_dirs = get_run_dirs(output_dir, args.run_name)

    if is_main_process():
        for d in [run_dirs['run_dir'], run_dirs['best_models'],
                  run_dirs['checkpoints'], run_dirs['logs']]:
            os.makedirs(d, exist_ok=True)

    if is_distributed:
        dist.barrier()

    logger, log_filename, _ = setup_logging(run_dirs['logs'])

    try:
        # ---- Load progress from previous runs ----
        progress = load_progress(run_dirs['progress_file'])
        completed_folds = progress['completed_folds']

        if is_main_process():
            logger.info(f"Arguments: {vars(args)}")
            logger.info(f"Run name: {args.run_name}")
            logger.info(f"Run directory: {run_dirs['run_dir']}")
            logger.info(f"Local rank: {local_rank}, World size: {world_size}")
            if completed_folds:
                logger.info(f"Previously completed folds: {list(completed_folds.keys())}")
                for fn, res in completed_folds.items():
                    logger.info(f"  {fn}: F1={res['best_val_f1']:.4f}")
            else:
                logger.info("Fresh run (no completed folds found)")

        # ---- Load data ----
        print_rank0("Loading training data...", logger)
        train_folds = load_training_data(args.data_path)

        print_rank0(f"Folds: {list(train_folds.keys())}", logger)

        # Update dataset params
        for fold_name in train_folds:
            for split in ['train_loader', 'val_loader']:
                ds = train_folds[fold_name][split].dataset
                ds.context_window = args.context_window
                ds.max_cache_size = args.night_cache_size

        # ---- Class weights ----
        global_class_weights = get_or_compute_class_weights(
            args.class_weights, train_folds, args.weight_method, logger
        )

        # ---- Train folds ----
        fold_results = dict(completed_folds)  # Start with previously completed folds
        total_folds = len(train_folds)
        overall_start = time.time()

        for fold_idx, (fold_name, fold_data) in enumerate(train_folds.items()):
            # Skip explicitly skipped folds
            if args.skip_folds and fold_name in args.skip_folds:
                print_rank0(f"Skipping {fold_name} (--skip_folds)", logger)
                continue

            # Skip already completed folds (auto-resume)
            if fold_name in completed_folds:
                print_rank0(f"Skipping {fold_name} (already completed, "
                           f"F1={completed_folds[fold_name]['best_val_f1']:.4f})", logger)
                continue

            torch.cuda.empty_cache()

            # Check for mid-fold checkpoint (auto-detected inside train_fold)
            checkpoint_path = find_checkpoint_for_fold(fold_name, run_dirs)
            if checkpoint_path:
                print_rank0(f"{fold_name}: Found checkpoint, will resume mid-fold", logger)
            else:
                print_rank0(f"{fold_name}: Starting fresh", logger)

            try:
                result = train_fold(
                    fold_name, fold_data, global_class_weights, args,
                    fold_idx, total_folds, run_dirs, local_rank, world_size,
                    is_distributed, logger
                )
                fold_results[fold_name] = result

                # Mark fold as completed in progress tracker
                if is_main_process():
                    progress['completed_folds'][fold_name] = {
                        'best_val_f1': result['best_val_f1'],
                        'best_val_acc': result['best_val_acc'],
                    }
                    save_progress(run_dirs['progress_file'], progress)
                    logger.info(f"\n{fold_name} COMPLETE: F1={result['best_val_f1']:.4f} "
                               f"(progress saved)")

                    # Clean up checkpoint for completed fold
                    cleanup_fold_checkpoint(fold_name, run_dirs)

            except Exception as e:
                if is_main_process():
                    logger.error(f"Failed {fold_name}: {e}", exc_info=True)
                    logger.info(f"Checkpoint preserved for {fold_name} — "
                               f"resubmit the same job to resume")
                continue

            # Progress report
            if is_main_process() and fold_results:
                n_done = len([k for k in fold_results if k in train_folds])
                avg_f1 = np.mean([r['best_val_f1'] for r in fold_results.values()])
                elapsed = time.time() - overall_start
                if n_done > len(completed_folds):
                    folds_this_run = n_done - len(completed_folds)
                    time_per_fold = elapsed / folds_this_run
                    remaining = (total_folds - n_done) * time_per_fold
                    logger.info(f"Progress: {n_done}/{total_folds} folds, Avg F1: {avg_f1:.4f}")
                    logger.info(f"Est. remaining: {remaining/3600:.1f} hours")

        # ---- Final summary ----
        if is_main_process() and fold_results:
            cv_f1 = [r['best_val_f1'] for r in fold_results.values()]
            cv_acc = [r['best_val_acc'] for r in fold_results.values()]
            total_time = time.time() - overall_start

            logger.info(f"\n{'='*60}")
            logger.info("FINAL CROSS-VALIDATION RESULTS")
            logger.info(f"{'='*60}")
            logger.info(f"Session time: {total_time/3600:.1f} hours")
            logger.info(f"Mean F1: {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")
            logger.info(f"Mean Acc: {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}")
            for fold_name, result in fold_results.items():
                logger.info(f"  {fold_name}: F1={result['best_val_f1']:.4f}, "
                           f"Acc={result['best_val_acc']:.4f}")
            logger.info(f"Best models: {run_dirs['best_models']}")

    except Exception as e:
        if is_main_process():
            logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if is_main_process():
            logger.info(f"Log: {log_filename}")
        cleanup_ddp()


if __name__ == "__main__":
    main()