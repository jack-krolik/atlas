# sleepdetector_optimized.py
# Optimized FlexibleSleepStageClassifier with GPU-native spectral features
# Changes from original:
#   1. SpectralFeatureExtractor rewritten to use torch.fft (stays on GPU, no CPU roundtrip)
#   2. All other architecture components unchanged for compatibility

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# Near the top of sleepdetector_optimized.py, after imports
try:
    from torch.amp import autocast
    def no_autocast(): return autocast('cuda', enabled=False)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast
    def no_autocast(): return autocast(enabled=False)

class PositionalEncoding(nn.Module):
    """Add positional encoding to embed temporal information"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TorchSpectralFeatureExtractor(nn.Module):
    """
    GPU-native spectral feature extractor using torch.fft.
    
    Pre-computes window, frequency axis, and band masks in __init__
    to avoid lazy initialization issues with torch.compile/CUDA graphs.
    """
    def __init__(self, sampling_rate=100, nperseg=1000, n_samples=3000):
        super().__init__()
        self.sampling_rate = sampling_rate

        # Frequency bands (Hz)
        self.band_ranges = [
            (0.5, 4.0),    # delta
            (4.0, 8.0),    # theta
            (8.0, 13.0),   # alpha
            (13.0, 30.0),  # beta
            (30.0, 40.0),  # low_gamma
            (12.0, 14.0),  # sigma (sleep spindles)
        ]

        # Pre-compute everything at init time
        nperseg = min(nperseg, n_samples)
        hop = nperseg // 2

        # Hann window
        window = torch.hann_window(nperseg)
        self.register_buffer('window', window)
        self.register_buffer('window_power', (window ** 2).sum().unsqueeze(0))

        # Frequency axis for rfft
        freqs = torch.fft.rfftfreq(nperseg, 1.0 / sampling_rate)

        # Pre-compute band masks
        band_masks = []
        for low, high in self.band_ranges:
            mask = (freqs >= low) & (freqs <= high)
            band_masks.append(mask)
        self.register_buffer('band_masks', torch.stack(band_masks))

        self._nperseg = nperseg
        self._hop = hop

    def forward(self, x):
        """
        Args:
            x: (batch, channels, samples) — raw signal on GPU
        Returns:
            features: (batch, channels * n_bands) — band power features
        """
        batch, channels, n_samples = x.shape
        nperseg = self._nperseg
        hop = self._hop

        # Pad if signal shorter than window
        if n_samples < nperseg:
            x = F.pad(x, (0, nperseg - n_samples))

        # Extract overlapping windows: (batch, channels, n_windows, nperseg)
        windows = x.unfold(dimension=2, size=nperseg, step=hop)

        # Apply Hann window
        windowed = windows * self.window

        # FFT
        fft_out = torch.fft.rfft(windowed, dim=-1)
        # PSD: |FFT|^2 / (fs * sum(window^2))
        psd = (fft_out.abs().pow(2)) / (self.sampling_rate * self.window_power)
        # Double non-DC, non-Nyquist bins
        psd[..., 1:-1] = psd[..., 1:-1] * 2.0

        # Average over windows -> (batch, channels, n_freqs)
        psd_avg = psd.mean(dim=2)

        # Extract band powers using pre-computed masks
        band_powers = torch.stack([
            psd_avg[:, :, mask].sum(dim=-1) for mask in self.band_masks
        ], dim=-1)

        # Numerical safety: log-scale band powers (prevents huge dynamic range)
        # and clamp to avoid NaN/inf propagation
        band_powers = torch.log1p(band_powers.clamp(min=0))
        band_powers = torch.nan_to_num(band_powers, nan=0.0, posinf=20.0, neginf=0.0)

        # Flatten to (batch, channels * n_bands)
        return band_powers.reshape(batch, -1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention for capturing relationships within epochs"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        residual = x

        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        output = self.w_o(context)
        return self.layer_norm(output + residual), attention_weights


class FlexibleIntraEpochCNN(nn.Module):
    """CNN that handles variable channels and epoch lengths with GPU spectral features"""
    def __init__(self, input_channels=4, cnn_features=256, normalize_per_epoch=True,
                 expected_length=3000, sampling_rate=100, use_spectral_features=True):
        super().__init__()
        self.normalize_per_epoch = normalize_per_epoch
        self.input_channels = input_channels
        self.expected_length = expected_length
        self.sampling_rate = sampling_rate
        self.use_spectral_features = use_spectral_features

        if use_spectral_features:
            self.spectral_extractor = TorchSpectralFeatureExtractor(sampling_rate)
            self.spectral_features_dim = input_channels * 6
        else:
            self.spectral_features_dim = 0

        self.kernel_1sec = int(sampling_rate)
        self.kernel_half_sec = int(sampling_rate // 2)
        self.kernel_quarter_sec = int(sampling_rate // 4)

        self.slow_branch = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=self.kernel_1sec,
                     padding=self.kernel_1sec//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=self.kernel_half_sec,
                     padding=self.kernel_half_sec//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.fast_branch = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=self.kernel_quarter_sec,
                     padding=self.kernel_quarter_sec//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=self.kernel_quarter_sec//2,
                     padding=self.kernel_quarter_sec//4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.spindle_branch = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=self.kernel_half_sec,
                     padding=self.kernel_half_sec//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=self.kernel_quarter_sec,
                     padding=self.kernel_quarter_sec//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        cnn_output_dim = 384 * 64

        if use_spectral_features:
            self.spectral_processor = nn.Sequential(
                nn.Linear(self.spectral_features_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            total_features = cnn_output_dim + 64
        else:
            total_features = cnn_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_features, cnn_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def per_epoch_per_channel_normalize(self, x):
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.std(x, dim=2, keepdim=True)
        eps = 1e-8
        return (x - mean) / (std + eps)

    def validate_input(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, channels, timepoints), got {x.dim()}D")
        return x

    def forward(self, x):
        x = self.validate_input(x)
        x_for_spectral = x  # Keep un-normalized copy for spectral features

        if self.normalize_per_epoch:
            x = self.per_epoch_per_channel_normalize(x)

        slow_features = self.slow_branch(x)
        fast_features = self.fast_branch(x)
        spindle_features = self.spindle_branch(x)

        slow_features = self.adaptive_pool(slow_features)
        fast_features = self.adaptive_pool(fast_features)
        spindle_features = self.adaptive_pool(spindle_features)

        cnn_combined = torch.cat([slow_features, fast_features, spindle_features], dim=1)
        cnn_flattened = cnn_combined.view(cnn_combined.size(0), -1)

        if self.use_spectral_features:
            # Force FP32 for FFT — FP16 overflows on raw EEG signals
            with no_autocast():
                spectral_features = self.spectral_extractor(x_for_spectral.float())
                processed_spectral = self.spectral_processor(spectral_features)
            combined_features = torch.cat([cnn_flattened, processed_spectral], dim=1)
        else:
            combined_features = cnn_flattened

        epoch_features = self.fusion(combined_features)
        return epoch_features


class CrossEpochAttention(nn.Module):
    """Cross-attention between current epoch and previous epochs."""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)

    def forward(self, current_epoch, previous_epochs):
        if previous_epochs is None:
            return current_epoch, None
        combined = torch.cat([previous_epochs, current_epoch], dim=1)
        attended, attention_weights = self.attention(combined)
        return attended, attention_weights


class FlexibleSleepStageClassifier(nn.Module):
    """Complete flexible architecture for variable channels and epoch lengths."""
    def __init__(self,
             input_channels=4,
             cnn_features=256,
             lstm_hidden=128,
             lstm_layers=2,
             attention_heads=8,
             num_classes=5,
             max_history=32,
             normalize_per_epoch=True,
             sampling_rate=100,
             epoch_duration=30,
             use_spectral_features=True,
             use_attention=True):
        super().__init__()

        self.input_channels = input_channels
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.expected_length = sampling_rate * epoch_duration
        self.use_spectral_features = use_spectral_features
        self.lstm_hidden = lstm_hidden

        self.intra_epoch_cnn = FlexibleIntraEpochCNN(
            input_channels, cnn_features, normalize_per_epoch,
            self.expected_length, sampling_rate, use_spectral_features
        )
        self.positional_encoding = PositionalEncoding(cnn_features)
        self.use_attention = use_attention
        if use_attention:
            self.cross_epoch_attention = CrossEpochAttention(cnn_features, attention_heads)

        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )

        self.lstm_layer_norm = nn.LayerNorm(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.max_history = max_history

    def forward(self, current_epoch, previous_epochs=None, return_attention=False):
        batch_size = current_epoch.size(0)

        if current_epoch.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {current_epoch.size(1)}")

        current_features = self.intra_epoch_cnn(current_epoch)
        current_features = current_features.unsqueeze(1)

        cross_attention = None

        if previous_epochs is not None:
            batch_size_prev, seq_len, channels, timepoints = previous_epochs.shape

            if channels != self.input_channels:
                raise ValueError(f"Previous epochs: expected {self.input_channels} channels, got {channels}")

            prev_reshaped = previous_epochs.view(-1, channels, timepoints)
            prev_features = self.intra_epoch_cnn(prev_reshaped)
            prev_features = prev_features.view(batch_size, seq_len, -1)

            prev_features = self.positional_encoding(prev_features)

            if self.use_attention:
                attended_sequence, cross_attention = self.cross_epoch_attention(
                    current_features, prev_features
                )
            else:
                attended_sequence = torch.cat([prev_features, current_features], dim=1)

            lstm_out, (h_n, c_n) = self.lstm(attended_sequence)
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            final_hidden = self.lstm_layer_norm(final_hidden)
        else:
            lstm_out, (h_n, c_n) = self.lstm(current_features)
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            final_hidden = self.lstm_layer_norm(final_hidden)

        logits = self.classifier(final_hidden)

        if return_attention:
            return logits, {
                'cross_attention': cross_attention,
                'lstm_output': lstm_out if previous_epochs is not None else None
            }
        return logits

    def get_attention_weights_for_epoch(self, current_epoch, previous_epochs):
        _, attention_dict = self.forward(current_epoch, previous_epochs, return_attention=True)
        if attention_dict['cross_attention'] is None:
            return None
        cross_attn = attention_dict['cross_attention']
        avg_attention = cross_attn.mean(dim=1)
        current_to_context = avg_attention[:, -1, :-1]
        return current_to_context


# ==================== Factory functions ====================

def create_flexible_model(input_channels=4, sampling_rate=100, epoch_duration=30,
                         normalize_per_epoch=True, num_classes=5,
                         use_spectral_features=True, use_attention=True):
    return FlexibleSleepStageClassifier(
        input_channels=input_channels,
        cnn_features=256,
        lstm_hidden=128,
        lstm_layers=2,
        attention_heads=8,
        num_classes=num_classes,
        max_history=32,
        normalize_per_epoch=normalize_per_epoch,
        sampling_rate=sampling_rate,
        epoch_duration=epoch_duration,
        use_spectral_features=use_spectral_features,
        use_attention=use_attention
    )


def create_model(normalize_per_epoch=True, use_spectral_features=True):
    return create_flexible_model(
        input_channels=4,
        sampling_rate=100,
        epoch_duration=30,
        normalize_per_epoch=normalize_per_epoch,
        use_spectral_features=use_spectral_features
    )