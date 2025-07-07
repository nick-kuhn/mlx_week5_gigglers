#!/usr/bin/env python3
"""
spectrogram.py

Compute log-Mel spectrograms using parameters from config.yaml.
"""

import yaml
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torch.nn as nn
# patch deprecated NumPy aliases for librosa compatibility
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "float"):   np.float = float
import librosa

# load config
dotenv = Path(__file__).resolve().parent.parent / "config.yml"
with open(dotenv) as f:
    cfg = yaml.safe_load(f)

# spectrogram settings live under defaults
defs     = cfg["defaults"]
spec_cfg = defs["spectrogram"]
audio_cfg = defs["audio"]

# spectrogram defaults from config
N_FFT      = spec_cfg["n_fft"]
HOP_LENGTH = spec_cfg["hop_length"]
N_MELS     = spec_cfg["n_mels"]
FMIN       = spec_cfg["fmin"]
FMAX       = spec_cfg["fmax"]
SAMPLE_RATE = audio_cfg["target_sampling_rate"]

# Faster GPU version using torchaudio
class MelSpecGPU(nn.Module):
    def __init__(self):
        super().__init__()
        # pull defaults from cfg["defaults"]
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: Tensor [batch, time]
        returns: Tensor [batch, n_mels, frames]
        """
        spec = self.melspec(waveform)
        log_spec = self.to_db(spec)
        return log_spec

# Slower CPU version using librosa
def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = None,
    hop_length: int = None,
    n_mels: int = None,
    fmin: int = None,
    fmax: int = None,
) -> np.ndarray:
    """
    Compute a log-scaled Mel spectrogram.
    Returns array of shape (n_mels, time_frames).
    """
    n_fft      = n_fft or N_FFT
    hop_length = hop_length or HOP_LENGTH
    n_mels     = n_mels or N_MELS
    fmin       = fmin or FMIN
    fmax       = fmax or FMAX

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel
