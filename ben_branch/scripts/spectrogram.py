#!/usr/bin/env python3
"""
spectrogram.py

Compute log-Mel spectrograms using parameters from config.yaml.
"""

import yaml
from pathlib import Path
import numpy as np
# patch deprecated NumPy aliases for librosa compatibility
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "float"):   np.float = float
import librosa

# load config
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yml") as f:
    cfg = yaml.safe_load(f)
spec_cfg = cfg["spectrogram"]

# spectrogram defaults from config
N_FFT = spec_cfg["n_fft"]
HOP_LENGTH = spec_cfg["hop_length"]
N_MELS = spec_cfg["n_mels"]
FMIN = spec_cfg["fmin"]
FMAX = spec_cfg["fmax"]


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
    # use config defaults if no override
    n_fft      = n_fft      or N_FFT
    hop_length = hop_length or HOP_LENGTH
    n_mels     = n_mels     or N_MELS
    fmin       = fmin       or FMIN
    fmax       = fmax       or FMAX

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
