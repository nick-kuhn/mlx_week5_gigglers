#!/usr/bin/env python3
"""
augmentations.py

Waveform and spectrogram augmentation utilities using config.yaml.
"""
import yaml
from pathlib import Path
import numpy as np
import random
import torch
import librosa

ROOT = Path(__file__).resolve().parent.parent
cfg  = yaml.safe_load(open(ROOT / "config.yml"))
defs = cfg["defaults"]

# pull augmentations from defaults
wf_cfg = defs["augmentations"]["waveform"]
sp_cfg = defs["augmentations"]["spec_masking"]


# waveform augs
def time_shift(wav: np.ndarray, sr: int):
    max_shift=int(wf_cfg["time_shift"]["shift_max"]*sr)
    return np.roll(wav, random.randint(-max_shift, max_shift))

def add_noise(wav: np.ndarray):
    lvl=wf_cfg["add_noise"]["noise_level"]
    return wav + np.random.randn(len(wav))*lvl

def pitch_shift(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Randomly shift pitch by a few semitones.
    Uses keyword args to avoid any signature mismatch.
    """
    low, high = wf_cfg["pitch_shift"]["n_steps"]
    n_steps    = random.uniform(low, high)
    # explicitly name the arguments so we call the right function
    return librosa.effects.pitch_shift(
        y=wav,
        sr=sr,
        n_steps=n_steps,
        # you can add bins_per_octave or res_type here if you like
    )

def time_stretch(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Randomly speed up or slow down audio.
    Accepts waveform and sample rate for signature consistency.
    """
    low, high = wf_cfg["time_stretch"]["rate"]
    rate      = random.uniform(low, high)
    return librosa.effects.time_stretch(y=wav, rate=rate)

def mixup(w1: np.ndarray,w2: np.ndarray):
    a=wf_cfg["mixup"]["alpha"]
    lam=np.random.beta(a,a)
    return lam*w1+(1-lam)*w2, lam

# spec augs
def freq_mask(spec: torch.Tensor) -> torch.Tensor:
    """
    Apply frequency masking to a spectrogram tensor.
    Assumes spec shape [C, n_mels, n_frames].
    """
    cfg_f = sp_cfg["freq"]
    # number of Mel bands is dim 1
    num_mels = spec.size(1)
    masked = spec.clone()
    for _ in range(cfg_f["num_masks"]):
        f = random.randint(0, cfg_f["F_param"])
        if f == 0:
            continue
        f0 = random.randint(0, num_mels - f)
        # mask all channels and time frames in that band range
        masked[:, f0 : f0 + f, :] = 0
    return masked

def time_mask(spec: torch.Tensor) -> torch.Tensor:
    """
    Apply time masking to a spectrogram tensor.
    Assumes spec shape [C, n_mels, n_frames].
    """
    cfg_t = sp_cfg["time"]
    # number of time frames is dim 2
    num_frames = spec.size(2)
    masked = spec.clone()
    for _ in range(cfg_t["num_masks"]):
        t = random.randint(0, cfg_t["T_param"])
        if t == 0:
            continue
        t0 = random.randint(0, num_frames - t)
        # mask all channels and mel bands in that time range
        masked[:, :, t0 : t0 + t] = 0
    return masked