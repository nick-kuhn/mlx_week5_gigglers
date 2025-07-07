#!/usr/bin/env python3
"""
dataloader.py

Defines UrbanSoundDataset and DataLoader factory. Expects your preprocessed
.-wav files organized under WAV_DIR/fold{fold}/ as:

  ben_branch/data/wav_files/
    ├─ fold1/
    ├─ fold2/
    └─ … 
Each filename must follow: fsID-classID-occurrenceID-sliceID.wav
"""

import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from spectrogram import compute_log_mel_spectrogram
from augmentations import (
  time_shift, add_noise, pitch_shift, time_stretch,
  freq_mask, time_mask
)
import random
import torch.nn.functional as F
from typing import List, Tuple

# ─── Load configuration ───────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
cfg        = yaml.safe_load(open(ROOT / "config.yml"))
PATHS      = cfg["paths"]
AUDIO      = cfg["audio"]
PREPROC    = cfg["preprocessing"]
TRAIN_CFG  = cfg["training"]["folds"]
AUG_WF     = cfg["augmentations"]["waveform"]
AUG_SP     = cfg["augmentations"]["spec_masking"]
MIX_CFG    = AUG_WF.get("mixup", {})
NUM_CLASSES = cfg.get("model", {}).get("num_classes", 10)

# ─── Constants ────────────────────────────────────────────────────────────────
SR        = AUDIO["target_sampling_rate"]             # target sample rate
WAV_DIR   = ROOT / PATHS["wav_output"]                # where foldX/ subdirs live
FIXED_DUR      = AUDIO["fixed_duration"]
TARGET_SAMPLES = int(FIXED_DUR * SR)
MIX_ALPHA = MIX_CFG.get("alpha", 0.2)                 # mixup Beta dist α
MIX_PROB  = MIX_CFG.get("prob", 0.0)                  # mixup on/off probability

# pull in n_fft/hop_length from the spectrogram config:
spec_cfg   = cfg["spectrogram"]
N_FFT      = spec_cfg["n_fft"]
HOP_LENGTH = spec_cfg["hop_length"]

# formula for frames: 1 + floor((samples - n_fft) / hop_length)
MAX_FRAMES = 1 + (TARGET_SAMPLES - N_FFT) // HOP_LENGTH

# Get the arrow metadata for fold mapping
from datasets import load_from_disk # keep here, don't move to top!

# after your constants…
ARROW_DIR = ROOT / PATHS["arrow_train"]
_arrow_ds = load_from_disk(str(ARROW_DIR))
FOLD_MAP  = {
  ex["slice_file_name"]: ex["fold"]
  for ex in _arrow_ds
}


# ─── Dataset ──────────────────────────────────────────────────────────────────
class UrbanSoundDataset(Dataset):
  """Dataset for UrbanSound8K .wav files with in-memory augmentations."""
  
  def __init__(self, folds: List[int], train: bool = True) -> None:
    """
    Args:
      folds: list of fold numbers to include (e.g. [1,2,...,9])
      train: whether to apply augmentations
    """
    self.train = train
    self.samples = []

    # flat scan of WAV_DIR
    for wav_path in WAV_DIR.glob("*.wav"):
        fname = wav_path.name
        fold = FOLD_MAP.get(fname)
        if fold in folds:
            class_id = int(fname.split("-")[1])
            self.samples.append((wav_path, class_id))


  def __len__(self) -> int:
    """Total number of clips in this dataset."""
    return len(self.samples)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor,int]:
    """
    Load one example, apply waveform & spectrogram augmentations,
    and return (spec, label).
    """
    wav_path, label = self.samples[idx]
    
    # load waveform as float32
    waveform, sr = sf.read(wav_path, dtype="float32")
    if sr != SR:
      raise ValueError(f"Expected SR={SR}, got {sr}")
    
    # ── waveform-level augmentations ────────────────────────────────
    if self.train:
      waveform = time_shift(waveform, SR)
      waveform = add_noise(waveform)
      waveform = pitch_shift(waveform, SR)
      waveform = time_stretch(waveform, SR)
    
    # ── compute log-Mel spectrogram ─────────────────────────────────
    spec_np = compute_log_mel_spectrogram(waveform, SR)

    # ── pad/truncate time axis to fixed MAX_FRAMES ───────────────
    frames = spec_np.shape[1]
    if frames < MAX_FRAMES:
        pad = MAX_FRAMES - frames
        spec_np = np.pad(spec_np, ((0,0),(0,pad)), mode="constant")
    else:
        spec_np = spec_np[:, :MAX_FRAMES]

    
    # ── to tensor (C × H × W) and float32 ───────────────────────────
    spec = torch.from_numpy(spec_np).unsqueeze(0).float()
    
    # ── spectrogram-level augmentations ─────────────────────────────
    if self.train:
      spec = freq_mask(spec)
      spec = time_mask(spec)

    return spec, label

# ─── Collate + MixUp ──────────────────────────────────────────────────────────
def collate_fn(
    batch: List[Tuple[torch.Tensor,int]]
) -> Tuple[torch.Tensor,torch.Tensor]:
  """
  Batch specs & labels, optionally apply MixUp and return
  (batch_specs, batch_labels_or_soft).
  """
  specs, labels = zip(*batch)
  specs_tensor = torch.stack(specs, 0)       # shape: (B, 1, n_mels, T)
  labels_tensor = torch.tensor(labels)       # shape: (B,)

  # apply MixUp if configured
  if MIX_PROB > 0 and random.random() < MIX_PROB:
    idx = torch.randperm(specs_tensor.size(0))
    lam = np.random.beta(MIX_ALPHA, MIX_ALPHA)
    specs_tensor = lam * specs_tensor + (1 - lam) * specs_tensor[idx]
    y1 = F.one_hot(labels_tensor, NUM_CLASSES).float()
    y2 = y1[idx]
    labels_tensor = lam * y1 + (1 - lam) * y2  # soft labels

  return specs_tensor, labels_tensor

# ─── DataLoader Factory ─────────────────────────────────────────────────────
def get_dataloader(
    split: str = "train",
    batch_size: int = 16
) -> DataLoader:
  """
  Returns a DataLoader for the given split ("train" or "val").
  Uses folds from config.yml and shuffles only on train.
  """
  folds = TRAIN_CFG["train"] if split == "train" else TRAIN_CFG["val"]
  dataset = UrbanSoundDataset(folds, train=(split == "train"))
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=(split == "train"),
    num_workers=PREPROC["num_workers"],
    collate_fn=collate_fn,
    pin_memory=True,
  )
