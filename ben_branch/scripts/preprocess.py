#!/usr/bin/env python3
"""
Preprocess UrbanSound8K Arrow files into .wav using config.yaml
"""

import yaml
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torchaudio.functional as F


# locate project root & load config
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yml") as f:
    cfg = yaml.safe_load(f)

# Get configs
INPUT_ARROW = ROOT / cfg["paths"]["arrow_train"]
OUTPUT_DIR   = ROOT / cfg["paths"]["wav_output"]
TARGET_SR    = cfg["audio"]["target_sampling_rate"]
NUM_WORKERS  = cfg["preprocessing"]["num_workers"]


def find_max_duration(ds: Dataset) -> float:
  """
  Scan the dataset once to find the maximum clip duration (in seconds).
  """
  max_dur = 0.0
  for ex in tqdm(ds, desc="⏱  Finding max duration"):
    arr = ex["audio"]["array"]
    sr = ex["audio"]["sampling_rate"]
    dur = arr.shape[0] / sr
    if dur > max_dur:
      max_dur = dur
  return max_dur


def preprocess_and_save(ds: Dataset, max_samples: int) -> None:
  """
  For each example in the dataset:
    - Downsample to TARGET_SR
    - Zero-pad or truncate to max_samples
    - Write out a .wav named by ex["slice_file_name"]
  """
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  for ex in tqdm(ds, desc="🎛  Preprocessing & saving"):
    arr = ex["audio"]["array"]
    sr = ex["audio"]["sampling_rate"]
    tensor = torch.from_numpy(arr)

    # resample if needed
    if sr != TARGET_SR:
      tensor = F.resample(tensor, orig_freq=sr, new_freq=TARGET_SR)

    wav = tensor.numpy()

    # pad or truncate
    length = wav.shape[0]
    if length < max_samples:
      wav = np.pad(wav, (0, max_samples - length))
    else:
      wav = wav[:max_samples]

    # build filename and write
    fname = ex["slice_file_name"]  # e.g. "100263-2-0-121.wav"
    out_path = OUTPUT_DIR / fname
    sf.write(str(out_path), wav, TARGET_SR)


def main() -> None:
  """
  1) Load Arrow dataset
  2) Find max duration
  3) Preprocess & save to .wav
  """
  print("📥 Loading dataset from", INPUT_ARROW)
  ds = load_from_disk(str(INPUT_ARROW))  # returns a Dataset for that split

  # 1️⃣ find max duration (secs) → convert to samples @16 kHz
  max_dur = find_max_duration(ds)
  max_samples = int(np.ceil(max_dur * TARGET_SR))
  print(f"🔎 Max duration = {max_dur:.3f}s → {max_samples} samples at {TARGET_SR}Hz")

  # 2️⃣ preprocess & dump .wav files
  preprocess_and_save(ds, max_samples)
  print("✅ All done — .wav files in", OUTPUT_DIR)


if __name__ == "__main__":
  main()
