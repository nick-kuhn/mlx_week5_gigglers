from pathlib import Path
from datasets import Dataset
from transformers import WhisperProcessor
import torch
import torchaudio
import random


class AudioDataset(Dataset):
  """
  Simple dataset that reads .wav files and extracts labels from filenames.
  Assumes files named like someaudio_classX.wav where X is 1..NUM_CLASSES.
  """
  def __init__(self, files_dir: Path, processor: WhisperProcessor, label_map: dict):
    self.files      = list(files_dir.glob("*.[wW][aA][vV]"))
    self.processor  = processor
    self.label_map  = label_map

  def __len__(self) -> int:
    return len(self.files)

  def __getitem__(self, idx: int) -> dict:
    file_path = self.files[idx]
    # load audio
    speech_array, sampling_rate = torchaudio.load(file_path)
    speech = speech_array.squeeze().numpy()
    # extract class from filename
    cls_str = file_path.stem.split("_")[2]
    label   = self.label_map[cls_str]
    # preprocess to log-Mel features
    inputs = self.processor(speech, sampling_rate=sampling_rate, return_tensors="pt")
    return {"input_features": inputs.input_features.squeeze(0), "label": label}

class DummyDataset(Dataset):
  """
  FOR DEBUGGING ONLY!
  Dummy dataset that returns random log-Mel tensors and random labels for debugging.
  """
  def __init__(self, processor: WhisperProcessor, n_items: int = 1000):
    self.processor = processor
    self.n_items = n_items
    # generate a blank log-Mel feature from 1s of white noise
    dummy_audio = torch.randn(16000)
    self.blank = processor(dummy_audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)

  def __len__(self) -> int:
    return self.n_items

  def __getitem__(self, idx: int) -> dict:
    fake_feats = self.blank + 0.01 * torch.randn_like(self.blank)
    fake_label = random.randrange(NUM_CLASSES)
    return {"input_features": fake_feats, "label": fake_label}
