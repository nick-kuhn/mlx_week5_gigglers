from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import WhisperProcessor
import torch
import torchaudio
import random
import pandas as pd
from typing import Optional


def load_from_hf_voice_commands():
    ds = load_dataset("ntkuhn/mlx_voice_commands_mixed", split="train")   # 1. pull dataset
    df = ds.to_pandas()                                                   # 2. convert to DataFrame
    df["audio_path"]  = df["audio"].apply(lambda x: x["path"])            # 3. extract cached wav path
    return df[["audio_path", "class_label"]]

class AudioDataset(Dataset):
  """
  Simple dataset that reads .wav files and extracts labels from filenames.
  Assumes files named like someaudio_classX.wav where X is 1..NUM_CLASSES.
  """
  def __init__(self, data_file: Path, processor: WhisperProcessor, label_map: dict, max_len: Optional[int] = None, from_hf: bool = False):

    if data_file is None:
        data_file = Path(__file__).resolve().parent.parent / "audio_dataset.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found")

    # load data
    
    if from_hf:
      self.data = load_from_hf_voice_commands()
    else:
      self.data = pd.read_csv(data_file)

    if max_len is not None:
      self.data = self.data.iloc[:max_len]  # limit to max_len rows
    self.filenames  = self.data["audio_path"]
    self.labels     = self.data["class_label"]
    self.processor  = processor
    self.label_map  = label_map

    self.unique_labels = self.labels.unique()
    self.num_classes   = len(self.unique_labels)
    # validate audio files unless from_hf
    if not from_hf:
      self._validate_audio_files()

  def __len__(self) -> int:
    return len(self.filenames)

  def __getitem__(self, idx: int) -> dict:
    file_path = self.filenames[idx]
    label     = self.labels[idx]
    # Convert string label to integer index
    label_idx = self.label_map[label]
    # load audio
    speech_array, sampling_rate = torchaudio.load(file_path)
    speech = speech_array.squeeze().numpy()
    # preprocess to log-Mel features
    inputs = self.processor(speech, sampling_rate=sampling_rate, return_tensors="pt")
    return {"input_features": inputs.input_features.squeeze(0), "label": torch.tensor(label_idx, dtype=torch.long)}

  def _validate_audio_files(self):
    """Check that all audio files in the dataset exist."""
    missing_files = []
    
    for idx, row in self.data.iterrows():
        file_path = Path(row['audio_path'])
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Found {len(missing_files)} missing audio files:")
        for file in missing_files[:5]:  # Show first 5
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        
        raise FileNotFoundError(
            f"Dataset validation failed: {len(missing_files)} audio files are missing. "
            f"Please check your audio_dataset.csv and ensure all files exist."
        )
    
    print(f"✅ All {len(self.data)} audio files exist")

class DummyDataset(Dataset):
  """
  FOR DEBUGGING ONLY!
  Dummy dataset that returns random log-Mel tensors and random labels for debugging.
  """
  def __init__(self, processor: WhisperProcessor, n_items: int = 1000, num_classes: int = 10):
    self.processor = processor
    self.n_items = n_items
    self.num_classes = num_classes
    # generate a blank log-Mel feature from 1s of white noise
    dummy_audio = torch.randn(16000)
    self.blank = processor(dummy_audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)

  def __len__(self) -> int:
    return self.n_items

  def __getitem__(self, idx: int) -> dict:
    fake_feats = self.blank + 0.01 * torch.randn_like(self.blank)
    fake_label = random.randrange(self.num_classes)
    return {"input_features": fake_feats, "label": torch.tensor(fake_label, dtype=torch.long)}
