from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
import torch
import torchaudio
import random
import pandas as pd
from typing import Optional
import io
import numpy as np

# The 'soundfile' library is used as the audio backend for HuggingFace datasets,
# which is configured in the main training script.
import soundfile as sf

# Import our augmentation utilities
from model.augmentations import create_augmentation_pipeline


class AudioDataset(Dataset):
  """
  Simple dataset that reads .wav files and extracts labels from filenames.
  Assumes files named like someaudio_classX.wav where X is 1..NUM_CLASSES.
  """
  def __init__(self, data_file: Path, processor: WhisperProcessor, label_map: dict, max_len: Optional[int] = None):

    if data_file is None:
        data_file = Path(__file__).resolve().parent.parent / "audio_dataset.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found")

    # load data
    
    self.data = pd.read_csv(data_file)

    if max_len is not None:
      self.data = self.data.iloc[:max_len]  # limit to max_len rows
    self.filenames  = self.data["audio_path"]
    self.labels     = self.data["class_label"]
    self.processor  = processor
    self.label_map  = label_map

    self.unique_labels = self.labels.unique()
    self.num_classes   = len(self.unique_labels)
    # validate audio files
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



class AudioHFDataset(Dataset):
  """
  Dataset that loads audio files from the HF dataset.
  This version avoids automatic decoding by HuggingFace and handles it manually
  to bypass issues with the torchcodec backend on Windows.
  """
  def __init__(self, processor: WhisperProcessor, label_map: dict, max_len: Optional[int] = None, ds: Optional[Dataset] = None, 
               apply_augmentations: bool = False, augmentation_config: Optional[dict] = None):
    
    # If a dataset is not passed in, load it from HuggingFace.
    # Otherwise, use the one that was provided.
    if ds is None:
        split = f"train[:{max_len}]" if max_len is not None else "train"
        ds = load_dataset("ntkuhn/mlx_voice_commands_mixed", split=split)
    
    # Crucial fix: Re-cast the dataset to disable decoding on the audio feature.
    # This creates a new dataset object with the desired (immutable) features.
    new_features = ds.features.copy()
    new_features['audio'] = Audio(sampling_rate=16000, decode=False)
    self.ds = ds.cast(new_features)
    
    self.processor = processor
    self.label_map = label_map
    self.num_classes = len(label_map)
    
    # Initialize augmentations
    self.apply_augmentations = apply_augmentations
    self.augmentation_pipeline = None
    if self.apply_augmentations:
        self.augmentation_pipeline = create_augmentation_pipeline(augmentation_config, sample_rate=16000)
        print(f"🔄 Augmentations enabled for recording samples")
    
    # Count and report recording vs generated samples
    self._analyze_dataset_composition()

  def _analyze_dataset_composition(self):
    """Analyze the dataset composition to count recording vs generated samples."""
    try:
        # Count by type if the field exists
        if 'type' in self.ds.features:
            recording_count = len([item for item in self.ds if item.get('type') == 'recording'])
            generated_count = len([item for item in self.ds if item.get('type') == 'generated'])
            print(f"📊 Dataset composition: {recording_count} recording samples, {generated_count} generated samples")
        else:
            print("⚠️  No 'type' field found in dataset - augmentations will be applied to all samples")
    except Exception as e:
        print(f"⚠️  Could not analyze dataset composition: {e}")

  def __len__(self) -> int:
    return len(self.ds)

  def __getitem__(self, idx: int) -> dict:
    item = self.ds[idx]
    
    # Manually decode the audio file from the in-memory bytes.
    # This is the most robust method, avoiding filesystem path issues.
    audio_data = item["audio"]
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data["bytes"]))

    # Ensure audio is 16kHz as required by Whisper
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Get the numpy array for processing
    waveform_np = waveform.squeeze().numpy()
    
    # Apply augmentations only to "recording" type samples
    if self.apply_augmentations and self.augmentation_pipeline is not None:
        sample_type = item.get('type', 'unknown')
        if sample_type == 'recording':
            # Apply waveform augmentations
            waveform_np = self.augmentation_pipeline.apply_waveform_augmentations(waveform_np)

    label = item["class_label"]
    
    # Convert string label to integer index
    label_idx = self.label_map[label]
    
    # Preprocess to log-Mel features
    inputs = self.processor(waveform_np, sampling_rate=16000, return_tensors="pt")
    
    return {
        "input_features": inputs.input_features.squeeze(0), 
        "label": torch.tensor(label_idx, dtype=torch.long),
        "sample_type": item.get('type', 'unknown')  # Include type for debugging
    }

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
