import sys
import random
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperModel
import torchaudio

# Global constants
NUM_CLASSES = 10  # number of target classes
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base folder for the script (so paths work anywhere)
# scratch is one level under the whisper directory, so go up two levels to reach whisper/
BASE_DIR = Path(__file__).resolve().parent.parent

# directories relative to the whisper folder
DATA_DIR = BASE_DIR.parent / "data" / "generated_audio"
MODEL_DIR = BASE_DIR / "models" / "whisper_tiny"

DUMMY_ITEMS = 1000

print(f"Using device: {DEVICE}, CUDA version: {torch.version.cuda}, GPU count: {torch.cuda.device_count()}")

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
  Dummy dataset that returns random log-Mel tensors and random labels for debugging.
  """
  def __init__(self, processor: WhisperProcessor, n_items: int = DUMMY_ITEMS):
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

class WhisperEncoderClassifier(nn.Module):
  """
  Classification model using Whisper's encoder as a feature extractor
  plus a simple linear head.
  """
  def __init__(self, model_dir: str, num_classes: int = NUM_CLASSES):
    super().__init__()
    # load pretrained encoder
    self.whisper = WhisperModel.from_pretrained(model_dir).encoder
    hidden_size = self.whisper.config.d_model
    # classification head
    self.classifier = nn.Linear(hidden_size, num_classes)

  def forward(self, input_features: torch.Tensor) -> torch.Tensor:
    """
    We input a single mel spec for each audio file via input_features: (batch, seq_len, feature_dim)
    the model will split the audio up into timeframes internally and output a hidden state for each
    timeframe. So encoder_outputs will have multiple embeddings per input audio sample.
    """
    # Get the encoder outputs, will split audio into timesteps internally
    encoder_outputs = self.whisper(input_features.to(DEVICE))
    # Get the last hidden state only for each time step
    last_hidden = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_size)
    # Mean pool over time steps
    pooled = last_hidden.mean(dim=1)
    # Run linear classifier
    return self.classifier(pooled)


def train(data_dir: str, model_dir: str, use_dummy: bool = False):
  # Discover all classes by filename
  audio_files = list(Path(data_dir).glob("*.[wW][aA][vV]"))
  classes     = sorted({f.stem.split("_")[2] for f in audio_files})
  print(f"Found {len(classes)} classes:", classes)
  label_to_idx = {cls: i for i, cls in enumerate(classes)}


  # prepare processor and dataset
  processor = WhisperProcessor.from_pretrained(model_dir)
  if use_dummy:
    # use DummyDataset for debugging without real files
    dataset = DummyDataset(processor)
    print("⚙️ Using dummy dataset with random data")
  else:
    dataset = AudioDataset(Path(data_dir), processor, label_to_idx)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

  # build model, loss, optimizer
  model = WhisperEncoderClassifier(model_dir, num_classes=len(classes)).to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  # one epoch example
  model.train()
  correct, seen = 0, 0
  for batch in loader:
    inputs  = batch["input_features"].to(DEVICE)
    labels  = batch["label"].to(DEVICE)

    logits  = model(inputs)
    loss    = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute batch accuracy
    preds    = logits.argmax(dim=1)
    correct += (preds == labels).sum().item()
    seen    += labels.size(0)
    acc      = correct / seen

    print(f"Batch loss: {loss.item():.4f}   |   running acc: {acc*100:.1f}%")


def main():
  # detect --dummy flag for debug
  use_dummy = "--dummy" in sys.argv
  train(DATA_DIR, MODEL_DIR, use_dummy)

if __name__ == "__main__":
  main()
