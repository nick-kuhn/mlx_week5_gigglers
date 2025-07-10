import sys
import random
from pathlib import Path
import pandas as pd
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperModel
import torchaudio

# Add the project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import AudioDataset, DummyDataset
from model.model import WhisperEncoderClassifier
from model.download_model import download_model

# Global constants
NUM_CLASSES = 10  # number of target classes
BATCH_SIZE = 16
EPOCHS = 10       # ← default number of epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base folder for the script (so paths work anywhere)
# scratch is one level under the whisper directory, so go up two levels to reach whisper/
BASE_DIR = Path(__file__).resolve().parent.parent

# directories relative to the whisper folder
DATA_DIR = BASE_DIR / "data" / "generated_audio"
MODEL_DIR = BASE_DIR / "whisper_models" / "whisper_tiny"

DUMMY_ITEMS = 1000

print(f"Using device: {DEVICE}, CUDA version: {torch.version.cuda}, GPU count: {torch.cuda.device_count()}")

import os
print(f'Found {len(os.listdir(DATA_DIR))} audio files')  # Ensure the data directory exists



def train(data_dir: str, model_dir: str, use_dummy: bool = False, epochs: int = EPOCHS):
  # Discover all classes by filename
  data_file = data_dir.parent.parent / "audio_dataset.csv"
  df = pd.read_csv(data_file)
  classes     = sorted(df["class_label"].unique())
  print(f"Found {len(classes)} classes:", classes)
  label_to_idx = {cls: i for i, cls in enumerate(classes)}


  #check if model_dir exists
  if not os.path.exists(model_dir):
    print(f"Model directory {model_dir} does not exist. Download model? (y/n)")
    if input() == "y":
      download_model(model_dir)
    else:
      print("Exiting...")
      return

  # prepare processor and dataset
  processor = WhisperProcessor.from_pretrained(model_dir)
  if use_dummy:
    # use DummyDataset for debugging without real files
    dataset = DummyDataset(processor, n_items=DUMMY_ITEMS, num_classes=len(classes))
    print("⚙️ Using dummy dataset with random data")
  else:
    dataset = AudioDataset(data_file, processor, label_to_idx)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

  # build model, loss, optimizer
  model = WhisperEncoderClassifier(model_dir, num_classes=len(classes)).to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  # one epoch example
  model.train() #TODO are freezing or not currently?
  correct, seen = 0, 0
  for epoch in range(1, epochs + 1):
    print(f"\n▶️  Epoch {epoch}/{epochs}")
    model.train()  # ensure training mode

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
      #TODO print true class distribution vs pred class distribution
      true_dist = labels.bincount(minlength=NUM_CLASSES)
      pred_dist = preds.bincount(minlength=NUM_CLASSES)
      print(f"True class distribution: {true_dist}")
      print(f"Pred class distribution: {pred_dist}")

def main():
  # detect --dummy flag for debug
  use_dummy = "--dummy" in sys.argv
  train(DATA_DIR, MODEL_DIR, use_dummy)

if __name__ == "__main__":
  main()
