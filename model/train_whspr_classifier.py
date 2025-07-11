# Set the audio backend for HuggingFace datasets BEFORE any other imports
# to prevent it from defaulting to torchcodec. This is crucial on systems
# (like Windows) where torchcodec's dependencies might be missing.
import soundfile as sf
import datasets
datasets.config.AUDIO_DECODE_BACKEND = "soundfile"

import sys
import random
from pathlib import Path
import pandas as pd
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import WhisperProcessor
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import wandb
import os
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Add the project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import AudioDataset, DummyDataset, AudioHFDataset
from model.model import WhisperEncoderClassifier

# Global constants
NUM_CLASSES = 10  # number of target classes
BATCH_SIZE = 128
EPOCHS = 10       # ← default number of epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base folder for the script (so paths work anywhere)
# scratch is one level under the whisper directory, so go up two levels to reach whisper/
BASE_DIR = Path(__file__).resolve().parent.parent

# directories relative to the whisper folder
DATA_DIR = BASE_DIR / "data" / "generated_audio"
MODEL_DIR = BASE_DIR / "whisper_models" / "whisper_tiny"
VALIDATION_DIR = BASE_DIR / "data" / "validation"
VALIDATION_CSV = VALIDATION_DIR / "recordings.csv"

DUMMY_ITEMS = 1000

print(f"Using device: {DEVICE}, CUDA version: {torch.version.cuda}, GPU count: {torch.cuda.device_count()}")

import os

try:
   print(f'Found {len(os.listdir(DATA_DIR))} audio files')  # Ensure the data directory exists
except FileNotFoundError:
    print(f"Data directory not found: {DATA_DIR}, proceeding anyway.")

def create_weighted_sampler(dataset, recording_weight=4.0, generated_weight=1.0):
    """Create weighted sampler for upsampling recording samples."""
    weights = []
    print(f"📊 Creating weighted sampler for {len(dataset)} samples...")
    
    # Access the underlying HuggingFace dataset directly to avoid processing audio
    for i in range(len(dataset)):
        item = dataset.ds[i]  # Access raw dataset item without processing
        sample_type = item.get('type', 'unknown')
        if sample_type == 'recording':
            weights.append(recording_weight)
        else:
            weights.append(generated_weight)
    
    recording_count = sum(1 for w in weights if w == recording_weight)
    generated_count = sum(1 for w in weights if w == generated_weight)
    print(f"📊 Weighted sampler created: {recording_count} recordings (weight={recording_weight}), {generated_count} generated (weight={generated_weight})")
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

def train(data_dir: str, model_dir: str, use_dummy: bool = False, from_hf: bool = False, epochs: int = EPOCHS):
  # Initialize wandb
  wandb.init(
      project="voice-command-classifier",
      config={
          "epochs": epochs,
          "batch_size": BATCH_SIZE,
          "learning_rate": 1e-4,
          "model": "WhisperEncoderClassifier",
          "dataset": "dummy" if use_dummy else "real",
          "optimizer": "Adam",
          "loss": "CrossEntropyLoss"
      }
  )
  

  # prepare processor and dataset
  processor = WhisperProcessor.from_pretrained(model_dir)

  # Correctly determine classes and label map based on the dataset source
  if use_dummy:
    # For dummy data, create a fixed set of classes
    classes = [f"dummy_class_{i}" for i in range(NUM_CLASSES)] # NUM_CLASSES is a global constant
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    dataset = DummyDataset(processor, n_items=DUMMY_ITEMS, num_classes=len(classes))
    print("⚙️ Using dummy dataset with random data")
  
  elif from_hf:
    print("Loading Hugging Face dataset to determine class labels...")
    # Load the full dataset once to get all unique labels
    hf_dataset = load_dataset("ntkuhn/mlx_voice_commands_mixed")
    train_ds = hf_dataset['train']
    val_ds = hf_dataset['validation']
    
    classes = sorted(train_ds.unique("class_label"))
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Create the training dataset with augmentations enabled
    dataset = AudioHFDataset(processor, label_to_idx, ds=train_ds, apply_augmentations=True)
    
    # Create the validation dataset (no augmentations for validation)
    val_dataset = AudioHFDataset(processor, label_to_idx, ds=val_ds, apply_augmentations=False)

  else: # Local CSV dataset
    data_file = data_dir.parent.parent / "audio_dataset.csv"
    df = pd.read_csv(data_file)
    classes = sorted(df["class_label"].unique())
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    dataset = AudioDataset(data_file, processor, label_to_idx)
    
    # Create a local validation dataset
    # Note: This uses the old manual loading inside run_validation.
    # To use a DataLoader, we would need to create a new Dataset class for validation data.
    # For now, we'll create a simple one.
    val_df = pd.read_csv(VALIDATION_CSV)
    # Filter out classes not in the training set
    val_df = val_df[val_df['command_token'].isin(classes)]
    # Rename for consistency
    val_df = val_df.rename(columns={'filename': 'audio_path', 'command_token': 'class_label'})
    val_dataset = AudioDataset(None, processor, label_to_idx) # Pass processor and map
    val_dataset.data = val_df # Manually set the data
    val_dataset.filenames = val_df['audio_path']
    val_dataset.labels = val_df['class_label']

  print(f"Found {len(classes)} classes:", classes)
  # Update wandb config with actual number of classes
  wandb.config.update({"num_classes": len(classes), "classes": classes})
  
  print(f"Using dataset: {type(dataset)}")

  if from_hf:
    sampler = create_weighted_sampler(dataset)
  else:
    sampler = None

  loader = DataLoader(
      dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=True if not from_hf else False, 
      sampler=sampler,
      num_workers=4, 
      pin_memory=True  # Speeds up CPU-to-GPU data transfer
  )
  
  val_loader = DataLoader(
      val_dataset,
      batch_size=BATCH_SIZE * 2, # Can use a larger batch size for validation
      shuffle=False,
      num_workers=4,
      pin_memory=True
  )

  # build model, loss, optimizer
  model = WhisperEncoderClassifier(model_dir, num_classes=len(classes)).to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  # Track model in wandb
  wandb.watch(model, log="all", log_freq=10)

  # one epoch example
  model.train() #TODO are freezing or not currently?
  correct, seen = 0, 0
  best_val_f1 = 0.0  # Track best validation F1 score
  
  for epoch in range(1, epochs + 1):
    print(f"\n▶️  Epoch {epoch}/{epochs}")
    model.train()  # ensure training mode

    correct, seen = 0, 0
    epoch_loss = 0.0
    num_batches = 0
    
    # Create progress bar for batches
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", 
                       unit="batch", leave=False)
    
    for batch in progress_bar:
      inputs  = batch["input_features"].to(DEVICE, non_blocking=True)
      labels  = batch["label"].to(DEVICE, non_blocking=True)

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

      # Track epoch loss
      epoch_loss += loss.item()
      num_batches += 1

      # Update progress bar with current metrics
      progress_bar.set_postfix({
          'loss': f'{loss.item():.4f}',
          'acc': f'{acc*100:.1f}%'
      })
    
    # Calculate epoch averages
    avg_epoch_loss = epoch_loss / num_batches
    train_acc = acc
    
    # Run validation at the end of each epoch
    val_acc, val_f1, val_loss = run_validation(model, criterion, val_loader, DEVICE)
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_epoch_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1
    })
    
    print(f"🎯 Epoch {epoch} Summary - Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f}")
    
    # Save checkpoint if this is the best model so far
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      checkpoint_path = BASE_DIR / "model" / "best_model.pt"
      checkpoint = {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'epoch': epoch,
          'classes': classes,
          'label_to_idx': label_to_idx,
          'num_classes': len(classes),
          'best_val_f1': best_val_f1,
          'val_acc': val_acc
      }
      
      # Create model directory if it doesn't exist
      checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
      
      torch.save(checkpoint, checkpoint_path)
      print(f"💾 New best model saved! Val F1: {val_f1:.3f} (previous best: {best_val_f1:.3f})")
      
      # Log model artifact to wandb
      wandb.save(str(checkpoint_path))
      
    else:
      print(f"📊 No improvement. Best Val F1 remains: {best_val_f1:.3f}")
  
  print(f"\n🏆 Training completed after {epochs} epochs")
  print(f"🎯 Best validation F1: {best_val_f1:.3f}")
  print(f"💾 Best model saved as: {BASE_DIR / 'model' / 'best_model.pt'}")
  print(f"📈 Final training accuracy: {train_acc*100:.1f}%")
  
  # Finish wandb run
  wandb.finish()

########## Functions for validation 
def run_validation(model, criterion, val_loader, device):
    """Run validation, calculating accuracy, F1, and loss in a single pass."""
    print("\n🔍 Running validation...")
    model.eval()
    
    # The old load_validation_data and preprocess_audio are no longer needed
    # as the DataLoader handles this.
    
    all_true_labels = []
    all_pred_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            # Run inference
            logits = model(inputs)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # For metrics
            predicted_indices = logits.argmax(dim=1)
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted_indices.cpu().numpy())
            
    # Calculate final metrics
    num_samples = len(all_true_labels)
    accuracy = (np.array(all_pred_labels) == np.array(all_true_labels)).mean() if num_samples > 0 else 0.0
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if num_samples > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    print(f"📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   F1 Score: {f1:.3f}")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Samples: {num_samples}")
    
    return accuracy, f1, avg_loss

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dummy", action="store_true", help="Use dummy dataset")
  parser.add_argument("--hf", action="store_true", help="Use HF dataset")
  parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS})")
  args = parser.parse_args()
  use_dummy = args.dummy
  from_hf = args.hf
  train(DATA_DIR, MODEL_DIR, use_dummy, from_hf, epochs=args.epochs)

if __name__ == "__main__":
  main()
