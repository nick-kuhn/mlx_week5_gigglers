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
import csv
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

# Add the project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import AudioDataset, DummyDataset
from model.model import WhisperEncoderClassifier

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
VALIDATION_DIR = BASE_DIR / "data" / "validation"
VALIDATION_CSV = VALIDATION_DIR / "recordings.csv"

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
  best_val_f1 = 0.0  # Track best validation F1 score
  
  for epoch in range(1, epochs + 1):
    print(f"\n▶️  Epoch {epoch}/{epochs}")
    model.train()  # ensure training mode

    correct, seen = 0, 0
    
    # Create progress bar for batches
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", 
                       unit="batch", leave=False)
    
    for batch in progress_bar:
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

      # Update progress bar with current metrics
      progress_bar.set_postfix({
          'loss': f'{loss.item():.4f}',
          'acc': f'{acc*100:.1f}%'
      })
    
    # Run validation at the end of each epoch
    val_acc, val_f1 = validate_model(model, processor, classes, label_to_idx, DEVICE)
    print(f"🎯 Epoch {epoch} Summary - Train Acc: {acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f}")
    
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
    else:
      print(f"📊 No improvement. Best Val F1 remains: {best_val_f1:.3f}")
  
  print(f"\n🏆 Training completed after {epochs} epochs")
  print(f"🎯 Best validation F1: {best_val_f1:.3f}")
  print(f"💾 Best model saved as: {BASE_DIR / 'model' / 'best_model.pt'}")
  print(f"📈 Final training accuracy: {acc*100:.1f}%")

########## Functions for validation 
def preprocess_audio(audio_path, processor):
    """Preprocess audio file for validation"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to numpy and squeeze
    audio_array = waveform.squeeze().numpy()
    
    # Process with Whisper processor
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    
    return inputs["input_features"]

def load_validation_data(validation_csv_path, take_every_nth=2):
    """Load validation data from CSV, taking every nth item"""
    recordings = []
    
    if not validation_csv_path.exists():
        print(f"⚠️  Validation CSV not found: {validation_csv_path}")
        return recordings
    
    with open(validation_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
    
    # Take every nth item for validation
    validation_records = all_records[::take_every_nth]
    
    for record in validation_records:
        filename = Path(record['filename']).name
        command_token = record['command_token']
        file_path = VALIDATION_DIR / filename
        
        if file_path.exists():
            recordings.append({
                'path': file_path,
                'class': command_token
            })
    
    print(f"📄 Loaded {len(recordings)} validation samples (every {take_every_nth}th from {len(all_records)} total)")
    return recordings

def validate_model(model, processor, classes, label_to_idx, device):
    """Run validation on real-world data"""
    print("\n🔍 Running validation...")
    
    # Load validation data
    validation_data = load_validation_data(VALIDATION_CSV)
    
    if not validation_data:
        print("⚠️  No validation data found")
        return 0.0, 0.0
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for item in validation_data:
            audio_path = item['path']
            true_class = item['class']
            
            # Skip if class not in training classes
            if true_class not in classes:
                continue
            
            try:
                # Preprocess audio
                input_features = preprocess_audio(audio_path, processor)
                input_features = input_features.to(device)
                
                # Run inference
                logits = model(input_features)
                predicted_idx = logits.argmax(dim=1).item()
                predicted_class = classes[predicted_idx]
                
                # Get true label index
                true_idx = label_to_idx[true_class]
                
                # Check if prediction is correct
                is_correct = predicted_idx == true_idx
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store for F1 calculation
                all_true_labels.append(true_idx)
                all_pred_labels.append(predicted_idx)
                
            except Exception as e:
                print(f"⚠️  Error processing {audio_path.name}: {str(e)}")
                continue
    
    # Calculate metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted') if len(all_true_labels) > 0 else 0.0
    
    print(f"📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   F1 Score: {f1:.3f}")
    print(f"   Samples: {total_predictions}")
    
    return accuracy, f1

def main():
  # detect --dummy flag for debug
  use_dummy = "--dummy" in sys.argv
  train(DATA_DIR, MODEL_DIR, use_dummy)

if __name__ == "__main__":
  main()
