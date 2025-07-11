#!/usr/bin/env python3
"""
Example script demonstrating how to train with augmentations applied only to recording samples.
This script shows how to customize augmentation parameters and run training.
"""

import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import AudioHFDataset
from model.model import WhisperEncoderClassifier
from model.augmentations import DEFAULT_CONFIG

# Custom augmentation configuration
# You can adjust these parameters based on your specific needs
CUSTOM_AUG_CONFIG = {
    "time_shift": {
        "shift_max": 0.15,  # Reduce time shift for voice commands
        "prob": 0.7
    },
    "add_noise": {
        "noise_level": 0.003,  # Lower noise level for speech
        "prob": 0.6
    },
    "pitch_shift": {
        "n_steps": [-1.5, 1.5],  # Smaller pitch changes for natural speech
        "prob": 0.4
    },
    "time_stretch": {
        "rate": [0.85, 1.15],  # Smaller speed changes
        "prob": 0.4
    },
    "mixup": {
        "alpha": 0.2,
        "prob": 0.2  # Lower probability for mixup
    },
    "spec_masking": {
        "freq_mask": {
            "F": 10,  # Smaller frequency masks for speech
            "num_masks": 1,
            "prob": 0.5
        },
        "time_mask": {
            "T": 15,  # Smaller time masks for speech
            "num_masks": 1,
            "prob": 0.5
        }
    }
}

def train_with_augmentations():
    """Example training function with augmentations."""
    
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 5
    
    print(f"🚀 Training with augmentations for recording samples only")
    print(f"📱 Using device: {DEVICE}")
    
    # Load dataset
    print("📁 Loading dataset...")
    hf_dataset = load_dataset("ntkuhn/mlx_voice_commands_mixed")
    train_ds = hf_dataset['train']
    val_ds = hf_dataset['validation']
    
    # Get classes
    classes = sorted(train_ds.unique("class_label"))
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    print(f"📊 Found {len(classes)} classes: {classes}")
    
    # Initialize processor
    model_dir = PROJECT_ROOT / "whisper_models" / "whisper_tiny"
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    
    # Create datasets with custom augmentation config
    print("🔄 Creating datasets with augmentations...")
    train_dataset = AudioHFDataset(
        processor, 
        label_to_idx, 
        ds=train_ds, 
        apply_augmentations=True,
        augmentation_config=CUSTOM_AUG_CONFIG
    )
    
    val_dataset = AudioHFDataset(
        processor, 
        label_to_idx, 
        ds=val_ds, 
        apply_augmentations=False  # No augmentations for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("🤖 Initializing model...")
    model = WhisperEncoderClassifier(str(model_dir), num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize wandb
    wandb.init(
        project="voice-command-augmented",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 1e-4,
            "augmentations": "recording_only",
            "augmentation_config": CUSTOM_AUG_CONFIG,
            "num_classes": len(classes)
        }
    )
    
    # Training loop
    print("🎓 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Track augmentation statistics
        recording_samples = 0
        generated_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Get batch data
            inputs = batch["input_features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            sample_types = batch["sample_type"]
            
            # Count sample types
            recording_samples += sum(1 for t in sample_types if t == 'recording')
            generated_samples += sum(1 for t in sample_types if t == 'generated')
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total*100:.1f}%'
            })
        
        # Epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        print(f"   Recording samples processed: {recording_samples}")
        print(f"   Generated samples processed: {generated_samples}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "recording_samples": recording_samples,
            "generated_samples": generated_samples
        })
    
    print("✅ Training completed!")
    wandb.finish()

def test_augmentations():
    """Quick test to verify augmentations are working correctly."""
    print("🧪 Testing augmentation pipeline...")
    
    # Load a small sample
    hf_dataset = load_dataset("ntkuhn/mlx_voice_commands_mixed", split="train[:10]")
    
    # Check if we have both recording and generated samples
    types = [item.get('type', 'unknown') for item in hf_dataset]
    print(f"📊 Sample types: {set(types)}")
    
    # Initialize processor
    model_dir = PROJECT_ROOT / "whisper_models" / "whisper_tiny"
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    
    # Get classes
    classes = sorted(hf_dataset.unique("class_label"))
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Test dataset with augmentations
    test_dataset = AudioHFDataset(
        processor, 
        label_to_idx, 
        ds=hf_dataset, 
        apply_augmentations=True,
        augmentation_config=CUSTOM_AUG_CONFIG
    )
    
    # Test a few samples
    print("🔍 Testing individual samples:")
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        sample_type = sample['sample_type']
        print(f"   Sample {i}: Type={sample_type}, Features shape={sample['input_features'].shape}")
    
    print("✅ Augmentation test completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with augmentations or test the pipeline")
    parser.add_argument("--test", action="store_true", help="Run augmentation test instead of training")
    args = parser.parse_args()
    
    if args.test:
        test_augmentations()
    else:
        train_with_augmentations() 