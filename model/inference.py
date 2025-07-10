import sys
import os
from pathlib import Path
import torch
import torchaudio
from transformers import WhisperProcessor
import numpy as np
import csv

# Add the project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import WhisperEncoderClassifier

# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base folder for the script
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODEL_DIR = BASE_DIR / "whisper_models" / "whisper_tiny"
CHECKPOINT_PATH = BASE_DIR / "model" / "best_model.pt"
VALIDATION_DIR = BASE_DIR / "data" / "validation"
RECORDINGS_CSV = VALIDATION_DIR / "recordings.csv"

def load_recordings_from_csv(csv_path):
    """Load recordings data from CSV file"""
    recordings = {}
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return recordings
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = Path(row['filename']).name  # Get just the filename, not the full path
            command_token = row['command_token']
            recordings[filename] = command_token
    
    print(f"📄 Loaded {len(recordings)} recordings from CSV")
    return recordings

def load_checkpoint(checkpoint_path):
    """Load the saved checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    classes = checkpoint['classes']
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    print(f"Loaded model with {len(classes)} classes:")
    print(f"Classes: {classes}")
    
    return checkpoint, classes, label_to_idx, idx_to_label

def preprocess_audio(audio_path, processor):
    """Preprocess audio file for inference"""
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

def run_inference():
    """Run inference on validation data"""
    print(f"Using device: {DEVICE}")
    
    # Check if checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Checkpoint not found at: {CHECKPOINT_PATH}")
        print("Please run training first to create the checkpoint.")
        return
    
    # Load recordings from CSV
    recordings = load_recordings_from_csv(RECORDINGS_CSV)
    if not recordings:
        print("❌ No recordings found in CSV file")
        return
    
    # Load checkpoint
    checkpoint, classes, label_to_idx, idx_to_label = load_checkpoint(CHECKPOINT_PATH)
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    
    # Initialize model
    model = WhisperEncoderClassifier(MODEL_DIR, num_classes=len(classes)).to(DEVICE)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✅ Model loaded successfully!")
    print(f"📁 Looking for validation files in: {VALIDATION_DIR}")
    
    # Get all wav files in validation directory that are also in CSV
    wav_files = []
    for wav_file in VALIDATION_DIR.glob("*.wav"):
        if wav_file.name in recordings:
            wav_files.append(wav_file)
    
    # Also check for .WAV files (uppercase)
    for wav_file in VALIDATION_DIR.glob("*.WAV"):
        if wav_file.name in recordings:
            wav_files.append(wav_file)
    
    if not wav_files:
        print(f"❌ No .wav/.WAV files found in {VALIDATION_DIR} that match CSV entries")
        return
    
    print(f"📄 Found {len(wav_files)} validation files matching CSV entries")
    
    # Track predictions
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    print(f"\n🔍 Running inference...")
    print("=" * 80)
    
    for audio_file in sorted(wav_files):
        filename = audio_file.name
        
        # Get true class from CSV
        true_class = recordings[filename]
        
        # Check if the class exists in our model
        if true_class not in classes:
            print(f"⚠️  Unknown class for {filename}: {true_class}")
            continue
        
        try:
            # Preprocess audio
            input_features = preprocess_audio(audio_file, processor)
            input_features = input_features.to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                logits = model(input_features)
                predicted_idx = logits.argmax(dim=1).item()
                predicted_class = idx_to_label[predicted_idx]
                
                # Get confidence (softmax probability)
                probs = torch.softmax(logits, dim=1)
                confidence = probs[0, predicted_idx].item()
            
            # Check if prediction is correct
            is_correct = predicted_class == true_class
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Store result
            results.append({
                'filename': filename,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} {filename:<25} | True: {true_class:<20} | Pred: {predicted_class:<20} | Conf: {confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
    
    # Print summary
    print("=" * 80)
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"📊 SUMMARY:")
        print(f"   Total files: {total_predictions}")
        print(f"   Correct predictions: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Print confusion matrix info
        print(f"\n🔍 Detailed Results:")
        for result in results:
            if not result['correct']:
                print(f"   Misclassified: {result['filename']} -> {result['true_class']} predicted as {result['predicted_class']}")
    else:
        print("❌ No predictions made!")

if __name__ == "__main__":
    run_inference()