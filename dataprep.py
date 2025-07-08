import torch
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pandas as pd

# Load dataset from HuggingFace
dataset = load_dataset("danavery/urbansound8K", split="train")

# Fold-based split
df = pd.DataFrame(dataset)
train_indices = df[df['fold'] != 10].index.tolist()
test_indices = df[df['fold'] == 10].index.tolist()
train_dataset = dataset.select(train_indices)
test_dataset = dataset.select(test_indices)

# Cast audio column to automatically decode audio to arrays
train_dataset = train_dataset.cast_column("audio", Audio())
test_dataset = test_dataset.cast_column("audio", Audio())

# Class label mapping
class_labels = sorted(set(dataset['class']))
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}

# Dataset parameters
SR = 16000        # Target sampling rate
N_MELS = 64       # Number of Mel bands
DURATION = 4      # Seconds, deep learning models typically use fixed-length inputs
TARGET_LENGTH = SR * DURATION  # Total samples (fixed length per clip), ensures that all audio clips are forced to 64k samples

class UrbanSoundDataset(Dataset):
    def __init__(self, dataset, label_to_idx):
        self.dataset = dataset
        self.label_to_idx = label_to_idx 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        y = item["audio"]["array"] #returns the audio waveform as a numpy array i.e. for the 64k samples it returns an array of audio amplitudes
        sr = item["audio"]["sampling_rate"] 
        # Resample if needed
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        # Pad or trim to fixed length
        if len(y) > TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        else:
            y = np.pad(y, (0, TARGET_LENGTH - len(y))) # Pad with zeros if shorter than TARGET_LENGTH

        # Compute Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        logmel = librosa.power_to_db(mel)
        #mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MELS)
        #stacked = np.stack([logmel, mfcc], axis=0)
        logmel = np.expand_dims(logmel, axis=0)  # Add channel dimension: [1, n_mels, time]
        label = self.label_to_idx[item["class"]]
        #return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return torch.tensor(logmel, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# PyTorch DataLoaders
train_torch_dataset = UrbanSoundDataset(train_dataset, label_to_idx)
test_torch_dataset = UrbanSoundDataset(test_dataset, label_to_idx)
print("Test set size:", len(test_dataset))
train_loader = DataLoader(train_torch_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_torch_dataset, batch_size=32, shuffle=False)
