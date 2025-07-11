#!/usr/bin/env python3
"""
Audio augmentation utilities for voice command training.
Adapted from ben_branch/scripts/augmentations.py for use with the current model.
"""

import numpy as np
import random
import torch
import librosa
import torchaudio.transforms as T
from typing import Tuple, Optional
import torch.nn.functional as F


# Default augmentation parameters - these can be moved to a config file later
DEFAULT_CONFIG = {
    "time_shift": {
        "shift_max": 0.2,  # Maximum shift as fraction of sample rate
        "prob": 0.8
    },
    "add_noise": {
        "noise_level": 0.005,  # Noise level
        "prob": 0.7
    },
    "pitch_shift": {
        "n_steps": [-2, 2],  # Range of semitones
        "prob": 0.5
    },
    "time_stretch": {
        "rate": [0.8, 1.25],  # Speed range
        "prob": 0.5
    },
    "mixup": {
        "alpha": 0.2,  # Beta distribution parameter
        "prob": 0.3
    },
    "spec_masking": {
        "freq_mask": {
            "F": 15,  # Frequency mask parameter
            "num_masks": 2,
            "prob": 0.6
        },
        "time_mask": {
            "T": 20,  # Time mask parameter
            "num_masks": 2,
            "prob": 0.6
        }
    }
}


class AudioAugmentations:
    """Audio augmentation class that applies various transformations to waveforms."""
    
    def __init__(self, config: Optional[dict] = None, sample_rate: int = 16000):
        """
        Initialize augmentations with configuration.
        
        Args:
            config: Dictionary of augmentation parameters. If None, uses DEFAULT_CONFIG.
            sample_rate: Sample rate of audio data.
        """
        self.config = config or DEFAULT_CONFIG
        self.sample_rate = sample_rate
        
        # Initialize spectrogram masking transforms
        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=self.config["spec_masking"]["freq_mask"]["F"]
        )
        self.time_mask = T.TimeMasking(
            time_mask_param=self.config["spec_masking"]["time_mask"]["T"]
        )
    
    def time_shift(self, wav: np.ndarray) -> np.ndarray:
        """Randomly shift audio in time."""
        if random.random() > self.config["time_shift"]["prob"]:
            return wav
            
        max_shift = int(self.config["time_shift"]["shift_max"] * self.sample_rate)
        shift = random.randint(-max_shift, max_shift)
        return np.roll(wav, shift)
    
    def add_noise(self, wav: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to audio."""
        if random.random() > self.config["add_noise"]["prob"]:
            return wav
            
        noise_level = self.config["add_noise"]["noise_level"]
        noise = np.random.randn(len(wav)) * noise_level
        return wav + noise
    
    def pitch_shift(self, wav: np.ndarray) -> np.ndarray:
        """Randomly shift pitch by semitones."""
        if random.random() > self.config["pitch_shift"]["prob"]:
            return wav
            
        n_steps_range = self.config["pitch_shift"]["n_steps"]
        n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
        
        try:
            return librosa.effects.pitch_shift(
                y=wav,
                sr=self.sample_rate,
                n_steps=n_steps
            )
        except Exception as e:
            print(f"Warning: Pitch shift failed: {e}")
            return wav
    
    def time_stretch(self, wav: np.ndarray) -> np.ndarray:
        """Randomly speed up or slow down audio."""
        if random.random() > self.config["time_stretch"]["prob"]:
            return wav
            
        rate_range = self.config["time_stretch"]["rate"]
        rate = random.uniform(rate_range[0], rate_range[1])
        
        try:
            return librosa.effects.time_stretch(y=wav, rate=rate)
        except Exception as e:
            print(f"Warning: Time stretch failed: {e}")
            return wav
    
    def apply_waveform_augmentations(self, wav: np.ndarray) -> np.ndarray:
        """Apply all waveform augmentations in sequence."""
        wav = self.time_shift(wav)
        wav = self.add_noise(wav)
        wav = self.pitch_shift(wav)
        wav = self.time_stretch(wav)
        return wav
    
    def apply_spectrogram_augmentations(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram masking augmentations."""
        # Apply frequency masking
        if random.random() < self.config["spec_masking"]["freq_mask"]["prob"]:
            for _ in range(self.config["spec_masking"]["freq_mask"]["num_masks"]):
                spec = self.freq_mask(spec)
        
        # Apply time masking  
        if random.random() < self.config["spec_masking"]["time_mask"]["prob"]:
            for _ in range(self.config["spec_masking"]["time_mask"]["num_masks"]):
                spec = self.time_mask(spec)
        
        return spec
    
    def mixup_batch(self, batch_audio: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation to a batch of audio samples.
        
        Args:
            batch_audio: Batch of audio tensors [B, ...]
            batch_labels: Batch of labels [B]
            
        Returns:
            Tuple of (mixed_audio, mixed_labels) where mixed_labels are soft labels
        """
        if random.random() > self.config["mixup"]["prob"]:
            return batch_audio, batch_labels
        
        alpha = self.config["mixup"]["alpha"]
        lam = np.random.beta(alpha, alpha)
        
        batch_size = batch_audio.size(0)
        indices = torch.randperm(batch_size)
        
        # Mix audio
        mixed_audio = lam * batch_audio + (1 - lam) * batch_audio[indices]
        
        # Create soft labels
        num_classes = batch_labels.max().item() + 1
        y_a = F.one_hot(batch_labels, num_classes).float()
        y_b = F.one_hot(batch_labels[indices], num_classes).float()
        mixed_labels = lam * y_a + (1 - lam) * y_b
        
        return mixed_audio, mixed_labels


def create_augmentation_pipeline(config: Optional[dict] = None, sample_rate: int = 16000) -> AudioAugmentations:
    """
    Factory function to create an augmentation pipeline.
    
    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate
        
    Returns:
        AudioAugmentations instance
    """
    return AudioAugmentations(config, sample_rate) 