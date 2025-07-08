import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
from datasets import load_dataset, Audio


class UrbanSoundsDataset(torch.utils.data.Dataset):

    _cached_dataset = None

    def __init__(self, split = "train", test_fold = 10, train_folds = None):
        
        if train_folds is None:
            train_folds = list(range(1, 11))
            train_folds.remove(test_fold)


        if type(train_folds) == int:
            train_folds = [train_folds]

        if type(self)._cached_dataset is None:
            type(self)._cached_dataset = load_dataset("danavery/urbansound8k", split='train')

        # Build fold index mapping once (cached)
        if not hasattr(type(self), '_fold_indices'):
            fold_indices = {}
            for i, sample in enumerate(type(self)._cached_dataset):
                fold = sample['fold']
                if fold not in fold_indices:
                    fold_indices[fold] = []
                fold_indices[fold].append(i)
            type(self)._fold_indices = fold_indices

        # Then use .select() instead of .filter()
        if split == "train":
            indices = [idx for fold in train_folds for idx in type(self)._fold_indices[fold]]
            self.ds = type(self)._cached_dataset.select(indices)
        elif split == "validation":
            indices = type(self)._fold_indices[test_fold]
            self.ds = type(self)._cached_dataset.select(indices)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.sampling_rate = 16_000
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

def create_spectrogram(audio_array, sampling_rate):
    mel_transform = MelSpectrogram(n_fft=2048, hop_length=512, n_mels=128)
    to_db = AmplitudeToDB()

    # Handle both single samples and batches
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    
    # If single sample (1D), add batch dimension for processing
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Apply transforms - they work on batches natively
    mel_spec = mel_transform(audio_tensor)
    db_spec = to_db(mel_spec)
    
    return db_spec

def urban_sounds_collate_fn(batch):

    if batch is None:
        return None
    
    audio_features = []
    labels = []

    #get sampling rate (of a single item)
    sampling_rate = batch[0]['audio']['sampling_rate']

    #extract labels and audio features, and pad/truncate the audio features
    max_length = 4 * sampling_rate
    for i in range(len(batch)):
        labels.append(batch[i]['classID'])
        audio_array = batch[i]['audio']['array']
        
        # Handle both padding and truncation
        if len(audio_array) > max_length:
            # Truncate if too long
            audio_array = audio_array[:max_length]
        elif len(audio_array) < max_length:
            # Pad if too short
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
        
        audio_features.append(audio_array)

    #create spectrogram - already returns a tensor
    audio_features = np.array(audio_features) 
    audio_features = create_spectrogram(audio_features, sampling_rate) #(batch_size, 128, time_steps = 126)
    
    # Add channel dimension for CNN: (batch_size, 128, 126) -> (batch_size, 1, 128, 126)
    audio_features = audio_features.unsqueeze(1)

    return audio_features, torch.tensor(labels, dtype=torch.long) #(batch_size, 1, 128, time_steps), (batch_size)

def play_audio_from_dataset(dataset, idx):
    """
    Play an audio file from the dataset.
    
    Args:
        dataset: UrbanSoundsDataset instance
        idx: Index of the audio file to play
    """
    try:
        # Option 1: Using IPython.display (works in Jupyter notebooks)
        from IPython.display import Audio as IPythonAudio, display
        
        sample = dataset[idx]
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']
        
        # Display audio player in Jupyter
        audio_widget = IPythonAudio(audio_array, rate=sampling_rate)
        display(audio_widget)
        
        print(f"Playing audio sample {idx}")
        print(f"Label: {sample['label']}")
        print(f"Duration: {len(audio_array) / sampling_rate:.2f} seconds")
        
        return audio_widget
        
    except ImportError:
        print("IPython.display not available. Try running in a Jupyter notebook.")
        
        # Option 2: Using sounddevice (alternative for non-Jupyter environments)
        try:
            import sounddevice as sd
            
            sample = dataset[idx]
            audio_array = sample['audio']['array']
            sampling_rate = sample['audio']['sampling_rate']
            
            print(f"Playing audio sample {idx}")
            print(f"Label: {sample['label']}")
            print(f"Duration: {len(audio_array) / sampling_rate:.2f} seconds")
            
            sd.play(audio_array, sampling_rate)
            sd.wait()  # Wait until playback is finished
            
        except ImportError:
            print("sounddevice not available. Install with: pip install sounddevice")
            print("Or use this in a Jupyter notebook for IPython.display.Audio")
