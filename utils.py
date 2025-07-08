import torch
from torchaudio.transforms import AmplitudeToDB, Spectrogram
import numpy as np



def plot_spectrogram_with_labels(spectrogram, sample_rate, n_fft=2048, hop_length=512, 
                               n_mels=128, title="Spectrogram", figsize=(12, 6)):
    """
    Plot a spectrogram with proper time and frequency labels.
    
    Args:
        spectrogram: 2D tensor/array of shape (n_mels, time_frames) or (freq_bins, time_frames)
        sample_rate: Sample rate of the audio
        n_fft: FFT size used for the spectrogram
        hop_length: Hop length used for the spectrogram
        n_mels: Number of mel bins (if mel spectrogram)
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError:
        print("matplotlib and librosa are required for plotting. Install with: pip install matplotlib librosa")
        return
    
    # Convert to numpy if it's a tensor
    if hasattr(spectrogram, 'numpy'):
        spec_np = spectrogram.numpy()
    else:
        spec_np = spectrogram
    
    # Remove batch dimension if present
    if spec_np.ndim == 3:
        spec_np = spec_np[0]  # Take first sample from batch
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate time axis
    time_frames = spec_np.shape[1]
    time_axis = np.arange(time_frames) * hop_length / sample_rate
    
    # Check if this is a mel spectrogram or linear spectrogram
    freq_bins = spec_np.shape[0]
    
    if freq_bins == n_mels:
        # Mel spectrogram
        # Create mel frequency axis
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sample_rate//2)
        
        # Plot using librosa's specshow for proper mel scaling
        img = librosa.display.specshow(
            spec_np, 
            sr=sample_rate, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='mel',
            fmax=sample_rate//2,
            ax=ax
        )
        ax.set_ylabel('Mel Frequency')
        
    else:
        # Linear spectrogram
        # Create linear frequency axis
        freq_axis = np.linspace(0, sample_rate/2, freq_bins)
        
        # Plot using imshow with proper extent
        img = ax.imshow(
            spec_np, 
            aspect='auto', 
            origin='lower',
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        )
        ax.set_ylabel('Frequency (Hz)')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Amplitude (dB)')
    
    # Improve layout
    plt.tight_layout()
    plt.show()

def plot_spectrogram_comparison(audio_array, sample_rate, title_prefix="Audio"):
    """
    Plot both linear and mel spectrograms side by side for comparison.
    
    Args:
        audio_array: 1D audio array
        sample_rate: Sample rate of the audio
        title_prefix: Prefix for the plot titles
    """
    try:
        import matplotlib.pyplot as plt
        from torchaudio.transforms import Spectrogram
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return
    
    # Create linear spectrogram
    linear_transform = Spectrogram(n_fft=2048, hop_length=512, power=2.0)
    to_db = AmplitudeToDB()
    
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Linear spectrogram
    linear_spec = linear_transform(audio_tensor)
    linear_spec_db = to_db(linear_spec)
    
    # Mel spectrogram (using existing function)
    mel_spec_db = create_spectrogram(audio_array, sample_rate)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot linear spectrogram
    plt.sca(ax1)
    plot_spectrogram_with_labels(
        linear_spec_db, 
        sample_rate, 
        n_fft=2048, 
        hop_length=512, 
        n_mels=linear_spec_db.shape[1],  # Use actual freq bins
        title=f"{title_prefix} - Linear Spectrogram",
        figsize=(8, 6)
    )
    
    # Plot mel spectrogram  
    plt.sca(ax2)
    plot_spectrogram_with_labels(
        mel_spec_db, 
        sample_rate, 
        n_fft=2048, 
        hop_length=512, 
        n_mels=128,
        title=f"{title_prefix} - Mel Spectrogram",
        figsize=(8, 6)
    )
    
    plt.tight_layout()
    plt.show()

def visualize_dataset_sample(dataset, idx):
    """
    Visualize a sample from the dataset with both audio waveform and spectrogram.
    
    Args:
        dataset: UrbanSoundsDataset instance
        idx: Index of the sample to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return
    
    # Get sample
    sample = dataset[idx]
    audio_array = sample['audio']['array']
    sample_rate = sample['audio']['sampling_rate']
    label = sample['label']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot waveform
    time_axis = np.arange(len(audio_array)) / sample_rate
    ax1.plot(time_axis, audio_array)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Waveform - Sample {idx} (Label: {label})')
    ax1.grid(True)
    
    # Plot spectrogram
    mel_spec = create_spectrogram(audio_array, sample_rate)
    
    # Use librosa for proper mel spectrogram display
    try:
        import librosa
        import librosa.display
        
        # Convert to numpy and remove batch dimension
        spec_np = mel_spec.numpy()
        if spec_np.ndim == 3:
            spec_np = spec_np[0]
        
        img = librosa.display.specshow(
            spec_np, 
            sr=sample_rate, 
            hop_length=512, 
            x_axis='time', 
            y_axis='mel',
            fmax=sample_rate//2,
            ax=ax2
        )
        ax2.set_ylabel('Mel Frequency')
        ax2.set_title(f'Mel Spectrogram - Sample {idx} (Label: {label})')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax2)
        cbar.set_label('Amplitude (dB)')
        
    except ImportError:
        print("librosa is required for mel spectrogram display. Install with: pip install librosa")
        # Fallback to basic imshow
        ax2.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower')
        ax2.set_ylabel('Mel Bin')
        ax2.set_xlabel('Time Frame')
        ax2.set_title(f'Mel Spectrogram - Sample {idx} (Label: {label})')
    
    plt.tight_layout()
    plt.show()
    