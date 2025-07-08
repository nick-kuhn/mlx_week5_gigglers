import torch
import matplotlib.pyplot as plt
from dataprep import train_torch_dataset, class_labels

N = 4  # Number of samples to visualize
for i in range(N):
    features, label = train_torch_dataset[i]  # features: [2, n_mels, time]
    logmel_np = features[0].numpy()
    mfcc_np = features[1].numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    axs[0].imshow(logmel_np, aspect="auto", origin="lower")
    axs[0].set_title(f"Log-Mel Spectrogram (Class: {class_labels[label.item()]})")
    axs[0].set_ylabel("Mel bands")
    axs[1].imshow(mfcc_np, aspect="auto", origin="lower")
    axs[1].set_title("MFCC")
    axs[1].set_xlabel("Time frames")
    axs[1].set_ylabel("MFCC Coef")
    plt.tight_layout()
    plt.show()