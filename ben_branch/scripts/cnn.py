### scripts/cnn.py
#!/usr/bin/env python3
"""
cnn.py – Configurable CNN model for UrbanSound8K classification.
"""
import yaml
from pathlib import Path
import torch
import torch.nn as nn

# ─── Model Definition ─────────────────────────────────────────────────────────
class CNNClassifier(nn.Module):
    """
    A configurable CNN for audio spectrogram classification.

    Configurable via config.yml under 'model':
      - conv_channels: list of ints for each Conv block
      - kernel_sizes: list of [h, w] for each Conv
      - pool_sizes: list of [ph, pw] for each block
      - mlp_hidden: list of ints for MLP layers
      - dropout: float dropout after pooling and MLP
      - num_classes: number of output classes
    """
    def __init__(self, model_cfg: dict):
        super().__init__()
        # read hyperparams
        chans      = model_cfg["conv_channels"]
        kernels    = model_cfg["kernel_sizes"]
        pools      = model_cfg["pool_sizes"]
        mlp_hidden = model_cfg["mlp_hidden"]
        dropout    = model_cfg["dropout"]
        num_classes= model_cfg["num_classes"]
        n_mels     = model_cfg["n_mels"]
        max_frames = model_cfg["max_frames"]

        # build convolutional blocks
        layers = []
        in_ch = 1  # input channel from spectrogram
        for out_ch, k, p in zip(chans, kernels, pools):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=tuple(k), padding=tuple(k//2 for k in k)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=tuple(p)),
                nn.Dropout2d(dropout)
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # compute flattened feature size by a dummy forward
        dummy = torch.zeros(1, 1, model_cfg.get("n_mels",128), model_cfg.get("max_frames",125))
        feat_dim = self.cnn(dummy).flatten(1).shape[1]

        # build MLP head
        mlp_layers = []
        last_dim = feat_dim
        for h in mlp_hidden:
            mlp_layers += [
                nn.Linear(last_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            last_dim = h
        mlp_layers.append(nn.Linear(last_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, n_mels, T]
        returns logits: [B, num_classes]
        """
        x = self.cnn(x)
        x = x.flatten(1)
        return self.mlp(x)
