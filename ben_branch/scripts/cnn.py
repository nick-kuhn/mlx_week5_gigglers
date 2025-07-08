#!/usr/bin/env python3
"""
cnn.py – Deeper ResNet‑style CNN for UrbanSound8K classification.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List


class ResidualBlock(nn.Module):
    """Basic 2‑conv residual block with optional downsampling"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout)

        # shortcut for downsampling when channels/stride differ
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.drop(out)
        return out


class CNNClassifier(nn.Module):
    """
    A deep ResNet‑style CNN for audio spectrogram classification.

    Configurable via config.yml under 'model':
      - conv_channels: list of ints for each block
      - pool_sizes:    list of [ph, pw] for pooling per block
      - mlp_hidden:    list of ints for MLP layers
      - dropout:       float dropout in blocks & head
      - num_classes:   number of output classes
    """
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        chans: List[int]   = model_cfg["conv_channels"]
        pools: List[List[int]] = model_cfg["pool_sizes"]
        mlp_hidden: List[int]  = model_cfg["mlp_hidden"]
        dropout: float         = model_cfg["dropout"]
        num_classes: int       = model_cfg["num_classes"]

        # Build residual blocks
        layers: List[nn.Module] = []
        in_ch = 1
        for idx, out_ch in enumerate(chans):
            stride = 2 if idx > 0 else 1
            layers.append(ResidualBlock(in_ch, out_ch, stride=stride, dropout=dropout))
            layers.append(nn.MaxPool2d(kernel_size=tuple(pools[idx])))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # MLP classifier head
        head: List[nn.Module] = []
        last_dim = chans[-1]
        for h in mlp_hidden:
            head.append(nn.Linear(last_dim, h))
            head.append(nn.ReLU(inplace=True))
            head.append(nn.Dropout(dropout))
            last_dim = h
        head.append(nn.Linear(last_dim, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, 1, n_mels, frames]
        Returns:
          logits: [B, num_classes]
        """
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
