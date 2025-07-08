import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# --- CNN Model ---
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 31, 128)  # 4 sec at 16kHz -> 64x126 mel spectrogram
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 63]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 31]
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, time, d_model]
        x = x + self.pe[:x.size(1)]
        return x

# --- Custom Transformer Encoder Block ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, seq, d_model]
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

# --- Full Transformer Classifier Model ---
class AudioTransformer(nn.Module):
    def __init__(self, n_mels=64, num_classes=10, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(n_mels, d_model)
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        x = x.squeeze(1).transpose(1, 2)  # [batch, time, n_mels]
        x = self.embedding(x)              # [batch, time, d_model]
        for encoder in self.encoder_blocks:
            x = encoder(x)                 # [batch, time, d_model]
        x = x.mean(dim=1)                  # global average pooling over time
        x = self.classifier(x)
        return x

