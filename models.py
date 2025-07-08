import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class ConvNetConfig:
    n_classes: int = 10
    base_channels: int = 32
    num_conv_blocks: int = 3
    classification_dropout: float = 0.5
    kernel_size: int = 3
    pool_kernel_size: tuple[int, int] = (2, 2)
    pool_type: str = "max"

class ConvNet(nn.Module):
    def __init__(self, config: Optional[ConvNetConfig] = ConvNetConfig()):
        super().__init__()
        self.config = config
        self.base_channels = config.base_channels
        self.num_conv_blocks = config.num_conv_blocks
        self.classification_dropout = config.classification_dropout
        self.kernel_size = config.kernel_size
        self.padding = self.kernel_size // 2

        # choose pooling layer class instead of a single instance
        if config.pool_type == "max":
            pool_cls = nn.MaxPool2d
        elif config.pool_type == "avg":
            pool_cls = nn.AvgPool2d
        else:
            raise ValueError(f"Invalid pool type: {config.pool_type}")
        # helper lambda to create a new pooling layer each time
        make_pool = lambda: pool_cls(config.pool_kernel_size)

        #First convolution block
        conv_blocks = [
            nn.Conv2d(1, self.base_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding), 
            nn.BatchNorm2d(self.base_channels), nn.ReLU(), make_pool()
        ]
        #Middle convolution blocks
        for i in range(1, self.num_conv_blocks):
            conv_blocks.extend([
                nn.Conv2d(self.base_channels * (2**(i-1)), self.base_channels * (2**i), kernel_size=self.kernel_size, stride=1, padding=self.padding), 
                nn.BatchNorm2d(self.base_channels * (2**i)), nn.ReLU(), make_pool()
            ])
        #Last convolution block
        conv_blocks.extend([
            nn.Conv2d(self.base_channels * (2**(self.num_conv_blocks-1)), self.base_channels * (2**self.num_conv_blocks), kernel_size=self.kernel_size, stride=1, padding=self.padding), 
            nn.BatchNorm2d(self.base_channels * (2**self.num_conv_blocks)), nn.ReLU(), make_pool()
        ])
        conv_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        #Compile the convolution blocks to a single module
        self.conv = nn.Sequential(*conv_blocks)

        self.dropout = nn.Dropout(self.classification_dropout)
        # Linear layer expects flattened features
        self.classifier = nn.Linear(self.base_channels * (2**self.num_conv_blocks), config.n_classes)  # 128 features -> n_classes

    def forward(self, x):
        # x.shape = (batch_size, 1, mel_bins, T)
        #out_channels = self.base_channels * (2**self.num_conv_blocks)
        x = self.conv(x) # (batch_size, out_channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, out_channels)
        
        x = self.dropout(x)
        
        x = self.classifier(x)  # (batch_size, n_classes)        
        return x

@dataclass
class EncoderConfig:
    n_classes: int
    n_layers: int
    n_heads: int
    kq_dim: int
    embed_dim: int
    max_seq_len: int

class TransformerEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.kq_dim = config.kq_dim
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.n_heads, dim_feedforward=self.kq_dim)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

def sinusoidal_embeddings(max_seq_len, embed_dim):
    """Create sinusoidal positional embeddings like Whisper's encoder"""
    position = torch.arange(max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                        -(math.log(10000.0) / embed_dim))
    
    pos_emb = torch.zeros(max_seq_len, embed_dim)
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    
    return pos_emb  


class SoundEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        #apply 1d convolutions to the columns of the spectrogram
        self.stem = nn.Sequential(
            nn.Conv1d(128, config.embed_dim, kernel_size=3, stride=1, padding=1),  # Along time dimension
            nn.GELU(),
            nn.Conv1d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1),  # Along time dimension
            nn.GELU()
        )
        #add a positional encoding to the embeddings (including CLS token)
        self.register_buffer('positional_encoding', 
                    sinusoidal_embeddings(config.max_seq_len + 1, config.embed_dim))
        #Add classification token
        self.cls_token = nn.Parameter(torch.randn(1,1, config.embed_dim))
        #apply transformer encoder to the spectrogram
        self.encoder = TransformerEncoder(config)
        #apply a linear layer to the output of the transformer encoder
        self.out_proj = nn.Linear(config.embed_dim, config.n_classes)



    def forward(self, x):
        if (x.shape[2] + 1) // 2 > self.config.max_seq_len:
            raise ValueError(f"Input sequence length {x.shape[2]} is too long (maximum: {2 * self.config.max_seq_len - 1})")
        # Input: (batch_size, 1, 128, T = 126) - 1 channel, 128 mel bins, 126 time steps
        x = x.squeeze(1) # (batch_size, 128, T)
        x = self.stem(x) # (batch_size, embed_dim, T)
        x = x.transpose(1,2) # (batch_size, T, embed_dim)
        
        # Add classification token to the beginning
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, T+1, embed_dim)
        
        x = x + self.positional_encoding[:x.shape[1]] # (batch_size, T+1, embed_dim)
        x = self.encoder(x)
        
        # Use only the CLS token for classification
        cls_output = x[:, 0]  # (batch_size, embed_dim)
        x = self.out_proj(cls_output)  # (batch_size, n_classes)
        return x





