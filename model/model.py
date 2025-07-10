import torch
from torch import nn
from transformers import WhisperModel


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        # Add learnable CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Attention between CLS token and all time steps
        pooled, _ = self.multihead_attn(cls_tokens, hidden_states, hidden_states)
        return pooled.squeeze(1)


class WhisperEncoderClassifier(nn.Module):
  """
  Classification model using Whisper's encoder as a feature extractor
  plus a simple linear head.
  """
  def __init__(self, model_dir: str, num_classes: int = 10, freeze_encoder: bool = True, dropout: float = 0.1):
    super().__init__()
    # load pretrained encoder
    try:
      self.whisper = WhisperModel.from_pretrained(model_dir).encoder
    except Exception as e:
      raise RuntimeError(f"Failed to load Whisper model from {model_dir}: {e}")

    self.num_classes = num_classes
    self.dropout = dropout
    self.hidden_size = self.whisper.config.d_model
    
    self.pooling = MultiHeadAttentionPooling(self.hidden_size)
    
    # classification head
    self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    if freeze_encoder:
      for param in self.whisper.parameters():
        param.requires_grad = False

  def unfreeze_last_n_layers(self, n: int):
    pass #TODO implement this

  def forward(self, input_features: torch.Tensor) -> torch.Tensor:
    """
    We input a single mel spec for each audio file via input_features: (batch, seq_len, feature_dim)
    the model will split the audio up into timeframes internally and output a hidden state for each
    timeframe. So encoder_outputs will have multiple embeddings per input audio sample.
    """
    # Get the encoder outputs, will split audio into timesteps internally
    encoder_outputs = self.whisper(input_features)
    # Get the last hidden state only for each time step
    hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_size)
    # Mean pool over time steps
    pooled = self.pooling(hidden_states)
    # Run linear classifier
    return self.classifier(pooled)