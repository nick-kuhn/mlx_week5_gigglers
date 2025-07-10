# whisper/scratch/print_model_summary.py
"""Load a locally saved Whisper model and print its layer-wise summary."""

from pathlib import Path
import argparse

import torch
from transformers import WhisperForConditionalGeneration

def print_summary(model: torch.nn.Module) -> None:
    """Print model architecture and parameter counts."""
    # total & trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:   {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}\n")
    # full layer-wise architecture
    print(model)

def main(model_dir: Path) -> None:
    """Load the model from disk and dump its summary."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    # load
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    # switch to eval mode (not strictly needed just for printing)
    model.eval()
    print_summary(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print a summary of a locally saved Whisper model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("ben_branch/whisper/models/whisper_tiny"),
        help="Path to your whisper_tiny folder",
    )
    args = parser.parse_args()
    main(args.model_dir)
