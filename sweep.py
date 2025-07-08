import itertools
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import UrbanSoundsDataset, urban_sounds_collate_fn
from train import train_model
from models import ConvNet, SoundEncoder, EncoderConfig


def run_sweep():
    # Prepare datasets and dataloaders once to reuse across runs
    print("Loading UrbanSound8K dataset...")
    train_dataset = UrbanSoundsDataset(split="train")
    val_dataset = UrbanSoundsDataset(split="validation")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=urban_sounds_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=urban_sounds_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyper-parameter grids
    convnet_grid = {
    "pool_kernel": [(2, 2), (2, 1), (1, 2)],
    "pool_type": ["max", "avg"],
    "kernel_size": [3, 5],
    "lr": [1e-3, 5e-4],
    "num_conv_blocks": [2, 3, 4],
    "base_channels": [16, 32, 64],
    "classification_dropout": [0.2, 0.5]
    }

    soundencoder_grid = {
        "n_layers": [2, 4],
        "n_heads": [4, 8],
        "embed_dim": [128, 256],
        "kq_dim": [512, 1024],
        "lr": [1e-3, 1e-4]
    }

    results = []

    # Sweep ConvNet
    print("\n========== ConvNet sweep ==========")
    for lr in convnet_grid["lr"]:
        hyperparams = {"lr": lr}
        print(f"Running ConvNet with params: {hyperparams}")
        model = ConvNet()
        metrics = train_model(
            model,
            train_loader,
            val_loader,
            epochs=10,
            lr=lr,
            device=device,
            max_training_steps=1000
        )
        val_acc = metrics["final_val_metrics"]["val_accuracy"]
        results.append({
            "model": "ConvNet",
            **hyperparams,
            "val_accuracy": val_acc
        })
        print(f"Validation accuracy: {val_acc:.4f}\n")

    # Sweep SoundEncoder
    print("\n========== SoundEncoder sweep ==========")
    keys = list(soundencoder_grid.keys())
    values_product = itertools.product(*[soundencoder_grid[k] for k in keys])
    for values in values_product:
        hp = dict(zip(keys, values))
        print(f"Running SoundEncoder with params: {hp}")

        # Build config for encoder
        config = EncoderConfig(
            n_classes=10,
            n_layers=hp["n_layers"],
            n_heads=hp["n_heads"],
            kq_dim=hp["kq_dim"],
            embed_dim=hp["embed_dim"],
            max_seq_len=64,
        )
        model = SoundEncoder(config)
        metrics = train_model(
            model,
            train_loader,
            val_loader,
            epochs=10,
            lr=hp["lr"],
            device=device,
            max_training_steps=1000
        )
        val_acc = metrics["final_val_metrics"]["val_accuracy"]
        results.append({
            "model": "SoundEncoder",
            **hp,
            "val_accuracy": val_acc
        })
        print(f"Validation accuracy: {val_acc:.4f}\n")

    # Sort results by accuracy descending
    results_sorted = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)

    # Save results to a timestamped JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"sweep_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_sorted, f, indent=2)

    print("\n========== Sweep completed ==========")
    print(f"Top 5 configurations:")
    for res in results_sorted[:5]:
        print(res)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    run_sweep() 