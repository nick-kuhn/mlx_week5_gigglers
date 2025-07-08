import itertools
import os
from datetime import datetime

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from dataset import UrbanSoundsDataset, urban_sounds_collate_fn
from models import ConvNet, ConvNetConfig, SoundEncoder, EncoderConfig


# ---------------------- Sweep configuration ----------------------
# Configuration is now kept in `sweep.yaml` for easier editing and CLI use.
SWEEP_YAML_PATH = os.path.join(os.path.dirname(__file__), "sweep.yaml")
with open(SWEEP_YAML_PATH, "r") as _f:
    sweep_config = yaml.safe_load(_f)


# ---------------------- Training routine ----------------------

def train():
    # Start a new run
    with wandb.init() as run:
        cfg = run.config

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Datasets & loaders (cached by datasets library, so OK per-run)
        train_ds = UrbanSoundsDataset(split="train")
        val_ds = UrbanSoundsDataset(split="validation")
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=urban_sounds_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=urban_sounds_collate_fn)

        # Build model based on sweep parameters
        if cfg.model_type == "ConvNet":
            conv_cfg = ConvNetConfig(
                n_classes=10,
                base_channels=cfg.base_channels,
                num_conv_blocks=cfg.num_conv_blocks,
                classification_dropout=cfg.classification_dropout,
                kernel_size=cfg.kernel_size,
                pool_kernel_size=tuple(cfg.pool_kernel),
                pool_type=cfg.pool_type,
            )
            model = ConvNet(conv_cfg)
        else:  # SoundEncoder
            enc_cfg = EncoderConfig(
                n_classes=10,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                kq_dim=cfg.kq_dim,
                embed_dim=cfg.embed_dim,
                max_seq_len=cfg.max_seq_len,
            )
            model = SoundEncoder(enc_cfg)

        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        global_step = 0
        for epoch in range(cfg.epochs):
            model.train()
            running_loss = 0.0
            for step, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (step + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    wandb.log({"train_loss": avg_loss, "epoch": epoch}, step=global_step)
                    running_loss = 0.0
                global_step += 1

            # --- Validation at epoch end ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch}, step=global_step)

        # Log final metrics (redundant but convenient)
        wandb.summary["best_val_accuracy"] = val_accuracy


# ---------------------- Sweep launcher ----------------------

def main():
    """Create a sweep on W&B and launch an agent that runs locally."""
    # You can set WANDB_PROJECT and WANDB_ENTITY env vars for nicer grouping.
    project = os.environ.get("WANDB_PROJECT", "UrbanSound8K")
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Created sweep: {sweep_id}. Running agent…")
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main() 