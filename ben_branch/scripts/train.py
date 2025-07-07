#!/usr/bin/env python3
"""
train.py – Training loop with grid sweep, WandB logging, scheduler, early stopping.
"""
import os
import yaml
import time
import random
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import wandb
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torchaudio
import torchaudio.transforms as T
from spectrogram import MelSpecGPU
from dataloader import get_dataloader
from cnn import CNNClassifier

# ─── Setup ───────────────────────────────────────────────────────────────
# project root
ROOT = Path(__file__).resolve().parent.parent

# load config
cfg = yaml.safe_load(open(ROOT / "config.yml"))
defs = cfg["defaults"]
swp  = cfg["sweep"]

# load .env for W&B key
load_dotenv(ROOT / ".env")
# assume WANDB_API_KEY in env

# reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─── Training fn ─────────────────────────────────────────────────────────

def train_and_evaluate():
    """Single run, picking all hyper‐params from wandb.config."""
    # Initialize W&B run (agent will supply config)
    wandb.init()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, CUDA version: {torch.version.cuda},  GPU count: {torch.cuda.device_count()}")

    # access hyperparameters from wandb.config
    cfg_w = wandb.config

    # access hyperparameters (and coerce away from strings
    bs         = int(cfg_w.batch_size)
    lr         = float(cfg_w.lr)
    sched_name = cfg_w.scheduler           # still a str
    epochs     = int(cfg_w.epochs)
    patience   = int(cfg_w.patience)

    # get data loaders
    train_loader = get_dataloader("train", batch_size=bs)
    val_loader   = get_dataloader("val",   batch_size=bs)

    # instantiate model with param dict
    model = CNNClassifier({
        "conv_channels": cfg_w.conv_channels,
        "kernel_sizes":  cfg_w.kernel_sizes,
        "pool_sizes":    cfg_w.pool_sizes,
        "mlp_hidden":    cfg_w.mlp_hidden,
        "dropout":       cfg_w.dropout,
        "num_classes":   cfg_w.num_classes,
        "n_mels":        cfg_w.n_mels,
        "max_frames":    cfg_w.max_frames
    }).to(device)

    # optimizer and scheduler setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = (CosineAnnealingLR(optimizer, T_max=epochs)
                 if sched_name == "cosine"
                 else StepLR(optimizer, step_size=10, gamma=0.5))

    # early stopping trackers
    best_val_loss = float("inf")
    patience_cnt  = 0

    # mixed precision scaler
    scaler = GradScaler()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # instantiate one MelSpecGPU and dB converter on the GPU
    mel_gpu = MelSpecGPU().to(device)
    db_gpu  = T.AmplitudeToDB(stype='power').to(device)

    # training loop
    for epoch in range(1, epochs + 1):
        # ----- training -----
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for i, (waveforms, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), 1):
            # send raw waveforms to GPU
            waveforms = waveforms.to(device)            # shape [B, T]
            labels    = labels.to(device)
     
            # compute GPU spectrograms
            specs = mel_gpu(waveforms)                  # [B, n_mels, frames]
            specs = db_gpu(specs)                       # log-scale in dB
            specs = specs.unsqueeze(1)                  # [B, 1, n_mels, frames]
     
            optimizer.zero_grad()
            with autocast():
                outputs = model(specs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # accumulate metrics
            preds = outputs.argmax(dim=1)
            running_loss     += loss.item() * waveforms.size(0)
            if labels.dim() == 1:
                running_corrects += (preds == labels).sum().item()
            else:
                running_corrects += (preds == labels.argmax(dim=1)).sum().item()
            total  += waveforms.size(0)

            # log batch metrics every 20 batches
            if i % 20 == 0:
                # compute batch‐level accuracy, handling soft/hard labels
                if labels.dim() == 1:
                    batch_acc = (preds == labels).float().mean().item()
                else:
                    hard_labels = labels.argmax(dim=1)
                    batch_acc   = (preds == hard_labels).float().mean().item()

                wandb.log({
                    "train_loss_batch": loss.item(),
                    "train_acc_batch": batch_acc
                }, step=(epoch-1)*len(train_loader) + i)


        # epoch training metrics
        epoch_loss = running_loss / total
        epoch_acc  = running_corrects / total
        wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc}, step=epoch)

                # ----- validation -----
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc="Validating", leave=False):
                waveforms = waveforms.to(device)
                labels    = labels.to(device)
                specs  = mel_gpu(waveforms)
                specs  = db_gpu(specs).unsqueeze(1)
                outputs = model(specs)
                loss    = criterion(outputs, labels)
                # aggregate loss
                val_loss     += loss.item() * waveforms.size(0)
                # compute predictions
                preds = outputs.argmax(dim=1)
                # handle mixup vs hard labels
                if labels.dim() == 1:
                    val_corrects += (preds == labels).sum().item()
                else:
                    val_corrects += (preds == labels.argmax(dim=1)).sum().item()
                val_total    += waveforms.size(0)
        # finalize metrics
        val_loss /= val_total
        val_acc   = val_corrects / val_total
        wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)

        # Print epoch summary
        print(
            f"Epoch {epoch}/{epochs} "
            f"Train ▶️ loss={epoch_loss:.4f} acc={epoch_acc:.4f} | "
            f"Val ▶️ loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # scheduler step
        scheduler.step()

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), ROOT / "best_model.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # finish run
    wandb.finish()

# ─── Sweep launcher using WandB agent ─────────────────────────────────────
if __name__ == "__main__":
    # build W&B sweep config from our sweep section in config.yml
    sweep_cfg = {
        "method":  "grid",
        "metric":  {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "conv_channels": {"values": swp["model"]["conv_channels"]},
            "kernel_sizes":  {"values": swp["model"]["kernel_sizes"]},
            "pool_sizes":    {"values": swp["model"]["pool_sizes"]},
            "mlp_hidden":    {"values": swp["model"]["mlp_hidden"]},
            "dropout":       {"values": swp["model"]["dropout"]},
            "num_classes":   {"value":  swp["model"]["num_classes"][0]},
            "n_mels":        {"value":  swp["model"]["n_mels"][0]},
            "max_frames":    {"value":  swp["model"]["max_frames"][0]},
            "batch_size":    {"values": swp["training"]["batch_size"]},
            "lr":            {"values": swp["training"]["lr"]},
            "epochs":        {"value":  swp["training"]["epochs"][0]},
            "patience":      {"value":  swp["training"]["patience"][0]},
            "scheduler":     {"values": swp["training"]["scheduler"]}
        }
    }
    # create sweep on W&B
    sweep_id = wandb.sweep(sweep_cfg, project="urbansound8k")
    # launch an agent that runs train_and_evaluate() for each combo
    wandb.agent(sweep_id, function=train_and_evaluate)
