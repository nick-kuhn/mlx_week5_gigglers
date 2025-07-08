#!/usr/bin/env python3
"""
train.py – Training loop with grid sweep, WandB logging, scheduler, early stopping.
FIXED: All major issues resolved
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
from sklearn.metrics import f1_score


# ─── Setup ───────────────────────────────────────────────────────────────
# project root
ROOT = Path(__file__).resolve().parent.parent

# load config
cfg = yaml.safe_load(open(ROOT / "config.yml"))
defs = cfg["defaults"]
swp  = cfg["sweep"]

# load .env for W&B key
load_dotenv(ROOT / ".env")

# reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_audio_batch(waveforms, mel_gpu, db_gpu):
    """
    Convert waveforms to normalized spectrograms.
    Input: waveforms [B, T] 
    Output: specs [B, n_mels, frames] - normalized
    """
    # Convert to mel spectrogram: [B, T] -> [B, n_mels, frames]
    specs = mel_gpu(waveforms)
    
    # Convert to dB scale: [B, n_mels, frames] -> [B, n_mels, frames]
    specs = db_gpu(specs)
    
    # Normalize per sample across freq and time dims
    # specs shape: [B, n_mels, frames]
    specs = (specs - specs.mean(dim=(-2,-1), keepdim=True)) / \
            (specs.std(dim=(-2,-1), keepdim=True) + 1e-6)
    
    return specs

# ─── Training fn ─────────────────────────────────────────────────────────

def train_and_evaluate():
    """Single run, picking all hyper‐params from wandb.config."""
    # Initialize W&B run (agent will supply config)
    wandb.init()
    
    # FIXED: Set seed for reproducibility
    set_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, CUDA version: {torch.version.cuda}, GPU count: {torch.cuda.device_count()}")

    # access hyperparameters from wandb.config
    cfg_w = wandb.config

    # access hyperparameters (and coerce away from strings)
    bs         = int(cfg_w.batch_size)
    lr         = float(cfg_w.lr)
    sched_name = cfg_w.scheduler
    epochs     = int(cfg_w.epochs)
    patience   = int(cfg_w.patience)

    # get data loaders
    train_loader = get_dataloader("train", batch_size=bs)
    val_loader   = get_dataloader("val",   batch_size=bs)
    
    # Validate data shapes
    print("Validating data shapes...")
    for waveforms, labels in train_loader:
        print(f"Train batch - Waveforms: {waveforms.shape}, Labels: {labels.shape}")
        break
    for waveforms, labels in val_loader:
        print(f"Val batch - Waveforms: {waveforms.shape}, Labels: {labels.shape}")
        break

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

    # instantiate spectrogram processors on GPU
    mel_gpu = MelSpecGPU().to(device)
    db_gpu  = T.AmplitudeToDB(stype='power').to(device)

    # training loop
    for epoch in range(1, epochs + 1):
        # ----- training -----
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for i, (waveforms, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), 1
        ):
            # Validate input shapes
            assert waveforms.dim() == 2, f"Expected 2D waveforms [B,T], got {waveforms.shape}"
            assert labels.dim() == 1, f"Expected 1D labels [B], got {labels.shape}"
            
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            # Ensure labels are integers (handle any mixup remnants)
            if labels.dim() > 1:
                labels = labels.argmax(dim=1)

            # Process audio with proper shape handling
            specs = process_audio_batch(waveforms, mel_gpu, db_gpu)
            
            # Add channel dimension for CNN if needed
            # Ensure specs are 4D: [B, 1, n_mels, frames]
            if specs.dim() == 3:
                specs = specs.unsqueeze(1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(specs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # accumulate train metrics
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * waveforms.size(0)
            running_corrects += (preds == labels).sum().item()
            total += waveforms.size(0)

            # Log every 50 batches instead of 5
            if i % 50 == 0:
                train_acc = (preds == labels).float().mean().item()
                step = (epoch - 1) * len(train_loader) + i
                wandb.log({
                    "train_loss_batch": loss.item(),
                    "train_acc_batch": train_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=step)

                print(f"Step {step} Train loss={loss:.4f} acc={train_acc:.4f}")

        # epoch training metrics
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        # ----- validation -----
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc="Validating", leave=False):
                waveforms, labels = waveforms.to(device), labels.to(device)
                
                # Ensure labels are integers
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
                
                # Use same audio processing as training
                specs = process_audio_batch(waveforms, mel_gpu, db_gpu)
                
                # ensure specs are 4D: [B, 1, n_mels, frames]
                if specs.dim() == 3:
                    specs = specs.unsqueeze(1)
                
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * waveforms.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()
                val_total += waveforms.size(0)

                # collect for F1
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # finalize validation metrics
        val_loss /= val_total
        val_acc = val_corrects / val_total

        # compute F1 score
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_f1_macro = f1_score(all_labels, all_preds, average="macro")

        # Enhanced debugging output
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        pred_dist = dict(zip(unique_preds, pred_counts))
        label_dist = dict(zip(unique_labels, label_counts))
        
        print(f"Epoch {epoch} - Prediction distribution: {pred_dist}")
        print(f"Epoch {epoch} - True label distribution: {label_dist}")
        
        # Check if model is predicting only one class (major red flag)
        if len(unique_preds) == 1:
            print("🚨 WARNING: Model predicting only one class! Check learning rate, loss, gradients.")
            
        # Check prediction entropy (low entropy = model is too confident/collapsed)
        pred_entropy = -np.sum((pred_counts/pred_counts.sum()) * np.log(pred_counts/pred_counts.sum() + 1e-8))
        print(f"Prediction entropy: {pred_entropy:.3f} (max={np.log(10):.3f})")

        # log all metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
            "pred_entropy": pred_entropy,
            "num_predicted_classes": len(unique_preds),
            "learning_rate": scheduler.get_last_lr()[0],
        })

        # Print epoch summary
        print(
            f"========= Epoch {epoch}/{epochs} =========\n"
            f"Train: loss={epoch_loss:.4f} acc={epoch_acc:.4f}\n"
            f"Val:   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1_macro:.4f}\n"
            f"Pred classes: {len(unique_preds)}/10, Entropy: {pred_entropy:.3f}"
        )

        # scheduler step
        scheduler.step()

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), ROOT / "best_model.pth")
            print(f"✅ New best model saved! Val loss: {val_loss:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    # finish run
    wandb.finish()

# ─── Sweep launcher ─────────────────────────────────────────────────────
if __name__ == "__main__":
    bayes_cfg = {
        "method": "bayes",
        "metric": {"name": "val_f1_macro", "goal": "maximize"},
        "parameters": {
            "conv_channels": {"values": swp["model"]["conv_channels"]},
            "mlp_hidden": {"values": swp["model"]["mlp_hidden"]},
            "dropout": {"values": swp["model"]["dropout"]},
            "lr": {"values": swp["training"]["lr"]},
            "scheduler": {"values": swp["training"]["scheduler"]},
            
            # fixed params
            "kernel_sizes": {"value": swp["model"]["kernel_sizes"][0]},
            "pool_sizes": {"value": swp["model"]["pool_sizes"][0]},
            "num_classes": {"value": swp["model"]["num_classes"][0]},
            "n_mels": {"value": swp["model"]["n_mels"][0]},
            "max_frames": {"value": swp["model"]["max_frames"][0]},
            "batch_size": {"value": swp["training"]["batch_size"][0]},
            "epochs": {"value": swp["training"]["epochs"][0]},
            "patience": {"value": swp["training"]["patience"][0]},
        }
    }

    sweep_id = wandb.sweep(bayes_cfg, project="urbansound8k")
    wandb.agent(sweep_id, function=train_and_evaluate, count=10)