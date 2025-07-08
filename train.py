from dataset import UrbanSoundsDataset, urban_sounds_collate_fn
import torch
import torch.nn as nn
from models import ConvNet
from torch.utils.data import DataLoader


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, max_training_steps=1000, log_interval=10, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = 0
    for epoch in range(epochs):
        model.train()
        stepi = []
        lossi = []
        train_loss = 0
        running_loss = 0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_loss += loss.item()
            if step % log_interval == 0 and step > 0:
                stepi.append(step)
                lossi.append(running_loss / log_interval)
                print(f"Step {step}, Loss: {running_loss / log_interval}")
                running_loss = 0
            total_steps += 1
            if total_steps >= max_training_steps:
                break
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        # Calculate average train loss based on actual number of batches processed
        num_batches_processed = min(step + 1, len(train_loader))
        avg_train_loss = train_loss / num_batches_processed
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
        if total_steps >= max_training_steps:
            break
    return {
        "stepi": stepi,
        "lossi": lossi,
        "final_val_metrics": val_metrics
    }

def evaluate_model(model, val_loader, criterion, device="cpu"):
    model.eval()
    val_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += len(labels)
    return {
        "val_loss": val_loss / len(val_loader),  # Average loss per batch
        "val_accuracy": total_correct / total_samples  # Accuracy based on total samples
    }

def main():
    train_dataset = UrbanSoundsDataset(split="train")
    val_dataset = UrbanSoundsDataset(split="validation")
    model = ConvNet()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=urban_sounds_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=urban_sounds_collate_fn)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)

if __name__ == "__main__":
    main()



