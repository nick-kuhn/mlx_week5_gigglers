import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, epochs=5, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch+1}: Loss {total_loss/total:.4f}, Acc {correct/total:.4f}")

def evaluate(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    from models import AudioCNN, AudioTransformer
    from dataprep import train_loader, test_loader, class_labels

    print("Training CNN model...")
    model = AudioCNN(num_classes=len(class_labels))
    train(model, train_loader, epochs=5, lr=1e-3)
    print("Evaluating CNN model...")
    evaluate(model, test_loader)

    print("\nTraining Transformer model...")
    transformer = AudioTransformer(
        n_mels=64,
        num_classes=len(class_labels),
        d_model=128,
        nhead=4,
        num_layers=3
    )
    train(transformer, train_loader, epochs=5, lr=1e-3)
    print("Evaluating Transformer model...")
    evaluate(transformer, test_loader)
