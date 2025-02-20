import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.caps_resnet import CapsResNet
from utils.dataset_loader import get_dataloader

# Select dataset
DATASET_NAME = "FashionMNIST"  # Options: "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100", "SVHN"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_loader, val_loader, test_loader = get_dataloader(DATASET_NAME, batch_size=128)

# Define model (change input channels and image size based on dataset)
if DATASET_NAME in ["CIFAR10", "CIFAR100", "SVHN"]:
    input_channels, dim = 3, 32
else:
    input_channels, dim = 1, 28

# Define number of classes
num_classes = 10 if DATASET_NAME != "CIFAR100" else 100


model = CapsResNet(input_channels=input_channels, num_classes=num_classes, dim=dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# Training loop
num_epochs = 120
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(torch.softmax(output, dim=1), 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = torch.max(torch.softmax(output, dim=1), 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Adjust learning rate
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save trained model
torch.save(model.state_dict(), f"capsresnet_{DATASET_NAME}.pth")
print(f"Model saved as capsresnet_{DATASET_NAME}.pth")
