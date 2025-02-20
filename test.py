import torch
from models.caps_resnet import CapsResNet
from utils.dataset_loader import get_dataloader

# Select dataset
DATASET_NAME = "FashionMNIST"  #  Options: "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100", "SVHN"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
_, _, test_loader = get_dataloader(DATASET_NAME, batch_size=128)

# Load model
num_classes = 10 if DATASET_NAME != "CIFAR100" else 100

if DATASET_NAME in ["CIFAR10", "CIFAR100", "SVHN"]:
    input_channels, dim = 3, 32
else:
    input_channels, dim = 1, 28


model = CapsResNet(input_channels=input_channels, num_classes=num_classes, dim=dim).to(device)
model.load_state_dict(torch.load(f"capsresnet_{DATASET_NAME}.pth"))
model.eval()

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(torch.softmax(output, dim=1), 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Print final accuracy
print(f"Test Accuracy on {DATASET_NAME}: {100 * correct / total:.2f}%")
