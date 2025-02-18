import argparse
import torch
import torch.optim as optim
from models import CapsResNet
from data_loader import load_data
from utils import CustomRandomErasing

def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss and backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Update loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%")

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total}%")

def main():
    parser = argparse.ArgumentParser(description='Train the CapsResNet model')
    parser.add_argument('--dataset', type=str, default='fmnist', choices=['mnist', 'fmnist', 'emnist', 'cifar10', 'cifar100', 'svhn'], help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # Load the data (train, validation, and test loaders)
    train_loader, val_loader, test_loader = load_data(args.dataset, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Instantiate the model
    model = CapsResNet(num_classes=10).to(device)  # Adjust number of classes based on dataset

    # Set up the optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
