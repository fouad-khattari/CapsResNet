import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils import CustomRandomErasing  # Assuming CustomRandomErasing is in utils.py

def get_transform(dataset_name, train=True):
    if dataset_name in ['mnist', 'fmnist']:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),  # Random crop with padding
                transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
                transforms.ToTensor(),  # Convert to tensor
                CustomRandomErasing(),  # Custom random erasing
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize for MNIST/FMNIST
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize for MNIST/FMNIST
            ])
    
    elif dataset_name in ['cifar10', 'cifar100']:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # Random crop with padding
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR datasets normalization
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR datasets normalization
            ])
    
    elif dataset_name == 'svhn':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
            ])
    
    elif dataset_name == 'emnist':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                CustomRandomErasing(),  # Custom random erasing
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize for EMNIST
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize for EMNIST
            ])
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return transform

def load_data(dataset_name, batch_size=64):
    # Define dataset transforms
    transform_train = get_transform(dataset_name, train=True)
    transform_test = get_transform(dataset_name, train=False)

    # Load the datasets based on dataset_name
    if dataset_name == 'fmnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'svhn':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    elif dataset_name == 'emnist':
        train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform_train)
        test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform_test)
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Split the dataset into train and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for train, validation, and test datasets
    num_workers = 2  # Adjust based on your system's capability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
