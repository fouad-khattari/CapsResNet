import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import random
import math

class CustomRandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=0.1307):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size(1) * img.size(2)  # Accessing dimensions of a tensor
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                img[:, x1:x1+h, y1:y1+w] = self.mean  # Ensure the operation is performed on the tensor
                return img

        return img




# Fonction pour récupérer les transformations adaptées au dataset
def get_transforms(dataset_name):
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CustomRandomErasing(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    elif dataset_name in ["FashionMNIST", "EMNIST"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CustomRandomErasing(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif dataset_name == "SVHN":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CustomRandomErasing(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return transform_train, transform_test

# Fonction pour récupérer les DataLoader en fonction du dataset choisi
def get_dataloader(dataset_name, batch_size=128, num_workers=2):
    datasets_dict = {
        "FashionMNIST": datasets.FashionMNIST,
        "EMNIST": lambda root, train, download, transform: datasets.EMNIST(root, split="balanced", train=train, download=download, transform=transform),
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "SVHN": lambda root, train, download, transform: datasets.SVHN(root, split='train' if train else 'test', download=download, transform=transform)
    }

    dataset_cls = datasets_dict.get(dataset_name)
    if not dataset_cls:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform_train, transform_test = get_transforms(dataset_name)

    # Chargement du dataset avec les transformations
    if dataset_name == "SVHN":
        train_dataset = dataset_cls(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = dataset_cls(root='./data', train=False, download=True, transform=transform_test)
    else:
        train_dataset = dataset_cls(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = dataset_cls(root='./data', train=False, download=True, transform=transform_test)

    # Division en ensemble d'entraînement et validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Création des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
