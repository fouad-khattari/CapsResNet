```md
# CapsResNet: A Hybrid Residual Capsule Network for Image Classification

This repository provides the **CapsResNet** model, a hybrid architecture combining **Residual Networks (ResNet) and Capsule Networks** for image classification. The model supports **FashionMNIST, EMNIST, CIFAR10, CIFAR100, and SVHN**.

This work is fully **reproducible** with clear **documentation, public source code, and dataset usage instructions**.

## 📌 Features
✔️ **ResNet + Capsule Network** combination  
✔️ **Supports multiple datasets**  
✔️ **Easy to train and test**  
✔️ **Fully open-source and reproducible**  

---

## 📌 Installation

### **1️⃣ Install Dependencies**
Ensure Python **3.10 or 3.11** is installed (Python 3.13 is **not** supported).  
Then, install the dependencies:

```bash
pip install -r requirements.txt
```

> **Windows Users:** Install **Microsoft Visual C++ Redistributable** if needed:  
> [Download VC++ 2019 (x64)](https://aka.ms/vs/16/release/vc_redist.x64.exe)

### **2️⃣ Install PyTorch**
For **GPU (CUDA 12.1)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
For **CPU only**:
```bash
pip install torch torchvision torchaudio
```

---

## 📌 Usage

### **1️⃣ Training the Model**
Run the training script:
```bash
python train.py
```
This will:
- Load the dataset
- Train the **CapsResNet** model
- Save the trained model as `capsresnet_{DATASET_NAME}.pth`

#### **🛠 Selecting a Dataset**
Modify `train.py` to choose a dataset:
```python
DATASET_NAME = "CIFAR10"  # Options: "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100", "SVHN"
```

### **2️⃣ Evaluating the Model**
After training, test the model using:
```bash
python test.py
```
This loads the saved model and computes accuracy on the test set.

---

## 📌 Datasets
This model supports **5 datasets**:

| Dataset       | Classes | Image Size | Type |
|--------------|---------|------------|------|
| FashionMNIST | 10      | 28x28      | Grayscale |
| EMNIST       | 47      | 28x28      | Grayscale |
| CIFAR10      | 10      | 32x32      | Color (RGB) |
| CIFAR100     | 100     | 32x32      | Color (RGB) |
| SVHN         | 10      | 32x32      | Color (RGB) |

Each dataset is **automatically downloaded** when running `train.py`.

---

## 📌 Code Structure
The repository is **modularized** for clarity:

```
CapsResNet-Classification/
│── models/
│   ├── residual_block.py      # Defines Residual Block
│   ├── capsule_layer.py       # Defines Capsule Layer
│   ├── caps_resnet.py         # Full CapsResNet model
│── utils/
│   ├── dataset_loader.py      # Loads datasets
│   ├── training.py            # Training logic
│   ├── evaluation.py          # Evaluation logic
│── train.py                    # Main training script
│── test.py                     # Model evaluation
│── requirements.txt            # Dependencies
│── README.md                   # Documentation
│── LICENSE                     # License (MIT)
```

---

## 📌 Reproducibility
To ensure **reproducibility**, we provide:
- **Fixed Random Seeds**: The code sets a **seed** for deterministic behavior.
- **Complete Code and Documentation**: All training parameters and datasets are described.
- **Public Model Checkpoints**: The trained models are available.

You can manually set a **random seed** in `train.py`:
```python
import torch
import numpy as np
import random

seed = 42  # Change this for different results
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

---

## 📌 Citation
If you use this work, please cite:

```
@article{your_paper,
  title={CapsResNet: A Hybrid Residual Capsule Network for Image Classification},
  author={Your Name(s)},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```

We have also included a **citation format** inside our **GitHub repository description** as required by the journal editor.

---

## 📌 License
This project is released under the **MIT License**. You are free to **use, modify, and distribute** it.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software...
```

---
