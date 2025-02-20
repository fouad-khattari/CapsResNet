# Enhancing Image Recognition with CapsResNet: A Hybrid Model Integrating CNNs, Residual Blocks, and Capsule Layers

CapsResNet is a hybrid deep learning model for image classification that leverages Capsule Networks and Residual Blocks to achieve robust performance across various image datasets. This repository provides the implementation of the model along with the necessary components for training and evaluation. You can train the model on popular datasets such as MNIST, FashionMNIST, EMNIST, CIFAR-10, CIFAR-100, and SVHN.

## Abstract
Capsule Networks (CapsNet) have gained attention due to their ability to capture spatial hierarchies in data and improve generalization. In this work, we present **CapsResNet**, a hybrid model that combines the advantages of Capsule Networks and Residual Networks to handle the complexity of image classification tasks. By integrating residual blocks with capsules, we aim to create a robust model that can effectively learn from diverse datasets, improving performance on both simple and complex image recognition tasks. Our model demonstrates competitive results compared to traditional CNNs and is flexible enough to be applied to various datasets. 

In this repository, we provide a modular and extensible implementation of **CapsResNet**, which allows you to experiment with different image datasets and customize the model's architecture.

## Model Overview

CapsResNet integrates **Capsule Networks** and **Residual Networks**:
1. **Capsule Networks**: Designed to capture the spatial relationships between features in an image, capsule networks offer a more structured and interpretable method of learning visual patterns.
2. **Residual Blocks**: Residual connections help mitigate the vanishing gradient problem and enable the model to learn more effectively by allowing gradients to flow more easily through the network.

The model architecture consists of the following components:
- **Residual Blocks**: Each residual block consists of two convolutional layers followed by batch normalization, with an optional downsampling layer.
- **Capsule Network Layer**: A layer that applies capsule operations for detecting and preserving spatial hierarchies in the feature maps.
- **Fully Connected Layer**: The final layer that maps the extracted features to the class probabilities.

## Data Augmentation

The augmentation method used in this implementation is **Random Erasing**, which helps to improve the generalization capability of the model by randomly erasing parts of the input image during training. This technique is especially useful for handling occlusions or incomplete data in images.

In this repository, the **`CustomRandomErasing`** class found in `utils.py` is an adaptation of the Random Erasing method, originally proposed by the authors of:

> "Random Erasing Data Augmentation" by Zhong et al., 2017. [DOI: 10.48550/arXiv.1708.04896](https://doi.org/10.48550/arXiv.1708.04896)

The `CustomRandomErasing` class randomly selects a rectangular region in the image and erases it by filling it with random pixels or a constant value. This helps the model to learn more robust features that are not overly reliant on specific parts of the input image.

## Training the Model
To clarify the instructions for training your model so that they are clear and user-friendly, here’s how you can phrase it:

---

**Training the Model**

To train the model, you will first need to specify the dataset you wish to use. This is done by setting the `DATASET_NAME` variable in the `train.py` script. Here are the steps:

1. Open the `train.py` file in a text editor.
2. Find the section of the code labeled "Select dataset".
3. Modify the `DATASET_NAME` variable to match the dataset you want to use. The available options are:
   - "FashionMNIST"
   - "EMNIST"
   - "CIFAR10"
   - "CIFAR100"
   - "SVHN"

For example, if you want to use CIFAR10, you would set it like this:
DATASET_NAME = "CIFAR10"

4. Run the training script by opening your command line interface and typing:
python train.py


### Hyperparameters:
- **`epochs`**: Number of training epochs (default is 120).
- **`batch_size`**: Size of each batch during training (default is 128).

### Model Architecture:
- The **CapsResNet** model is composed of **Residual Blocks** followed by **Capsule Layers**.
- You can customize the architecture, learning rate, and other hyperparameters directly in the script or via command-line arguments.

## Files Overview:
- **`train_model.py`**: The main script for training and evaluating the model. You can specify the dataset, epochs, and batch size through command-line arguments.
- **`models.py`**: Contains the model architecture, including `CapsResNet` and `ResidualBlock`.
- **`data_loader.py`**: Responsible for loading and preprocessing datasets.
- **`utils.py`**: Contains utility functions, including the `CustomRandomErasing` data augmentation method.

## Installation and Setup

1. Clone this repository:
   git clone https://github.com/yourusername/CapsResNet.git
   cd CapsResNet

2. Install the required dependencies:
   pip install -r requirements.txt

3. Download the necessary datasets (this will be handled automatically when you run the training script for the first time).

4. Run the train.py script with the desired dataset and hyperparameters.


