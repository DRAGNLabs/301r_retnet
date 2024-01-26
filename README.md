# Retnet VS Transformers

## Introduction

This project focuses on the comparison and analysis of RetNet vs Transformers, utilizing Microsoft's TorchScale as the base architecture.

## Features

- [Feature 1]
- [Feature 2]
- [Feature 3]

## Base Architecture - Microsoft TorchScale

This project is built upon [Microsoft TorchScale](https://github.com/microsoft/torchscale), which serves as the base architecture for our research. TorchScale provides a set of tools and utilities for training and evaluating deep learning models efficiently. We have leveraged its capabilities to develop our comparison between RetNet and Transformers.

## Installation and Setup

To get started with this project, first clone the RetNet repository using the following command:

````bash
git clone https://github.com/DRAGNLabs/301r_retnet.git
cd 301r_retnet
```

Make sure to have Python=3.11
When you use a .sh script, run this to grant permission "chmod +x retnet.sh" for the script you want to use
Then do this "./retnet.sh" to run the script

_Hyperparameters Scaling_

### Python Code Examples

#### 0. **Preparing Data**:

This repository is designed to be run on a cluster. As such, the data must be downloaded and prepared before running the training script.

To download data from the HuggingFace Hub, run the `download_data.py` script with the following parameters:

- dataset_name : the name of the dataset on HF hub
- dataset_subset : the subset to use
- dataset_dir : the name of the folder to save the dataset in. Typically, you can just keep this the same as the dataset_name

#### 1. **Setting Up a Basic Model**:

This snippet shows how to initialize a neural network with basic hyperparameters.

```python
import torch
import torch.nn as nn
from your_project_model import RetNetModel  # Replace with your actual model import

# Basic hyperparameters
hyperparams = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_layers": 4,
    "embed_dim": 256,
    "num_heads": 8,
    "dropout_rate": 0.1
}

model = RetNetModel(num_layers=hyperparams["num_layers"],
                    embed_dim=hyperparams["embed_dim"],
                    num_heads=hyperparams["num_heads"],
                    dropout_rate=hyperparams["dropout_rate"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
````

#### 2. **Implementing Learning Rate Scheduler**:

An example of using a learning rate scheduler.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    # Training process
    ...
    scheduler.step()  # Update learning rate
```

#### 3. **Random Search for Hyperparameter Tuning**:

Pseudo-code for implementing random search.

```python
import random

def random_search_hyperparams():
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [32, 64, 128]
    num_layers_options = [2, 4, 6]

    # Randomly select hyperparameters
    lr = random.choice(learning_rates)
    batch_size = random.choice(batch_sizes)
    num_layers = random.choice(num_layers_options)

    # Create and return a model with these hyperparameters
    model = create_model(lr, batch_size, num_layers)
    return model

# Example usage
tuned_model = random_search_hyperparams()
# Train and evaluate this model
```

### Command Line Examples

#### 1. **Running a Training Script with Hyperparameters**:

Assuming you have a training script `train.py`, you can run it with different hyperparameters.

```bash
python train.py --learning-rate 0.001 --batch-size 64 --num-layers 4 --embed-dim 256
```

#### 2. **Using a Configuration File**:

For complex models, managing hyperparameters via a config file can be more convenient.

```bash
python train.py --config config.yaml
```

In `config.yaml`, you can have:

```yaml
learning_rate: 0.001
batch_size: 64
num_layers: 4
embed_dim: 256
```

### Note

- Replace placeholders like `your_project_model` and `RetNetModel` with actual module and class names from your project.
- These examples are meant to be adapted to your specific project setup and requirements.
- The effectiveness of hyperparameter tuning can depend heavily on the specific nature of the dataset and task.
