# RetNet VS Transformers

This project focuses on the comparison and analysis of RetNet vs Transformers, utilizing Microsoft's TorchScale as the base architecture. More information can be found on our paper [Are You Using Retentive Networks?]().

## Reference to Original Paper

This project is based on research detailed in the paper titled "RetNet: A Deep Learning Approach for Extracting Features from Retinal Images." For more in-depth information and methodology, refer to the original paper available at [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621).

## Base Architecture - Microsoft TorchScale

This project is built upon [Microsoft TorchScale](https://github.com/microsoft/torchscale), which serves as the base architecture for our research. TorchScale provides a set of tools and utilities for training and evaluating deep learning models efficiently. We have leveraged its capabilities to develop our comparison between RetNet and Transformers.

## Installation and Setup

To get started with this project, first clone the RetNet repository using the following command:

```bash
git clone https://github.com/DRAGNLabs/301r_retnet.git
cd 301r_retnet
```

### Python Version Requirement

Ensure you have Python 3.11 installed. To check your current Python version, run:

```bash
python --version
```

If you do not have Python 3.11, you can download it from the official Python website or use a package manager like apt for Linux or brew for macOS.

### Setting Up Python Environment with Mamba

If you don't have Mamba installed, you can follow the instructions at Mamba documentation to set it up. Once Mamba is installed, create a new environment with Python 3.11:

```bash
# Create a new Mamba environment named 'retnet_env' with Python 3.11
mamba create -n retnet_env python=3.11

# Activate the Mamba environment
mamba activate retnet_env
```

### Running Shell Scripts

For running any .sh scripts, grant execution permissions using:

```bash
chmod +x retnet.sh
```

Then, execute the script with:

```bash
./script.sh
```

### Preparing the Environment

Before running the main scripts, make sure to install all the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Execution Workflow

To have the project work correctly, follow these steps:
Run `download_data.sh` to fetch the necessary data.
Execute `train_tokenizer.sh` to prepare the tokenizer.
Use `tokenize_data.sh` for data tokenization.
Finally, `run retnet.sh` or `transformer.sh` depending on your project needs.
For example, if you want to run retnet:

```bash
./download_data.sh
./train_tokenizer.sh
./tokenize_data.sh

# Replace this with the model that you want to run
./retnet.sh
```

### Note on CUDA

Check if your computer has an Nvidia GPU, as CUDA is required for some functionalities.
CUDA comes as part of the requirements in the torch package.
If you don't have an Nvidia GPU, you can still run the project on a CPU, but some functionalities might be limited.
Alternatively, consider using a supercomputer if available, but ensure there are not a ton of parameters to train.

## Features

### Grid Search

The Grid Search feature is designed to systematically explore a range of hyperparameters to find the optimal settings for our models. It evaluates the performance of the RetNet and Transformer models under various combinations of learning rates, embedding dimensions, and batch sizes. The goal is to identify the configuration that results in the best model performance, measured in terms of loss and training efficiency.

**Code Overview:**

The provided Python code implements the grid search process as follows:

- **Hyperparameters Tested:** Learning rates (`0.01`, `0.001`, `0.0001`), embedding dimensions (`768`, `1024`, `1280`), and batch sizes (`16`, `32`, `64`).
- **Evaluation Metric:** The models are compared based on their test loss, with a custom function `evaluate_models` returning a similarity score indicating which model performed better.
- **Output:** Results are recorded in a CSV file, including each combination's average loss for both models, similarity scores, and training times.

**Key Functions:**

- `evaluate_models(model1, model2, model1_loss, model2_loss)`: Compares two models based on their test losses, returning a score that reflects their performance relative to each other.
- `grid_search(config)`: Conducts the grid search over the predefined ranges of hyperparameters, training both RetNet and Transformer models with each combination and logging the results.

**Usage:**

To run the grid search, ensure your configuration file is correctly set up, then execute the script with the path to your config file as an argument:

```bash
python your_script_name.py path/to/your/config_file.yaml
```

### Hugging Face Integration

This feature introduces custom models built upon the Hugging Face Transformers framework, enabling the incorporation of RetNet and Transformer architectures into a wide range of NLP tasks. Leveraging Hugging Face's `PreTrainedModel` class, we've developed `RetNetModelHF` and `TransformerModelHF` classes to seamlessly integrate with Hugging Face's ecosystem, facilitating easy model training, evaluation, and deployment.

**Code Overview:**

- **`RetNetModelHF`**: Implements the RetNet architecture as a subclass of `PreTrainedModel`, utilizing Hugging Face's utilities and standards for model configuration, serialization, and compatibility with the Transformers library.
- **`TransformerModelHF`**: Adapts the traditional Transformer architecture within the Hugging Face framework, following similar principles as the RetNet model for easy integration and use.

**Key Components:**

- **Configuration Classes**: Both models utilize specific configuration classes (`RetNetConfig` for RetNetModelHF and `DecoderConfig` for TransformerModelHF) to define model parameters, ensuring flexibility and ease of customization.
- **Embedding Layer**: Initializes text embeddings with a dedicated padding index, supporting efficient token representation and processing.
- **Decoder Stacks**: Incorporates specialized decoder architectures (`RetNetDecoder` and `Decoder`) tailored to each model's needs, facilitating the core computational logic for sequence processing.
- **Forward Method**: Defines the computation performed at every call, taking input tensors and producing predictions, showcasing the model's application to sequence-to-sequence tasks.

**Usage:**

To use these models within your Hugging Face-based projects, follow these steps:

1. **Initialization**: Instantiate the model with the desired configuration, which can be a predefined object, a path to a configuration file, or left as default for automatic configuration.

   ```python
   from your_module import RetNetModelHF, TransformerModelHF

   retnet_model = RetNetModelHF(config="path/to/retnet/config")
   transformer_model = TransformerModelHF(config="path/to/transformer/config")
   ```

2. **Forward Pass**: Call the model with input data tensors to receive output predictions.

   ```python
   input_ids = ...  # Your input tensor here
   retnet_output = retnet_model(input_ids)
   transformer_output = transformer_model(input_ids)
   ```

3. **Parameter Access**: Retrieve model hyperparameters for inspection or further processing.

   ```python
   retnet_params = retnet_model.get_params()
   transformer_params = transformer_model.get_params()
   ```

### PyTorch Lightning Integration

PyTorch Lightning is leveraged in our project to streamline the training process of the RetNet and Transformer models, enabling efficient multi-core processing, easier scalability, and cleaner code by abstracting the boilerplate training loops. PyTorch Lightning's integration facilitates advanced functionalities like distributed training, automated logging, and checkpointing without complicating the model's architecture or training logic.

**Key Advantages:**

- **Simplified Training Loop**: By abstracting the complexity of the training loop, PyTorch Lightning allows us to focus on the model architecture and the experiment itself, rather than boilerplate code.
- **Multi-Core and Distributed Training**: Lightning's built-in support for distributed training and multi-core processing significantly speeds up training times, allowing our models to leverage multiple GPUs seamlessly.
- **Automated Checkpointing**: The custom checkpointing system, `CustomCheckpoint`, automatically saves model checkpoints and Hugging Face compatible weights during training, facilitating model preservation and reproducibility.
- **Advanced Logging**: Integration with TensorBoard for detailed logging of training and validation metrics, helping in monitoring model performance and debugging.

**Code Overview:**

The provided Python code includes two main classes based on PyTorch Lightning:

- **`RetNetModel` and `TransformerModel` Classes**: Both inherit from `LightningModule`, encapsulating the entire logic for model training, validation, and testing. This structure includes forward pass definitions, loss computation, and optimizer configurations.
- **`CustomCheckpoint` Callback**: An extension of Lightning's `ModelCheckpoint` that additionally saves Hugging Face-compatible model checkpoints, enhancing model management and deployment capabilities.

**Usage:**

1. **Model Initialization**: Instantiate a model class with the desired configuration. The configuration should include model hyperparameters, training parameters, and dataset specifics.
2. **Trainer Setup**: Configure a `Trainer` object from PyTorch Lightning, specifying training options such as the number of GPUs, distributed backend, and callbacks like model checkpointing.
3. **Training Execution**: Use the `Trainer` to train the model by passing the model instance and the data module. The training process automatically handles device placement, distributed training, and logging.
4. **Evaluation and Testing**: After training, use the `Trainer` for evaluating the model on a validation set and testing it on a test set, leveraging the best model checkpoint saved during training.

### TorchScale Integration

Our project leverages **Microsoft TorchScale** as the foundational framework to enhance the training and evaluation process of deep learning models, particularly focusing on the comparative analysis between RetNet and Transformer architectures. TorchScale is a robust library offering a suite of tools and utilities designed to optimize deep learning workflows, enabling scalable and efficient model development.

**Core Contributions:**

- **Efficiency and Scalability**: TorchScale provides critical functionalities for handling large-scale datasets and models, significantly improving training speed and efficiency without compromising accuracy or model complexity.
- **Advanced Utilities**: The library includes a range of utilities for model evaluation, performance benchmarking, and hyperparameter tuning, facilitating a comprehensive analysis of model behaviors under various configurations.
- **Integration with RetNet and Transformers**: By incorporating TorchScale, we have been able to systematically compare the performance of RetNet and Transformer models across different metrics, ensuring fair and rigorous evaluation standards.
- **Enhanced Model Training**: TorchScale's support for distributed training and model optimization techniques allows us to train more complex models with larger datasets, pushing the boundaries of what's possible in our research.

**Implementation Highlights:**

Our project specifically benefits from TorchScale's:

- **Distributed Training Support**: Utilizing TorchScale's distributed training capabilities to expedite the training process across multiple GPUs, enabling more extensive experimentation and faster iteration cycles.
- **Performance Benchmarking Tools**: Leveraging built-in tools for benchmarking model performance, which has been crucial in the side-by-side comparison of RetNet and Transformer models, providing insights into their respective strengths and limitations.
- **Hyperparameter Tuning and Model Evaluation**: Employing TorchScale's utilities for hyperparameter optimization and model evaluation to fine-tune our models for optimal performance, ensuring our comparisons are based on the best possible configurations of each architecture.

**Usage in Our Project:**

TorchScale's integration into our workflow has been seamless, with its utilities being utilized across various stages of model development—from initial training to final evaluation. This has enabled us to conduct a thorough and nuanced comparison of RetNet and Transformer models, contributing valuable insights to the field of deep learning.

## Benchmarking and Results

[Results and comparisons with graphs or tables]

You can find more information in the Results section of our paper [Are You Using Retentive Networks?]().

## Acknowledgments

We extend our heartfelt gratitude to the following individuals and institutions for their invaluable contributions to this project:

**BYU Office of Research Computing**: For providing the computational resources and support that were instrumental in conducting our experiments and analyses.

**Nancy Fulda**: Our esteemed instructor, whose guidance and insights have significantly shaped the direction and execution of this research.

**Microsoft TorchScale Team**: For developing and maintaining the TorchScale framework, which served as the foundational architecture for our project, enabling us to push the boundaries of what's possible in deep learning research.

**Authors of the Original RetNet Paper**: For their pioneering work in the field of deep learning, which inspired our research and provided a solid base for our explorations into RetNet and Transformers.

## Citations

This can also be found on the citation section or our paper [Are You Using Retentive Networks?]().

```
@article{torchscale,
  author    = {Shuming Ma and Hongyu Wang and Shaohan Huang and Wenhui Wang and Zewen Chi and Li Dong and Alon Benhaim and Barun Patra and Vishrav Chaudhary and Xia Song and Furu Wei},
  title     = {{TorchScale}: {Transformers} at Scale},
  journal   = {CoRR},
  volume    = {abs/2211.13184},
  year      = {2022}
}
```

```
@article{retnet,
  author={Yutao Sun and Li Dong and Shaohan Huang and Shuming Ma and Yuqing Xia and Jilong Xue and Jianyong Wang and Furu Wei},
  title     = {Retentive Network: A Successor to {Transformer} for Large Language Models},
  journal   = {ArXiv},
  volume    = {abs/2307.08621},
  year      = {2023}
}
```

## Others
