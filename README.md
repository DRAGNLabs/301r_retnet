# RetNet VS Transformers

This project focuses on the comparison and analysis of RetNet vs Transformers, utilizing Microsoft's TorchScale as the base architecture. More information can be found on our paper [Are You Using Retentive Networks?](https://www.overleaf.com/read/fgqvqnmzfncf#a32eb5).

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

To document the "Grid Search" feature in your README with a focus on the provided Python code, you can follow this structured approach. This format highlights the functionality and purpose of the code while using Markdown formatting to ensure clarity and readability.

---

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

- Hugging Face
- Lightning
- Torchscale

## Benchmarking and Results

[Results and comparisons with graphs or tables]

You can find more information in the Results section of our paper [Are You Using Retentive Networks?](https://www.overleaf.com/read/fgqvqnmzfncf#a32eb5).

## Acknowledgments

We extend our heartfelt gratitude to the following individuals and institutions for their invaluable contributions to this project:

**BYU Office of Research Computing**: For providing the computational resources and support that were instrumental in conducting our experiments and analyses.

**Nancy Fulda**: Our esteemed instructor, whose guidance and insights have significantly shaped the direction and execution of this research.

**Microsoft TorchScale Team**: For developing and maintaining the TorchScale framework, which served as the foundational architecture for our project, enabling us to push the boundaries of what's possible in deep learning research.

**Authors of the Original RetNet Paper**: For their pioneering work in the field of deep learning, which inspired our research and provided a solid base for our explorations into RetNet and Transformers.

## Citations

This can also be found on the citation section or our paper [Are You Using Retentive Networks?](https://www.overleaf.com/read/fgqvqnmzfncf#a32eb5).

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
