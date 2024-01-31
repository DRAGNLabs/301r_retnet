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

### Note on CUDA

Check if your computer has an Nvidia GPU, as CUDA is required for some functionalities.
CUDA comes as part of the requirements in the torch package.
If you don't have an Nvidia GPU, you can still run the project on a CPU, but some functionalities might be limited.
Alternatively, consider using a supercomputer if available, but ensure there are not a ton of parameters to train.

## Features

- [Feature 1]
- [Feature 2]
- [Feature 3]

## Benchmarking and Results

[Results and comparisons with graphs or tables]

## Acknowledgments

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
