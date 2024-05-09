# <PAPER_TITLE_PLACEHOLDER> 📝

This project compares and analyzes the RetNet and the Transformer architectures, utilizing Microsoft's TorchScale library for implementation. More information can be found in our paper # <PAPER_TITLE_PLACEHOLDER> 📝 and corresponding trained models can be found on the DRAGN-Labs [HuggingFace](https://huggingface.co/DRAGN-Labs/DRAGN-3B-Transformer) page. 

## Reference to Original Paper

Our study is based on research detailed in the paper [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621). For more in-depth information and methodology of the RetNet architecture, refer to this paper.

## Base Architecture - Microsoft TorchScale

This project is built upon [Microsoft TorchScale](https://github.com/microsoft/torchscale), which provides basic implementations of each architecture for our research. TorchScale provides a library of foundational architecture implementations for training Transformer-based deep learning models. We have leveraged its capabilities to compare RetNet and Transformer architectures.

## Installation and Setup

To get started with this project, first clone this repository using the following command:

```bash
git clone https://github.com/DRAGNLabs/301r_retnet.git
cd 301r_retnet
```

### 1: Development Environment

Ensure you have Python 3.11 installed. If you do not have Python 3.11, you can download it from the [official Python website](https://www.python.org) or use a package manager.

Optionally, you can create a virtual environment. Then install all the necessary dependencies. An example with [Mamba](https://mamba.readthedocs.io/en/latest/index.html) is:
```bash
# Optionally create a new Mamba environment with Python 3.11 and specify a name
mamba create -n <YOUR_ENV_NAME> python=3.11

# Activate the Mamba environment
mamba activate <YOUR_ENV_NAME>
```

Follow NVIDIA Conda CUDA installation steps below:
```bash
# Make sure GPU available
lspci | grep -i nvidia

mamba install cuda -c nvidia
```

Make sure that `ninja` is installed:
```bash
mamba install ninja
```

Once your environment has been prepared, install all required packages:
```bash
pip install -r requirements.txt
```

To install Flash Attention:
```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

### 2: YAML Configuration Files

This project uses YAML configuration files to store all pipeline parameters and paths. The design choice of the YAML file is intended to eliminate repetition of commonly used parameters across code, as well as simplify future changes and refactors, allow developers to add new parameters, and make all settings visible to the user in one consolidated place.

To prepare a YAML config file, copy [template_config.yaml](./configs/template_config.yaml) into the [user_configs](./configs/user_configs/) folder. Fill out all parameters accordingly. Absolute paths are preferred for any path variables, but the repository is set up to work flexibly with any desired directory structure.

*Note:* In most cases, YAML does not expect strings. Adding quotation marks around arguments in the config file can lead to unexpected errors.

### 3: Script Execution

> [!TIP]
> Once a YAML config file is prepared, it can be passed into any script in the pipeline. Before you run any scripts, it is recommended to copy all of them into the [user_scripts](./scripts/user_scripts/) directory and modify the scripts to point to the right config file.

The Python scripts for data preprocessing and training can be run via the Bash scripts found in [scripts](./scripts/), or through Slurm via the scripts in [slurm](./slurm/). This README refers to the normal Bash scripts, but the process for running the scripts via Slurm is similar.

The expected order of script execution is as follows:
1. Change current directory to [scripts/user_scripts](./scripts/user_scripts/).
2. Run [download_data.sh](./scripts/download_data.sh) to fetch the necessary data.
3. Execute [split_data.sh](./scripts/split_data.sh) to divide the dataset into splits.
4. Execute [train_tokenizer.sh](./scripts/train_tokenizer.sh) to prepare the tokenizer.
5. Use [tokenize_data.sh](./scripts/tokenize_data.sh) for data tokenization and include the names of the data splits separated by spaces.
6. Finally, run [train_model.sh](./scripts/train_model.sh).

For example, if you want to train a RetNet model:

```bash
cd scripts/user_scripts

./download_data.sh
./split_data.sh
./train_tokenizer.sh
./tokenize_data.sh train validation test
./train_model.sh
```

More details for each of these steps are described more detail in the following sections.

#### Data Preprocessing

This repository uses [Dask](https://docs.dask.org/en/stable/) to load and process data. Our code is configured to load datasets into Dask from Parquet files. If using a different file format, this may need to be changed.

##### Downloading Data

This repository is designed to utilize datasets available through the HuggingFace Hub. There are two ways to download data:

1. By running [download_data.py](./src/download_data.py), via the [download_data.sh](./scripts/download_data.sh) script, you can download data through the [HuggingFace Filesystem](https://huggingface.co/docs/huggingface_hub/v0.22.0.rc0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem). To do this, ensure that the correct parameters are set in the configuration YAML, specifically `hf_filesystem_path`. This must correspond to the correct HF Filesystem path. This method is good for smaller datasets that can easily fit in memory.

2. By cloning the HuggingFace Dataset repo directly, and downloading the necessary data. [download_c4.sh](./scripts/download_c4.sh) exists as an example of this for the C4 dataset. This method is good for very large datasets.

It should be noted that datasets can come in a variety of different formats. Currently, this repo works best with [Parquet](https://parquet.apache.org/) files. If the data is downloaded in a different format, the code may need to be changed to accomodate.

##### Splitting the Data

After downloading the data, you can split the data into separate train/validation/test splits via the [split_data.py](./src/split_data.py) script, run through [split_data.sh](./scripts/split_data.sh).

Optionally, within [split_data.py](./src/split_data.py), you can specify to shuffle the data while splitting. This is more expensive for larger datasets. This and any other preprocessing or pretokenization steps should occur here prior to splitting the data.

##### Training a Tokenizer

A tokenizer can be trained by running the [train_tokenizer.py](./src/train_tokenizer.py) script through [train_tokenizer.sh](./scripts/train_tokenizer.sh). Ensure that the proper paths are set in your configuration file.

##### Tokenizing the Data

[tokenizer_data.py](./src/tokenize_data.py) will tokenize your data, using the tokenizer you have specified. An additional parameter, `split`, is needed to pass into this script. It can be `train`, `validation`, or `test`. This allows you to tokenize each split in parallel. [tokenize_data.sh](./scripts/tokenize_data.sh) is setup to do each split in parallel. When running through Slurm, you will need to start a job for each split. See [tokenize_data.sh](./slurm/tokenize_data.sh) for more information.

#### Training

You can train a model by running [train_model.py](./src/train_model.py) through [train_model.sh](./scripts/train_model.sh). During training, data is loaded lazily through Dask, and padded/truncated dynamically for each batch. This behaviour can be seen/changed in [dataset.py](./src/dataset.py)

## Features

### Grid Search

The Grid Search feature is designed to systematically explore a range of hyperparameters and compare RetNet and Transformer models with corresponding parameters at each point. This evaluates both architectures with various combinations of learning rates, embedding dimensions, feed-forward dimensions, sequence lengths, and number of heads. The goal is to identify the configuration that results in the best model performance, measured in terms of loss and training efficiency.

**Code Overview:**

We implement the grid search process as follows:

- **Hyperparameters Tested:** Learning rates (`0.001`, `0.0005`, `0.0001`), embedding dimensions (`768`, `1024`, `1280`), feed-forward dimensions (`1024`, `2048`), heads (`4`, `8`), and sequence lengths (`256`, `512`) for a total of 72 unique combinations per model architecture.
- **Evaluation Metric:** The models are compared based on their test loss, with a custom function `evaluate_models` indicating which model performed better.
- **Output:** Results are recorded in a CSV file, including each combination's average loss for both models, similarity scores, and training times.

**Usage:**

To run the grid search, ensure your configuration file is correctly set up, then execute the script with the path to your config file as an argument:

```bash
python3 ../../src/grid_search.py configs/user_configs/<YOUR_CONFIG_HERE>.yaml
```

### Hugging Face Integration

This feature introduces custom models built upon the Hugging Face Transformers library, enabling the incorporation of RetNet and Transformer architectures into a wide range of NLP tasks. Leveraging Hugging Face's `PreTrainedModel` class, we've developed `RetNetModelHF` and `TransformerModelHF` classes to seamlessly integrate with Hugging Face's ecosystem, facilitating easy model training, evaluation, and deployment.

**Code Overview:**

- **`RetNetModelHF`**: Implements the RetNet architecture as a subclass of PreTrainedModel, using Hugging Face's utilities and standards for model configuration, serialization, and compatibility with the Transformers library.
- **`TransformerModelHF`**: Implements the Transformer architecture as a subclass of PreTrainedModel, using Hugging Face's utilities and standards for model configuration, serialization, and compatibility with the Transformers library.
- **Configuration Classes**: Both models utilize specific configuration classes (`RetNetConfig` for RetNetModelHF and `DecoderConfig` for TransformerModelHF) to define model parameters, ensuring flexibility and ease of customization.

**Usage:**

To use these models within your Hugging Face-based projects, follow these steps:

1. **Initialization**: Instantiate the model with the desired configuration, which can be a predefined object, a path to a configuration file, or left as default for automatic configuration.

   ```python
   from <YOUR_MODULE> import RetNetModelHF, TransformerModelHF

   longnet_model = LongNetModelHF(config="path/to/longnet/config")
   retnet_model = RetNetModelHF(config="path/to/retnet/config")
   transformer_model = TransformerModelHF(config="path/to/transformer/config")
   ```

2. **Forward Pass**: Call the model with input data tensors to receive output predictions.

   ```python
   input_ids = ...  # Your input tensor here
   longnet_output = longnet_model(input_ids)
   retnet_output = retnet_model(input_ids)
   transformer_output = transformer_model(input_ids)
   ```

3. **Parameter Access**: Retrieve model hyperparameters for inspection or further processing.

   ```python
   longnet_params = longnet_model.get_params()
   retnet_params = retnet_model.get_params()
   transformer_params = transformer_model.get_params()
   ```

## Benchmarking for Generation Quality

We use EleutherAI's open-source language model evaluation harness to empirically evaluate our models across a suite of different NLP tasks. Run the evaluation suite as follows:
First, edit the 'tasks' parameter in the YAML file. Specify all tasks you would like to run, e.g.,
```
tasks:
  - "hellaswag"
  - "winogrande"
```
Alternatively, you can use `tasks: '*'` to run all benchmarks in the suite. These tasks will need to download if not yet stored locally at `~/.cache/huggingface/datasets/`. Navigate to the `slurm/run_eval.sh`, copy the script, and substitute your yaml file for the placeholder. Finally, execute:

```
# Activate environment, if using one
mamba activate <YOUR_ENV_HERE>

cd /301r_retnet/slurm/

# Give your file a descriptive name, (e.g., 'retnet_40540_run_eval.sh')
cp run_eval.sh user_slurm/<NAME_OF_NEW_FILE>.sh

bash <NAME_OF_NEW_FILE>/.sh
```
Results will be sent to a CSV.

## Carbon Emissions Tracking

This project uses [CodeCarbon](https://github.com/mlco2/codecarbon) to track emissions in offline mode, meaning no data is reported to the public API. This outputs a csv file with stats with duration in seconds and power consumption measured in kilowatts. Carbon emissions (denoted by '_emissions_') is a calculation of the specified energy consumption profile and the energy consumed measured in kg.

Sample output:

| timestamp           | project_name | run_id                              | duration (sec) | emissions (kg) | emissions_rate | cpu_power (kW) | gpu_power (kW) | ram_power (kW) | cpu_energy (kW) | gpu_energy (kW) | ram_energy (kW) | energy_consumed (kW) | country_name | country_iso_code | region | cloud_provider | cloud_region | os                                           | python_version | codecarbon_version | cpu_count | cpu_model                        | gpu_count | gpu_model          | longitude | latitude | ram_total_size | tracking_mode | on_cloud | pue |
|---------------------|--------------|-------------------------------------|----------|-----------|----------------|-----------|-----------|-----------|------------|------------|------------|-----------------|--------------|------------------|--------|----------------|--------------|---------------------------------------------|----------------|-------------------|-----------|----------------------------------|-----------|-------------------|-----------|----------|----------------|---------------|----------|-----|
| 2024-03-13T16:52:45 | codecarbon   | dea8afbd-973d-4396-b103-f09eb94c1457 | 180.817  | 0.02549   | 0.000141       | 140.0     | 1066.512  | 24.0      | 0.007032   | 0.048665   | 0.0012     | 0.056896        | USA          | USA              | Utah   | gcp            | us-west3    | Linux-3.10.0-1160.108.1.el7.x86_64-x86_64-with-glibc2.17 | 3.11.6         | 2.3.4             | 8         | AMD EPYC 7763 64-Core Processor | 8         | 8 x NVIDIA A100-SXM4-80GB |           |          | 64             | machine       | Y        | 1.0 |

## Acknowledgments

We extend our heartfelt gratitude to the following individuals and institutions for their invaluable contributions to this project:

**Nancy Fulda**: Our esteemed instructor, whose guidance and insights have significantly shaped the direction and execution of this research.

**BYU Office of Research Computing**: For providing the computational resources and support that were instrumental in conducting our experiments and analyses.

**Authors of the Original RetNet Paper**: We acknowledge their contributions to novel encoder architectures, which guided our investigations into RetNet and Transformers and offered a foundational framework for our research.

**Microsoft TorchScale Team**: For developing and maintaining the TorchScale framework, which served as the foundational architecture for our project, enabling us to push the boundaries of what's possible in deep learning research.

## Citations

> [!NOTE]
> Our paper is awaiting publication and our full citation will be given soon.
