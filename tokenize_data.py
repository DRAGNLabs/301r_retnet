import datasets
import sys
import yaml

from utils import Struct
from os import environ
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from argparse import ArgumentParser
from pathlib import Path

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_data(
        tokenized_data_name: str,
        tokenized_data_folder: str,
        tokenizer_folder: str,
        seq_len: int,
        dataset_dir: str,
        dataset_subset: str,
        text_feature: str,
        splits: list[float],
        rand_seed: int) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    
    # Retrieve iterators for each split of the dataset
    print(f'Data dir: {dataset_dir}')
    entire_dataset = datasets.load_dataset(
        "parquet",
        data_files=str(Path(dataset_dir) / f"{dataset_subset}.parquet"),
        split="all")

    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    filter_fun = lambda inst_dict : bool(inst_dict[text_feature].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Split into training, validation, and testing datasets
    train_validtest = entire_dataset.train_test_split(
        train_size=splits[0],
        shuffle=True,
        seed=rand_seed)
    valid_test = train_validtest["test"].train_test_split(
        train_size=splits[1] / (splits[1] + splits[2]),
        shuffle=True,
        seed=rand_seed)
    entire_dataset = datasets.DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_folder)

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
        tokenizer(
            instances_dict[text_feature],
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_token_type_ids=False,
            return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True)

    # Drop now unnecessary text_feature column
    entire_dataset = entire_dataset.remove_columns(column_names=text_feature)

    #This code saves the now tokenized dataset as a .parquet folder, making a folder in the data directory called tokenized if one does not already exist.
    tokenized_data_path = Path(tokenized_data_folder)
    print(f'Saving tokenized data to {tokenized_data_path}')
    if not tokenized_data_path.exists():
        tokenized_data_path.mkdir(parents=True)

    if isinstance(entire_dataset, datasets.arrow_dataset.Dataset):
        entire_dataset.to_parquet(tokenized_data_path  / f'{tokenized_data_name}.parquet')
    elif isinstance(entire_dataset, datasets.dataset_dict.DatasetDict):
        for key, value in entire_dataset.items():
            filename = key + '.parquet'
            value.to_parquet(tokenized_data_path / filename)
    else:
        print('Dataset is not of type datasets.arrow_dataset.Dataset or datasets.dataset_dict.DatasetDict')


if __name__ == "__main__":
    # Get arguments

    args = sys.argv
    config_path =args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config.tokenized_data_name, config.tokenized_data_dir, config.tokenizer_path, config.seq_len, config.raw_dataset_dir, config.dataset_subset, config.dataset_feature, config.splits, config.rand_seed)

    

