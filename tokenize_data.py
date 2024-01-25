import datasets

from argparse import ArgumentParser
from math import isclose
from os import environ
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

def tokenize_data(
        data_dir: str,
        dataset_name: str,
        datasets_dir: str,
        seq_len: int,
        tokenized_data_name: str,
        tokenizer_folder: str,
        dataset_subset: str=None,
        rand_seed: int=None,
        splits: list[float]=[0.7, 0.2, 0.1],
        text_feature: str="text") -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    
    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert isclose(sum(splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        f"datasets must sum up to 1 ({' + '.join(map(str, splits))} != 1)!"
    
    # Retrieve iterators for each split of the dataset
    print(f"Datasets dir: {datasets_dir}")
    entire_dataset = datasets.load_from_disk(Path(datasets_dir) / dataset_name)

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
    tokenized_dataset_dir = Path(data_dir) / "tokenized_datasets" / dataset_name
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {tokenized_dataset_dir}")
    for key, value in entire_dataset.items():
        filename = key + '.parquet'
        value.to_parquet(tokenized_dataset_dir / filename)


if __name__ == "__main__":
    # Get arguments
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory where all data expect datasets are saved.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of Hugging Face dataset.")
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Configuration/subset of dataset to use.")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        required=True,
        help="Relative path from base of repository to directory in which to download the dataset.")
    parser.add_argument(
        "--rand-seed",
        type=int,
        default=None,
        help="Seed used during dataset shuffling, ignored if None.")
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Context window/sequence length.")
    parser.add_argument(
        "--splits", 
        type=float, 
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument(
        "--text-feature",
        type=str,
        default="text",
        help="Name of the feature/column of the dataset to use.")
    parser.add_argument(
        "--tokenized-data-name",
        type=str,
        required=True,
        help="Name of tokenized data.")
    parser.add_argument(
        "--tokenizer-folder",
        type=str,
        required=True,
        help="Folder to save tokenizer to.")

    args = parser.parse_args()
    tokenize_data(**vars(args))
