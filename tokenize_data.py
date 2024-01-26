import datasets
import sys
import yaml

from utils import Struct
from os import environ
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

def tokenize_data(
        # data_dir: str,
        # dataset_name: str,
        # datasets_dir: str,
        # seq_len: int,
        tokenized_data_name: str,
        tokenized_data_dir:str,
        tokenizer_path: str,
        seq_len: int,
        raw_dataset_dir: str,
        dataset_subset: str,
        text_feature: str,
        splits: list[float],
        rand_seed: int) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    
    # Test the dataset splits add up to 1, using isclose for rounding errors
    
    # assert isclose(sum(splits), 1), \
    #     "The dataset splits for the training, validation, and testing " + \
    #     f"datasets must sum up to 1 ({' + '.join(map(str, splits))} != 1)!"
    
    # Retrieve iterators for each split of the dataset
    print(f"Datasets dir: {raw_dataset_dir}")
    entire_dataset = datasets.load_from_disk(Path(raw_dataset_dir))

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

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
    tokenized_dataset_dir = Path(raw_dataset_dir) / "tokenized_datasets"
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {tokenized_dataset_dir}")
    for key, value in entire_dataset.items():
        filename = key + '.parquet'
        value.to_parquet(tokenized_dataset_dir / filename)


if __name__ == "__main__":
    # Get arguments

    args = sys.argv
    config_path =args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config.tokenized_data_name, config.tokenized_data_dir, config.tokenizer_path, config.seq_len, config.raw_dataset_dir, config.dataset_subset,config.dataset_feature, config.splits, config.rand_seed)

    

