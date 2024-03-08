import math
import sys
import yaml

from pathlib import Path
from transformers import PreTrainedTokenizerFast
from utils import Struct

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd

import pyarrow as pa

def tokenize_data(config):

    # Load the dataset from disk into dask
    train_dataset = dd.read_parquet(path=Path(config.raw_dataset_path) / 'train' / '*.parquet') # TODO: consider loading only the text column
    val_dataset = dd.read_parquet(path=Path(config.raw_dataset_path) / 'validation' / '*.parquet')
    test_dataset = dd.read_parquet(path=Path(config.raw_dataset_path) / 'test' / '*.parquet')
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Tokenize the datasets
    # tokenization = lambda instances_dict : \
    #     tokenizer(
    #         instances_dict[config.dataset_feature],
    #         padding="max_length",
    #         truncation=True,
    #         max_length=config.seq_len + 1,
    #         return_token_type_ids=False,
    #         return_attention_mask=False,
    #         return_tensors="np") # TODO: test this

    def tokenization_partition(partition):
        #print('HEY: ', partition[config.dataset_feature].tolist())
        tokenization_dataframe = lambda series: \
            tokenizer(
                series,
                padding="max_length",
                truncation=True,
                max_length=config.seq_len + 1,
                return_token_type_ids=False,
                return_attention_mask=False)["input_ids"]

        tokenized_data = partition[config.dataset_feature].map(tokenization_dataframe, na_action='ignore').to_frame()

        return tokenized_data

    train_dataset = train_dataset.map_partitions(tokenization_partition)
    val_dataset = val_dataset.map_partitions(tokenization_partition)
    test_dataset = test_dataset.map_partitions(tokenization_partition)
    # list df columns
    #print('columns: ', train_dataset.columns)
    # Drop now unnecessary text_feature column
    # entire_dataset = entire_dataset.remove_columns(
    #     column_names=config.dataset_feature)

    # Make sure directory for tokenized dataset exists
    tokenized_dataset_dir = Path(config.tokenized_dataset_path)
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {config.tokenized_dataset_path}")
    # for key, value in entire_dataset.items():
    #     filename = key + '.parquet'
    #     value.to_parquet(tokenized_dataset_dir / filename)
    
    train_dataset.to_parquet(tokenized_dataset_dir / 'train', schema={"text": pa.list_(pa.int64())})
    val_dataset.to_parquet(tokenized_dataset_dir / 'validation', schema={"text": pa.list_(pa.int64())})
    test_dataset.to_parquet(tokenized_dataset_dir / 'test', schema={"text": pa.list_(pa.int64())})

    print('Done!')


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config)
