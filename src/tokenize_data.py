import argparse
import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pyarrow as pa
import yaml

from pathlib import Path
from transformers import PreTrainedTokenizerFast
from utils import Struct

ProgressBar().register()

def tokenize_data(config, split):

    # Dataset path
    dataset_path = Path(config.raw_dataset_path) / split / '*.parquet'

    # Load the dataset from disk into dask
    dataset = dd.read_parquet(path=dataset_path, 
                              columns=[config.dataset_feature])
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    def tokenization_partition(partition):
        tokenization_dataframe = lambda series: \
            tokenizer(
                series,
                padding=False,
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False)["input_ids"]

        tokenized_data = partition[config.dataset_feature] \
            .map(tokenization_dataframe, na_action='ignore').to_frame()

        return tokenized_data

    dataset = dataset.map_partitions(tokenization_partition)

    # Make sure directory for tokenized dataset exists
    tokenized_dataset_dir = Path(config.tokenized_dataset_path)
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {config.tokenized_dataset_path}")
    
    dataset.to_parquet(tokenized_dataset_dir / split, 
                             schema={"text": pa.list_(pa.int64())})

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize data')
    parser.add_argument('config_path', 
                        type=str, 
                        help='Path to the config file')
    parser.add_argument('split', 
                        type=str, 
                        choices=['train', 'test', 'validation'], 
                        help='Dataset split to use')
    args = parser.parse_args()

    config_path = args.config_path
    split = args.split

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config, split)
