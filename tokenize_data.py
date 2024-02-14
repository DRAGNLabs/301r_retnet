import datasets
import math
import sys
import yaml

from pathlib import Path
from transformers import PreTrainedTokenizerFast
from utils import Struct

def tokenize_data(config):
    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert math.isclose(sum(config.splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        "datasets must sum up to 1 " + \
        f"({' + '.join(map(str, config.splits))} != 1)!"

    # Retrieve iterators for each split of the dataset
    print(f"Datasets dir: {config.raw_dataset_path}")
    entire_dataset = datasets.load_from_disk(Path(config.raw_dataset_path))

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
        tokenizer(
            instances_dict[config.dataset_feature],
            padding="max_length",
            truncation=True,
            max_length=config.seq_len + 1,
            return_token_type_ids=False,
            return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True, num_proc=config.num_proc)

    # Drop now unnecessary text_feature column
    entire_dataset = entire_dataset.remove_columns(
        column_names=config.dataset_feature)

    # Make sure directory for tokenized dataset exists
    tokenized_dataset_dir = Path(config.tokenized_dataset_path)
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {config.tokenized_dataset_path}")
    for key, value in entire_dataset.items():
        filename = key + '.parquet'
        value.to_parquet(tokenized_dataset_dir / filename)


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config)
