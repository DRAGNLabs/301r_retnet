import datasets
import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask_ml
import sys
import time
import yaml

from datasets import DatasetDict
from pathlib import Path
from utils import Struct

ProgressBar().register()

def split_data(config):
    """ 
    Filter and split the dataset into training, validation, and testing datasets.
    """
    # Create folder to save this dataset's files in
    dataset_dir = Path(config.raw_dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Read the dataset from disk
    dataset = dd.read_parquet(dataset_dir / "*.parquet")

    # Filter out rows with only whitespace
    dataset = dataset[dataset[config.dataset_feature].str.strip() != '']

    # Split into training, validation, and testing datasets
    train, test_valid = dask_ml.model_selection.train_test_split(
        dataset,
        shuffle=False, # Very expensive for large datasets
        train_size=config.splits[0],
        random_state=config.rand_seed)
    test, validation = dask_ml.model_selection.train_test_split(
        test_valid,
        shuffle=False,
        train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
        random_state=config.rand_seed)

    train.to_parquet(dataset_dir / 'train')
    validation.to_parquet(dataset_dir / 'validation')
    test.to_parquet(dataset_dir / 'test')

    print("Finished")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    split_data(config)
