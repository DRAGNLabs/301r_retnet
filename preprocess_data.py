import datasets
import sys
import time
import yaml

from datasets import DatasetDict
from pathlib import Path
from utils import Struct

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
import dask_ml

def download_data(config):
    """ Download dataset from Hugging Face.

    It is useful to download the dataset before trying to train the model when
    the training will take place in a location without access to the internet.

    Args:
        dataset_name (str): Name of Hugging Face dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
        datasets_dir (str): Absolute path to the directory in which Hugging Face
            datasets are downloaded.
    """
    # Create folder to save this dataset's files in
    dataset_dir = Path(config.raw_dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # Start timer
    start = time.time()
    dataset = dd.read_json("/home/jo288/compute/retnet/data/datasets/c4_dask/c4/en/*.json.gz")
    # Stop timer
    end = time.time()
    print(f"Time to load data: {end - start}")

    start = time.time()
    print('FILTERING')
    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    # filter_fun = lambda inst_dict : bool(
    #     inst_dict[config.dataset_feature].strip())

    # Filter out undesired data instances
    # dataset = dataset.map_partitions(filter_fun)a

    # Filter out rows with only whitespace
    dataset = dataset[dataset[config.dataset_feature].str.strip() != '']

    end = time.time()
    print(f"Time to filter data: {end - start}")

    start = time.time()
    print('SPLITTING')
    # Split into training, validation, and testing datasets
    train, test = dask_ml.model_selection.train_test_split(
        dataset,
        train_size=config.splits[0],
        random_state=config.rand_seed)
    test, validation = dask_ml.model_selection.train_test_split(
        test,
        train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
        random_state=config.rand_seed)
    
    end = time.time()
    print(f"Time to split data: {end - start}")
    
    # train.compute()
    # validation.compute()
    # test.compute()

    # dataset = DatasetDict({
    #     "train": train_validtest["train"],
    #     "validation": valid_test["train"],
    #     "test": valid_test["test"]})

    start = time.time()
    print('SAVING TO DISK')
    train.to_parquet(dataset_dir / 'train')
    validation.to_parquet(dataset_dir / 'validation')
    test.to_parquet(dataset_dir / 'test')
    end = time.time()
    print(f"Time to save data to disk: {end - start}")
    # Save splits to file
    # dataset.save_to_disk(
    #     dataset_dict_path=Path(config.raw_dataset_path),
    #     num_proc=config.num_proc)

    # for key, value in dataset.items():
    #     value.to_csv(dataset_dir / (key+'.csv'),
    #                  columns=[config.dataset_feature],
    #                  num_proc=config.num_proc)

    # for key, value in dataset.items():
    #     # shard dataset into smaller files
    #     num_shards = 500

    #     for i in range(num_shards):
    #         print(f"Sharding {key} dataset, shard {i} of {num_shards}")
    #         shard = value.shard(
    #             num_shards=num_shards,
    #             index=i,
    #             keep_in_memory=True)
    #         shard.to_csv(dataset_dir / (key + f'_shard_{i}.csv'), num_proc=config.num_proc)
    #     #value.to_csv(dataset_dir / (key+'.csv'), num_proc=config.num_proc) #TODO: test this

    print("Download completed.")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    download_data(config)
