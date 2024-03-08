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
import dask.bag as db
from dask.distributed import LocalCluster

import pandas as pd

def download_data(config):

    dataset_dir = Path('/home/jo288/compute/retnet/data/tokenized_dataset/c4/train')
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # Start timer
    start = time.time()
    #dataset = dd.read_json("/home/jo288/compute/retnet/data/datasets/c4_dask/c4/en/*.json.gz")
    #dataset = db.read_text("/home/jo288/compute/retnet/data/datasets/c4_dask/c4/en/*.json.gz")
    test_dataset = dd.read_parquet(path=dataset_dir / '*.parquet')
    test_dataset.compute()
    # Load parquet files into pandas
    #data_files = [str(f) for f in dataset_dir.glob('*.parquet')]
    #df = pd.concat((pd.read_parquet(f, engine = 'pyarrow') for f in data_files))
    # Stop timer
    end = time.time()
    print(f"Time to load data: {end - start}")



if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    download_data(config)
