import dask
dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
import sys
sys.path.append("..")
import yaml

from huggingface_hub import login
from pathlib import Path
from utils import Struct

def download_data(config):
    """ 
    Download dataset from Hugging Face through the 
    HuggingFace file system, and save as parquet.

    There are a variety of ways to download datasets 
    from HuggingFace, and the best way depends on the dataset.

    This function uses the HuggingFace file system to download the data directly into
    Dask dataframes, which then save the data to disk in parquet format.

    This may work for smaller datasets, but for larger datasets,
    it is better to use curl/wget to download the data directly to disk. 
    Or, clone a git repository that contains the data.
    """
    # Login to HF Hub. You will need a token to do this.
    login()
    # Create folder to save this dataset's files in
    dataset_dir = Path(config.raw_dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Beginning download")
    print(f"File path: {dataset_dir}")
    print(f"Data name: {config.dataset_name}")
    print(f"Data subset: {config.dataset_subset}")

    # This path may need to be modified depending on the dataset.
    fs_path = f"hf://datasets/{config.dataset_name}/{config.dataset_subset}"

    # Can also use 'read_json', etc.; depends on HF repo.
    dataset = dd.read_parquet(fs_path)

    print("Saving to disk")
    dataset.to_parquet(dataset_dir)
    print("Download completed.")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    download_data(config)
