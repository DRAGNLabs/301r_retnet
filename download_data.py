import datasets
import sys
import yaml

from pathlib import Path
from utils import Struct

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

    print("Beginning download")
    print(f"File path: {dataset_dir}")
    print(f"Data name: {config.dataset_name}")
    print(f"Data subset: {config.dataset_subset}")
    dataset = datasets.load_dataset(
            path=config.dataset_name,
            name=config.dataset_subset,
            split="all",
            trust_remote_code=True,
            num_proc=config.num_proc)

    # Check if dataset is of type datasets.arrow_dataset.Dataset
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        filename = config.dataset_subset + ".parquet"
        dataset.to_parquet(dataset_dir / filename)
    else:
        raise Exception("Dataset is not of type " + \
            "datasets.arrow_dataset.Dataset or " + \
            "datasets.dataset_dict.DatasetDict")
    print("Download completed.")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    download_data(config)
