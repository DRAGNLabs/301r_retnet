import datasets
from argparse import ArgumentParser
from pathlib import Path

def download_data(
        dataset_name: str,
        dataset_subset: str,
        dataset_root_dir: str):
    """ Download dataset from Hugging Face.

    It is useful to download the dataset before trying to train the model when
    the training will take place in a location without access to the internet.

    Args:
        dataset_name (str): Name of Hugging Face dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
        dataset_root_dir (str): Absolute path to the directory in which Hugging
            Face datasets are downloaded.
    """
    # Create folder to save this dataset's files in
    dataset_dir = Path(dataset_root_dir) / dataset_name
    dataset_dir.mkdir(parents=True)

    print("Beginning download")
    print(f"File path: {dataset_dir}")
    print(f"Data name: {dataset_name}")
    print(f"Data subset: {dataset_subset}")
    dataset = datasets.load_dataset(
            path=dataset_name,
            name=dataset_subset,
            split="all",
            trust_remote_code=True)
    
    # check if dataset is of type datasets.arrow_dataset.Dataset
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        filename = dataset_subset + ".parquet"
        dataset.to_parquet(dataset_dir / filename)
    else:
        raise Exception("Dataset is not of type " + \
            "datasets.arrow_dataset.Dataset or datasets.dataset_dict.DatasetDict")
    print("Download completed.")


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(prog="Data Downloader")

    parser.add_argument("--dataset-name", type=str, required=True,
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, required=True,
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")

    args = parser.parse_args()

    download_data(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_root_dir=args.dataset_dir)
