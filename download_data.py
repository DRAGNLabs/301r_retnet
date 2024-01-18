from datasets import load_dataset as load_ds
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT_NAME = "301r_retnet"

def download_data(
        dataset_name: str,
        dataset_subset: str,
        dataset_dir: str):
    """ Download dataset from Hugging Face.

    It is useful to download the dataset before trying to train the model when
    the training will take place in a location without access to the internet.

    Args:
        dataset_name (str): Name of Hugging Face dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
        dataset_dir (str): Absolute path to directory in which to download the
            dataset. If None, will default to the "data" directory at the root
            of the repository.
    """
    # Get path of repository root folder
    repo_root_dir = Path(__file__)
    while REPO_ROOT_NAME not in repo_root_dir.name:
        repo_root_dir = repo_root_dir.parent

    if args.dataset_dir is None:
        data_dir = repo_root_dir / "data"
    else:
        data_dir = args.dataset_dir

    print("Beginning download")
    entire_dataset = load_ds(
        path=args.dataset_name,
        name=args.dataset_subset,
        split="all",
        cache_dir=data_dir,
        trust_remote_code=True)

if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(prog="Data Downloader")

    parser.add_argument("--dataset-name", type=str, required=True,
        default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, required=True,
        default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, default=None,
        help="Directory to save dataset to.")

    args = parser.parse_args()

    download_data(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_dir=args.dataset_dir)
