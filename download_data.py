from datasets import (
    load_dataset as load_ds)
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT_NAME = "301r_retnet"

def main():
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
        prog="Data Downloader")

    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--dataset-dir", type=str, default="data",
        help="Directory to save dataset to.")

    args = parser.parse_args()

    # Get path of repository root folder
    repo_root_dir = Path("/grphome/grp_retnet/compute") # This is hard coded. You can pass in your own path too
    # while REPO_ROOT_NAME not in repo_root_dir.name:
    #     repo_root_dir = repo_root_dir.parent
    
    data_dir=repo_root_dir / args.dataset_dir

    print('Beginning download')
    entire_dataset = load_ds(
            path=args.dataset_name,
            name=args.dataset_subset,
            split="all",
            cache_dir=data_dir,
            trust_remote_code=True)

if __name__ == "__main__":
    main()