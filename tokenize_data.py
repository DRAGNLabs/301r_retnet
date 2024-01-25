import datasets

from argparse import ArgumentParser
from os import environ
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_data(
        tokenized_data_name: str,
        tokenized_data_folder: str,
        tokenizer_folder: str,
        dataset_name: str,
        seq_len: int,
        datasets_dir: str,
        dataset_subset: str=None,
        text_feature: str="text",
        splits: list[float]=[0.7, 0.2, 0.1],
        rand_seed: int=None) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    
    # Retrieve iterators for each split of the dataset
    print(f"Datasets dir: {datasets_dir}")
    entire_dataset = datasets.load_dataset(
        "parquet",
        data_files=str(Path(datasets_dir) / dataset_name / f"{dataset_subset}.parquet"),
        split="all")

    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    filter_fun = lambda inst_dict : bool(inst_dict[text_feature].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Split into training, validation, and testing datasets
    train_validtest = entire_dataset.train_test_split(
        train_size=splits[0],
        shuffle=True,
        seed=rand_seed)
    valid_test = train_validtest["test"].train_test_split(
        train_size=splits[1] / (splits[1] + splits[2]),
        shuffle=True,
        seed=rand_seed)
    entire_dataset = datasets.DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_folder)

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
        tokenizer(
            instances_dict[text_feature],
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_token_type_ids=False,
            return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True)

    # Drop now unnecessary text_feature column
    entire_dataset = entire_dataset.remove_columns(column_names=text_feature)

    #This code saves the now tokenized dataset as a .parquet folder, making a folder in the data directory called tokenized if one does not already exist.
    path = Path(tokenized_data_folder)
    print(f"Saving tokenized data to {path}")
    if not path.exists():
        path.mkdir(parents=True)

    if isinstance(entire_dataset, datasets.arrow_dataset.Dataset):
        entire_dataset.to_parquet(path / f"{tokenized_data_name}.parquet")
    elif isinstance(entire_dataset, datasets.dataset_dict.DatasetDict):
        for key, value in entire_dataset.items():
            filename = key + '.parquet'
            value.to_parquet(path / filename)
    else:
        print("Dataset is not of type datasets.arrow_dataset.Dataset or datasets.dataset_dict.DatasetDict")


if __name__ == "__main__":
    # Get arguments
    parser = ArgumentParser()
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of Hugging Face dataset.")
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Configuration/subset of dataset to use.")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        required=True,
        help="Relative path from base of repository to directory in which to download the dataset.")
    parser.add_argument(
        "--rand-seed",
        type=int,
        default=None,
        help="Seed used during dataset shuffling, ignored if None.")
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Context window/sequence length.")
    parser.add_argument(
        "--splits", 
        type=float, 
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument(
        "--text-feature",
        type=str,
        default="text",
        help="Name of the feature/column of the dataset to use.")
    parser.add_argument(
        "--tokenized-data-folder",
        type=str,
        required=True,
        help="Folder to save tokenizered data to.")
    parser.add_argument(
        "--tokenized-data-name",
        type=str,
        required=True,
        help="Name of tokenized data.")
    parser.add_argument(
        "--tokenizer-folder",
        type=str,
        required=True,
        help="Folder to save tokenizer to.")

    args = parser.parse_args()
    tokenize_data(**vars(args))
